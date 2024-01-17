import argparse
from hydra import initialize, compose
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import traceback
import cv2
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from calvin_agent.utils.utils import add_text, format_sftp_path
from calvin_env.envs.play_table_env import get_env

sys.path.append('/home/amete7/diffusion_dynamics/diff_skill/code')
from model import SkillAutoEncoder
from gpt_prior_global import GPT, GPTConfig

logger = logging.getLogger(__name__)

EP_LEN = 192
NUM_SEQUENCES = 2
ACTION_HORIZON = 32
DEFAULT_LOG = "/home/amete7/calvin/calvin_models/calvin_agent/evaluation/eval_log"
SAVE_ROLLOUT_VIDEO = False
EXP_NAME = "calvin_agent"

def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


class CustomModel(CalvinBaseModel):
    def __init__(self,config_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_indices = 8
        self.dummy_index = 1001

        model_name = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_model = self.clip_model.to(self.device)

        model_cfg = OmegaConf.load(config_path)
        model_ckpt = model_cfg.paths.model_weights_path
        priot_ckpt = model_cfg.paths.prior_weights_path

        gpt_config = GPTConfig(vocab_size=model_cfg.prior.vocab_size, block_size=model_cfg.prior.block_size, output_dim=model_cfg.prior.output_dim, discrete_input=True)
        self.gpt_prior_model = GPT(gpt_config).to(self.device)
        state_dict = torch.load(priot_ckpt, map_location='cuda')
        self.gpt_prior_model.load_state_dict(state_dict)
        self.gpt_prior_model = self.gpt_prior_model.to(self.device)
        self.gpt_prior_model.eval()
        print('gpt_prior_model_loaded')

        self.decoder = SkillAutoEncoder(model_cfg.model)
        state_dict = torch.load(model_ckpt, map_location='cuda')
        self.decoder.load_state_dict(state_dict)
        self.decoder = self.decoder.to(self.device)
        self.decoder.eval()
        print('decoder_loaded')

    def reset(self,lang_annotation):
        """
        This is called
        """
        self.lang_emb = self.get_language_features(lang_annotation)
    
    def get_clip_features(self,image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features

    def get_language_features(self,text):
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            language_features = self.clip_model.get_text_features(**inputs)
        return language_features

    def get_indices(self, attach_emb):
        indices = [self.dummy_index]
        for _ in range(self.max_indices):
            x = torch.tensor(indices).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.gpt_prior_model(x, None, attach_emb, [0,0])
            next_token = logits[0,-1,:].argmax().item()
            indices.append(next_token)
        return torch.tensor(indices[1:]).unsqueeze(0).to(self.device)

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        observation = obs
        # print(observation['rgb_obs'])
        front_rgb = observation['rgb_obs']['rgb_static']
        gripper_rgb = observation['rgb_obs']['rgb_gripper']
        robot_state = observation['robot_obs']
        # print(robot_state.shape,'robot_state_shape')
        robot_state = np.concatenate([robot_state[:6],[robot_state[14]]])
        robot_state = torch.tensor(robot_state).unsqueeze(0).to(self.device)
        front_emb = self.get_clip_features(front_rgb)
        gripper_emb = self.get_clip_features(gripper_rgb)
        
        init_emb = torch.cat((front_emb,gripper_emb,robot_state),dim=-1).float().to(self.device)
        attach_emb = (self.lang_emb,init_emb)

        with torch.no_grad():
            indices = self.get_indices(attach_emb)
        # print(indices,'indices')
        with torch.no_grad():
            z = self.decoder.vq.indices_to_codes(indices)
            action = self.decoder.decode(z, init_emb).squeeze(0).cpu().numpy()
        action[:,-1] = (((action[:,-1] >= 0) * 2) - 1).astype(int)
        return action

def evaluate_policy(model, env, epoch=0, eval_log_dir=DEFAULT_LOG, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch, exp=EXP_NAME)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset(lang_annotation)
    start_info = env.get_info()
    save_video = SAVE_ROLLOUT_VIDEO
    if save_video:
        output_video_path = f'rollout_{subtask}.mp4'
        frame_size = (200,200)
        fps = 15
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'xvid' for MP4, 'MJPG' for AVI
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for step in range(EP_LEN//ACTION_HORIZON):
        actions = model.step(obs, lang_annotation)
        for timestep in range(ACTION_HORIZON):
            action_to_take = actions[timestep].copy()
            action_to_take = ((action_to_take[0],action_to_take[1],action_to_take[2]),(action_to_take[3],action_to_take[4],action_to_take[5]),(action_to_take[-1],))
            obs, _, _, current_info = env.step(action_to_take)
            if save_video:
                rgb = env.render(mode="rgb_array")[:,:,::-1]
                video_writer.write(rgb)

            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if debug:
                    print(colored("success", "green"), end=" ")
                return True
    if save_video:
        video_writer.release()
        print(f"Video saved to {output_video_path}")
    if debug:
        print(colored("fail", "red"), end=" ")
    return False


def main(env_cfg):
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")

    parser.add_argument("--config_path", default="/home/amete7/calvin/calvin_models/conf/diff_skill_config.yaml", type=str, help="Path to model config file.")
    args = parser.parse_args()

    # evaluate a custom model
    if args.custom_model:
        model = CustomModel(args.config_path)
        env = hydra.utils.instantiate(env_cfg.env)
        evaluate_policy(model, env, debug=args.debug)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = get_epoch(checkpoint)
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


if __name__ == "__main__":
    # main()
    with initialize(config_path="../../../calvin_env/conf/"):
        # print("config path:")
        env_cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
        env_cfg.env["use_egl"] = False
        env_cfg.env["show_gui"] = False
        env_cfg.env["use_vr"] = False
        env_cfg.env["use_scene_info"] = True
    try:
        # Your code that may raise an exception here
        main(env_cfg)
    except Exception as e:
        # Handle the exception or simply do nothing to suppress the local variable display
        traceback.print_exc()
