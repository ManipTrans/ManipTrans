# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import hydra

from omegaconf import DictConfig, OmegaConf
import lib


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict["params"]["config"]

    train_cfg["device"] = cfg.rl_device

    train_cfg["population_based_training"] = False
    train_cfg["pbt_idx"] = None

    train_cfg["full_experiment_name"] = cfg.get("full_experiment_name")

    print(f"Using rl_device: {cfg.rl_device}")
    print(f"Using sim_device: {cfg.sim_device}")
    print(train_cfg)

    try:
        model_size_multiplier = config_dict["params"]["network"]["mlp"]["model_size_multiplier"]
        if model_size_multiplier != 1:
            units = config_dict["params"]["network"]["mlp"]["units"]
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(
                f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}'
            )
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="config", config_path="../cfg")
def launch_rlg_hydra(cfg: DictConfig):
    import os
    from datetime import datetime

    import isaacgym
    from hydra.utils import to_absolute_path

    if cfg.display:
        import cv2
        import numpy as np

        cv2.imshow("dummy", np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)

    import maniptrans_envs
    from lib.utils.reformat import omegaconf_to_dict, print_dict
    from lib.utils.utils import set_np_formatting, set_seed

    from lib.utils.rlgames_utils import (
        RLGPUAlgoObserver,
        MultiObserver,
        ComplexObsRLGPUEnv,
    )
    from lib.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from lib.rl.runner import Runner
    from lib.rl.network_builder import DictObsBuilder
    from lib.rl.sep_network_builder import SepDictObsBuilder
    from lib.rl.models import ModelA2CContinuousLogStd, SepModelA2CContinuousLogStd
    from rl_games.algos_torch.model_builder import register_network, register_model
    from lib.utils.wandb_utils import WandbVideoCaptureWrapper
    from lib.rl.network_builder_residual_sh import ResRHDictObsBuilder, ResLHDictObsBuilder
    from lib.rl.network_builder_residual_bih import ResBiHDictObsBuilder

    from lib.rl.res_models import (
        ModelA2CContinuousLogStdResRH,
        ModelA2CContinuousLogStdResLH,
        ModelA2CContinuousLogStdResBiH,
    )

    register_model("my_continuous_a2c_logstd", ModelA2CContinuousLogStd)
    register_network("dict_obs_actor_critic", DictObsBuilder)
    register_network("sep_dict_obs_actor_critic", SepDictObsBuilder)
    register_model("sep_my_continuous_a2c_logstd", SepModelA2CContinuousLogStd)

    register_network("res_rh_dict_obs_actor_critic", ResRHDictObsBuilder)
    register_model("res_rh_my_continuous_a2c_logstd", ModelA2CContinuousLogStdResRH)
    register_network("res_lh_dict_obs_actor_critic", ResLHDictObsBuilder)
    register_model("res_lh_my_continuous_a2c_logstd", ModelA2CContinuousLogStdResLH)
    register_network("res_bih_dict_obs_actor_critic", ResBiHDictObsBuilder)
    register_model("res_bih_my_continuous_a2c_logstd", ModelA2CContinuousLogStdResBiH)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        if type(cfg.checkpoint) == str:
            cfg.checkpoint = to_absolute_path(cfg.checkpoint)
        elif type(cfg.checkpoint) == list:
            cfg.checkpoint = [to_absolute_path(cp) for cp in cfg.checkpoint]

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env():
        kwargs = dict(
            sim_device=cfg.sim_device,
            rl_device=cfg.rl_device,
            graphics_device_id=cfg.graphics_device_id,
            multi_gpu=cfg.multi_gpu,
            cfg=cfg.task,
            display=cfg.display,
            record=cfg.capture_video,
            has_headless_arg=False,
        )
        if not cfg.headless:
            assert "pcd" not in cfg.task_name.lower(), "TODO: add GUI support for PCD tasks"
        if "pcd" not in cfg.task_name.lower():
            kwargs["headless"] = cfg.headless
            kwargs["has_headless_arg"] = True
        envs = maniptrans_envs.lib.make(**kwargs)
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = WandbVideoCaptureWrapper(
                envs,
                n_parallel_recorders=cfg.n_parallel_recorders,
                n_successful_videos_to_record=cfg.n_successful_videos_to_record,
            )
        return envs

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": create_isaacgym_env,
        },
    )

    obs_spec = {}
    if "central_value_config" in cfg.rl_train.params.config:
        critic_net_cfg = cfg.rl_train.params.config.central_value_config.network
        obs_spec["states"] = {
            "names": list(critic_net_cfg.inputs.keys()),
            "concat": not critic_net_cfg.name == "complex_net",
            "space_name": "state_space",
        }

    vecenv.register("RLGPU", lambda config_name, num_actors: ComplexObsRLGPUEnv(config_name))

    rlg_config_dict = omegaconf_to_dict(cfg.rl_train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if cfg.test:
        prefix = "dump_" if cfg.save_rollouts else "test_"
        experiment_dir = os.path.join(
            "dumps",
            prefix + cfg.rl_train.params.config.name + "__{date:%m-%d-%H-%M-%S}".format(date=datetime.now()),
        )
    else:
        experiment_dir = os.path.join(
            "runs",
            cfg.rl_train.params.config.name  # + "__dataIdx_"
            # + "-".join(str(k) for k in cfg.dataIndices)
            + "__" + "{date:%m-%d-%H-%M-%S}".format(date=datetime.now()),
        )
        cfg.rl_train.params.config.full_experiment_name = experiment_dir.replace("runs/", "")
        runner.params["config"]["full_experiment_name"] = experiment_dir.replace("runs/", "")
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    runner.run(
        {
            "train": not cfg.test,
            "play": cfg.test,
            "checkpoint": cfg.checkpoint,
            "from_ckpt_epoch": cfg.from_ckpt_epoch,
            "sigma": cfg.sigma if cfg.sigma != "" else None,
            "save_rollouts": {
                "save_rollouts": cfg.save_rollouts,
                "rollout_saving_fpath": os.path.join(experiment_dir, "rollouts.hdf5"),
                "save_successful_rollouts_only": cfg.save_successful_rollouts_only,
                "num_rollouts_to_save": cfg.num_rollouts_to_save,
                "num_rollouts_to_run": cfg.num_rollouts_to_run,
                "min_episode_length": cfg.min_episode_length,
            },
        }
    )


if __name__ == "__main__":
    launch_rlg_hydra()
