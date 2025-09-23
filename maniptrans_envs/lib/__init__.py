from typing import Callable, Dict
import os
from omegaconf import DictConfig

from .envs import *


__ALL__ = ["TASK_MAP", "make"]


TASK_MAP = {
    "DexHandImitatorRH": DexHandImitatorRHEnv,
    "DexHandImitatorLH": DexHandImitatorLHEnv,
    "ResDexHandRH": DexHandManipRHEnv,
    "ResDexHandLH": DexHandManipLHEnv,
    "ResDexHandBiH": DexHandManipBiHEnv,
    "ResDexHandBiHSO": DexHandManipBiHSOEnv,
}


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def _get_rlgames_env_creator(
    # used to create the vec task
    task_config: dict,
    task_name: str,
    sim_device: str,
    rl_device: str,
    graphics_device_id: int,
    display: bool = False,
    record: bool = False,
    has_headless_arg: bool = False,
    headless: bool = True,
    # used to handle multi-gpu case
    multi_gpu: bool = False,
    post_create_hook: Callable = None,
):
    """Parses the configuration parameters for the environment task and creates a VecTask

    Args:
        task_config: environment configuration.
        task_name: Name of the task, used to evaluate based on the imported name (eg 'Trifinger')
        sim_device: The type of env device, eg 'cuda:0'
        rl_device: Device that RL will be done on, eg 'cuda:0'
        graphics_device_id: Graphics device ID.
        multi_gpu: Whether to use multi gpu
        post_create_hook: Hooks to be called after environment creation.
            [Needed to setup WandB only for one of the RL Games instances when doing multiple GPUs]
    Returns:
        A VecTaskPython object.
    """

    def create_rlgpu_env():
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """
        if multi_gpu:
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            global_rank = int(os.getenv("RANK", "0"))

            # local rank of the GPU in a node
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            world_size = int(os.getenv("WORLD_SIZE", "1"))

            print(f"global_rank = {global_rank} local_rank = {local_rank} world_size = {world_size}")

            _sim_device = f"cuda:{local_rank}"
            _rl_device = f"cuda:{local_rank}"

            task_config["rank"] = local_rank
            task_config["rl_device"] = _rl_device
        else:
            _sim_device = sim_device
            _rl_device = rl_device

        # create native task and pass custom config
        kwargs = dict(
            cfg=task_config,
            rl_device=_rl_device,
            sim_device=_sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
        )
        if has_headless_arg:
            kwargs["headless"] = headless
        env = TASK_MAP[task_name](**kwargs)

        if post_create_hook is not None:
            post_create_hook()

        return env

    return create_rlgpu_env


def make(
    *,
    sim_device: str,
    rl_device: str,
    graphics_device_id: int = -1,
    multi_gpu: bool = False,
    cfg: DictConfig,
    display: bool = False,
    record: bool = False,
    has_headless_arg: bool = False,
    headless: bool = True,
):
    assert cfg is not None
    cfg_dict = omegaconf_to_dict(cfg)

    create_rlgpu_env = _get_rlgames_env_creator(
        task_config=cfg_dict,
        task_name=cfg_dict["name"],
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        multi_gpu=multi_gpu,
        display=display,
        record=record,
        has_headless_arg=has_headless_arg,
        headless=headless,
    )
    return create_rlgpu_env()
