import gym
import torch
import wandb
from rl_games.common.algo_observer import AlgoObserver

from lib.utils.utils import retry
from lib.utils.reformat import omegaconf_to_dict


class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb

        wandb_unique_id = f"uid_{experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")

        cfg = self.cfg

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method="fork"),
            )

            if cfg.wandb_logcode_dir:
                wandb.run.log_code(root=cfg.wandb_logcode_dir)
                print("wandb running directory........", wandb.run.dir)

        print("Initializing WandB...")
        try:
            init_wandb()
        except Exception as exc:
            print(f"Could not initialize WandB! {exc}")

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)


class WandbVideoCaptureWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        n_parallel_recorders: int = 1,
        n_successful_videos_to_record: int = 50,
    ):
        super().__init__(env)
        n_parallel_recorders = min(n_parallel_recorders, env.num_envs)
        self._n_recorders = n_parallel_recorders
        self._videos = [[] for _ in range(n_parallel_recorders)]
        self._rcd_idxs = [i for i in range(env.num_envs) if i % (env.num_envs // n_parallel_recorders) == 0][
            :n_parallel_recorders
        ]
        self._n_video_saved = 0
        self._n_successful_video_saved = 0
        self._n_successful_videos_to_record = n_successful_videos_to_record
        wandb.init()
        
    def reset(self, **kwargs):
        self._videos = [[] for _ in range(self._n_recorders)]
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        for i, idx in enumerate(self._rcd_idxs):
            self._videos[i].append(self.env.camera_obs[idx].clone())
        if torch.any(done):
            for i, idx in enumerate(self._rcd_idxs):
                if done[idx]:
                    video = torch.stack(self._videos[i])[..., :-1]  # (T, H, W, C), RGBA -> RGB
                    video = video.to(dtype=torch.uint8)
                    video = video.permute(0, 3, 1, 2).detach().cpu().numpy()  # (T, C, H, W)
                    video = wandb.Video(video, fps=10, format="mp4")
                    succeeded = self.env.success_buf
                    failed = self.env.failure_buf
                    status = "timeout"
                    if succeeded[idx]:
                        status = "success"
                        self._n_successful_video_saved += 1
                    elif failed[idx]:
                        status = "failure"
                    wandb.log({f"test_video/video-{self._n_video_saved}_{status}": video})
                    self._n_video_saved += 1
                    self._videos[i] = []
                    if self._n_successful_video_saved >= self._n_successful_videos_to_record:
                        exit()
        return obs, reward, done, info
