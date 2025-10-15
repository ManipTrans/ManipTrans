import numpy as np
import torch.nn as nn
import torch
from copy import deepcopy
from gym import spaces
import rl_games.common.divergence as divergence
from rl_games.common.extensions.distributions import CategoricalMasked
from lib.utils.torch_utils import recurse_freeze, freeze_batchnorm_stats
from lib.rl.moving_avg import RunningMeanStd, RunningMeanStdObs
from .models import BaseModel, BaseModelNetwork


class ModelA2CContinuousLogStdResRH(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, "a2c")
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

            self.base_model_obs_shape = kwargs["base_model_obs_shape"]
            base_model_kwargs = deepcopy(kwargs)
            base_model_kwargs["obs_shape"] = spaces.Dict(
                {
                    k: spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(v,),
                    )
                    for k, v in self.base_model_obs_shape.items()
                }
            )
            self.base_model = BaseModelNetwork(**base_model_kwargs)
            if kwargs["rh_base_model_checkpoint"] is not None:
                base_model_ckp = torch.load(kwargs["rh_base_model_checkpoint"])
                self.base_model.load_state_dict(
                    {k: v for k, v in base_model_ckp["model"].items() if "a2c_network" not in k}
                )
            else:
                from termcolor import cprint

                cprint("\nWARNING: The first-stage imitator model is not loaded, using random weights instead.", "red")
                cprint("WARNING: This may slow convergence. Consider pretraining the imitator model.\n", "red")
            recurse_freeze(self.base_model)
            freeze_batchnorm_stats(self.base_model)
            self.base_model.eval()

        def train(self, mode: bool = True):
            self.training = mode
            for name, module in self.named_children():
                if name in ["base_model"]:
                    module.train(False)  # always eval
                else:
                    module.train(mode)
            return self

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def load_state_dict(self, state_dict):
            return super().load_state_dict(state_dict)

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)
            base_model_obs = {k: deepcopy(v[:, : self.base_model_obs_shape[k]]) for k, v in input_dict["obs"].items()}
            base_model_obs = self.base_model.norm_obs(base_model_obs)
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            mu, logstd, value, states, base_action = self.a2c_network(
                input_dict, {"obs": base_model_obs, **{k: v for k, v in input_dict.items() if k != "obs"}}
            )

            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    "neglogpacs": torch.squeeze(neglogp),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                    "base_actions": base_action,
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return (
                0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
                + logstd.sum(dim=-1)
            )


class ModelA2CContinuousLogStdResLH(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, "a2c")
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

            self.base_model_obs_shape = kwargs["base_model_obs_shape"]
            base_model_kwargs = deepcopy(kwargs)
            base_model_kwargs["obs_shape"] = spaces.Dict(
                {
                    k: spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(v,),
                    )
                    for k, v in self.base_model_obs_shape.items()
                }
            )
            self.base_model = BaseModelNetwork(**base_model_kwargs)
            if kwargs["lh_base_model_checkpoint"] is not None:
                base_model_ckp = torch.load(kwargs["lh_base_model_checkpoint"])
                self.base_model.load_state_dict(
                    {k: v for k, v in base_model_ckp["model"].items() if "a2c_network" not in k}
                )
            else:
                from termcolor import cprint

                cprint("\nWARNING: The first-stage imitator model is not loaded, using random weights instead.", "red")
                cprint("WARNING: This may slow convergence. Consider pretraining the imitator model.\n", "red")
            recurse_freeze(self.base_model)
            freeze_batchnorm_stats(self.base_model)
            self.base_model.eval()

        def train(self, mode: bool = True):
            self.training = mode
            for name, module in self.named_children():
                if name in ["base_model"]:
                    module.train(False)  # always eval
                else:
                    module.train(mode)
            return self

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def load_state_dict(self, state_dict):
            return super().load_state_dict(state_dict)

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)
            base_model_obs = {k: deepcopy(v[:, : self.base_model_obs_shape[k]]) for k, v in input_dict["obs"].items()}
            base_model_obs = self.base_model.norm_obs(base_model_obs)
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            mu, logstd, value, states, base_action = self.a2c_network(
                input_dict, {"obs": base_model_obs, **{k: v for k, v in input_dict.items() if k != "obs"}}
            )

            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    "neglogpacs": torch.squeeze(neglogp),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                    "base_actions": base_action,
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return (
                0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
                + logstd.sum(dim=-1)
            )


class ModelA2CContinuousLogStdResBiH(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, "a2c")
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

            self.base_model_obs_shape = kwargs["base_model_obs_shape"]
            base_model_kwargs = deepcopy(kwargs)
            base_model_kwargs["obs_shape"] = spaces.Dict(
                {
                    k: spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(v,),
                    )
                    for k, v in self.base_model_obs_shape.items()
                }
            )
            self.rh_base_model = BaseModelNetwork(**base_model_kwargs)
            self.lh_base_model = BaseModelNetwork(**base_model_kwargs)
            if kwargs["rh_base_model_checkpoint"] is not None:
                rh_base_model_ckp = torch.load(kwargs["rh_base_model_checkpoint"])
                self.rh_base_model.load_state_dict(
                    {k: v for k, v in rh_base_model_ckp["model"].items() if "a2c_network" not in k}
                )
            else:
                from termcolor import cprint

                cprint("\nWARNING: The first-stage imitator model is not loaded, using random weights instead.", "red")
                cprint("WARNING: This may slow convergence. Consider pretraining the imitator model.\n", "red")
            recurse_freeze(self.rh_base_model)
            freeze_batchnorm_stats(self.rh_base_model)
            self.rh_base_model.eval()
            if kwargs["lh_base_model_checkpoint"] is not None:
                lh_base_model_ckp = torch.load(kwargs["lh_base_model_checkpoint"])
                self.lh_base_model.load_state_dict(
                    {k: v for k, v in lh_base_model_ckp["model"].items() if "a2c_network" not in k}
                )
            else:
                from termcolor import cprint

                cprint("\nWARNING: The first-stage imitator model is not loaded, using random weights instead.", "red")
                cprint("WARNING: This may slow convergence. Consider pretraining the imitator model.\n", "red")
            recurse_freeze(self.lh_base_model)
            freeze_batchnorm_stats(self.lh_base_model)
            self.lh_base_model.eval()

        def train(self, mode: bool = True):
            self.training = mode
            for name, module in self.named_children():
                if name in ["base_model", "lh_base_model", "rh_base_model"]:
                    module.train(False)  # always eval
                else:
                    module.train(mode)
            return self

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def load_state_dict(self, state_dict):
            return super().load_state_dict(state_dict)

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)
            rh_base_model_obs = {
                k: deepcopy(v[:, : self.base_model_obs_shape[k]]) for k, v in input_dict["obs"].items()
            }
            rh_base_model_obs = self.rh_base_model.norm_obs(rh_base_model_obs)
            lh_base_model_obs = {
                k: deepcopy(v[:, v.shape[1] // 2 : v.shape[1] // 2 + self.base_model_obs_shape[k]])
                for k, v in input_dict["obs"].items()
            }
            lh_base_model_obs = self.lh_base_model.norm_obs(lh_base_model_obs)
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            mu, logstd, value, states, base_action = self.a2c_network(
                input_dict,
                {"obs": rh_base_model_obs, **{k: v for k, v in input_dict.items() if k != "obs"}},
                {"obs": lh_base_model_obs, **{k: v for k, v in input_dict.items() if k != "obs"}},
            )

            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    "neglogpacs": torch.squeeze(neglogp),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                    "base_actions": base_action,
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return (
                0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
                + logstd.sum(dim=-1)
            )