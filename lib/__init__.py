from omegaconf import DictConfig, OmegaConf


def _is_cuda_solver(x, y):
    if isinstance(y, int):
        return y >= 0
    if isinstance(y, str):
        if "cuda" in y.lower():
            return True
        else:
            return x.lower() in y.lower()


def get_ndof(hand_name):
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

    return DexHandFactory.create_hand(hand_name, "right").n_dofs


def get_nbody(hand_name):
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

    return DexHandFactory.create_hand(hand_name, "right").n_bodies


OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", _is_cuda_solver)
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg == "" else arg)
OmegaConf.register_new_resolver("is_both_hands", lambda dim, side: dim if not side.startswith("BiH") else dim * 2)
OmegaConf.register_new_resolver("is_sep_model", lambda mode, model: ("" if mode != "sep" else "sep_") + model)
OmegaConf.register_new_resolver(
    "is_united_model", lambda mode, side, dim: dim * 2 if mode == "united" and side.startswith("BiH") else dim
)
OmegaConf.register_new_resolver(
    "res_side",
    lambda side, model: ("res_lh_" if side == "LH" else ("res_rh_" if side == "RH" else "res_bih_")) + model,
)
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
OmegaConf.register_new_resolver("floor_divide", lambda x, y: x // y)
OmegaConf.register_new_resolver(
    "find_rl_train_config",
    lambda x: x + "PPO" if x[-3:] != "PCD" else x[:-3] + "PPO",
)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver(
    "ndof",
    lambda x: get_ndof(x),
)  # assuming right and left hands have the same number of dofs
OmegaConf.register_new_resolver(
    "nbody",
    lambda x: get_nbody(x),
)  # assuming right and left hands have the same number of bodies
