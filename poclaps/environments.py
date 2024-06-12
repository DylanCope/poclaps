from poclaps.simple_gridworld_game import SimpleGridWorldGame

from typing import Type

import gymnax
from gymnax.environments import environment
from gymnax.registration import registered_envs


_registered_environments = {
    "SimpleGridWorld-v0": SimpleGridWorldGame,
}


def register_env(env_name: str, env_class: Type):
    _registered_environments[env_name] = env_class


def make(env_name: str, **kwargs) -> environment.Environment:
    if env_name in registered_envs:
        return gymnax.make(env_name, **kwargs)
    elif env_name in _registered_environments:
        env = _registered_environments[env_name](**kwargs)
        return env, env.default_params.replace(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
