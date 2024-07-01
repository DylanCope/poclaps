from poclaps.simple_gridworld_game import (
    SimpleGridWorldGame,
    EnvParams as SimpleGridWorldEnvParams,
)
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from chex import Array
from typing import NamedTuple


class Transition(NamedTuple):
    env_state: struct.PyTreeNode
    done: Array
    action: Array
    message: Array
    reward: Array
    log_prob: Array
    obs: Array
    info: dict
    episode_id: int


class SimpleGridWorldCommPolicy:
    """
    """

    def __init__(self, seed: int, env_params: SimpleGridWorldEnvParams):
        self.seed = seed
        self.env_params = env_params
        self.n_msgs = env_params.grid_size * env_params.grid_size
        grid_indices = list(range(self.n_msgs))
        np.random.seed(seed)
        np.random.shuffle(grid_indices)
        self.msg_map = dict(enumerate(grid_indices))
        self.mapping = jnp.array(list(self.msg_map.values()))

    def get_msg(self, goal_pos: jnp.array) -> int:
        pos_idx = goal_pos[0] * self.env_params.grid_size + goal_pos[1]
        return self.mapping[pos_idx]


def rollout_with_msgs(env: SimpleGridWorldGame,
                      env_params: SimpleGridWorldEnvParams,
                      policy: callable,
                      steps: int,
                      n_envs: int = 4,
                      rng: jnp.array = None,
                      comm_policy_seed: int = 0,
                      rollout_state=None):

    comm_policy = SimpleGridWorldCommPolicy(comm_policy_seed, env_params)

    @jax.jit
    def _env_step(rollout_state, _):
        env_state, last_obs, rng, ep_ids = rollout_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, _ = policy(last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        msg = jax.lax.map(
            lambda g: comm_policy.get_msg(g),
            env_state.env_state.goal_pos
        )

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, n_envs)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, action, env_params)
        ep_ids = jnp.where(done, ep_ids + n_envs, ep_ids)

        transition = Transition(
            env_state, done, action, msg, reward, log_prob, last_obs, info, ep_ids
        )
        rollout_state = (env_state, obsv, rng, ep_ids)
        return rollout_state, transition

    if rollout_state is None:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, n_envs)
        obsv, env_state = jax.vmap(env.reset,
                                   in_axes=(0, None))(reset_rng, env_params)
        ep_ids = jnp.arange(n_envs)
        rollout_state = (env_state, obsv, rng, ep_ids)

    rollout_state, traj_batch = jax.lax.scan(
        _env_step, rollout_state, None, steps
    )

    metrics = {}

    metrics['mean_reward'] = (
        (traj_batch.info['returned_episode_returns'] * traj_batch.info['returned_episode']).sum()
        / traj_batch.info['returned_episode'].sum()
    )

    metrics['mean_episode_len'] = (
        (traj_batch.info['returned_episode_lengths'] * traj_batch.info['returned_episode']).sum()
        / traj_batch.info['returned_episode'].sum()
    )

    metrics['n_episodes'] = traj_batch.info["returned_episode"].sum()

    return rollout_state, traj_batch, metrics
