from poclaps.train.ckpt_cb import load_ckpt
from poclaps.train.ppo import make_train as make_ppo_train
from poclaps.train.ppo import FlattenObservationWrapper, LogWrapper, Transition
from poclaps import environments

from pathlib import Path
import yaml


run_dir = Path('outputs/2024-06-11/19-46-55')


with open(f'{run_dir}/.hydra/config.yaml') as f:
    config = yaml.safe_load(f)
    config['OUTPUT_DIR'] = run_dir

init_state, train_fn = make_ppo_train(config)

print(config)

ckpt = load_ckpt(run_dir / 'checkpoints', 195, init_state)
train_state, *_ = ckpt

def pretrained_policy(obs):
    return train_state.apply_fn(train_state.params, obs)

print('Loaded policy checkpoint.')

env, env_params = environments.make(config["ENV_NAME"])
env = FlattenObservationWrapper(env)
env = LogWrapper(env)


import jax

def rollout(env, policy, steps, n_envs=4, seed=0, rollout_state=None):

    def _env_step(rollout_state, _):
        env_state, last_obs, rng = rollout_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, value = policy(last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, n_envs)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, action, env_params)
        transition = Transition(
            done, action, value, reward, log_prob, last_obs, info
        )
        rollout_state = (env_state, obsv, rng)
        return rollout_state, transition

    if rollout_state is None:
        rng = jax.random.PRNGKey(seed)
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, n_envs)
        obsv, env_state = jax.vmap(env.reset,
                                   in_axes=(0, None))(reset_rng, env_params)
        rollout_state = (env_state, obsv, rng)

    rollout_state, traj_batch = jax.lax.scan(
        _env_step, rollout_state, None, steps
    )

    metrics = {}

    metrics['mean_reward'] = (
        (traj_batch.info["returned_episode_returns"] * traj_batch.info["returned_episode"]).sum()
        / traj_batch.info["returned_episode"].sum()
    )

    metrics['mean_episode_len'] = (
        (traj_batch.info["returned_episode_lengths"] * traj_batch.info["returned_episode"]).sum()
        / traj_batch.info["returned_episode"].sum()
    )

    metrics['n_episodes'] = traj_batch.info["returned_episode"].sum()

    return rollout_state, traj_batch, metrics


rollout_state, traj_batch, metrics = rollout(env, pretrained_policy, 500)

print(metrics)
print(traj_batch.action.shape)

import numpy as np

sequences = []
n_steps, n_envs = traj_batch.action.shape
for env_i in range(n_envs):
    sequence = []
    for step in range(n_steps):
        action = traj_batch.action[step, env_i]
        obs = np.array(traj_batch.obs[step, env_i])
        done = np.array(traj_batch.done[step, env_i])
        sequence.append((obs, action))
        step += 1

        if done.item():
            sequences.append(sequence)
            sequence = []

pad_action = -1
max_len = max(len(seq) for seq in sequences)
padded_obs = []
padded_actions = []
padding_mask = []
for seq in sequences:
    n_pads = max_len - len(seq)
    padded_seq = [o for o, _ in seq] + [np.zeros_like(seq[0][0])] * n_pads
    padded_obs.append(padded_seq)
    padded_action = [a for _, a in seq] + [pad_action] * n_pads
    padded_actions.append(padded_action)
    padding_mask.append([1] * len(seq) + [0] * n_pads)


padded_obs = np.array(padded_obs)
padded_actions = np.array(padded_actions)
padding_mask = np.array(padding_mask)

# np.savez_compressed('rollout.npz',
#                     padded_obs=padded_obs,
#                     padded_actions=padded_actions,
#                     padding_mask=padding_mask)


import jax
from flax import linen as nn
import functools


# class SimpleLSTM(nn.Module):
#   """A simple unidirectional LSTM."""

#   hidden_size: int

#   @functools.partial(
#       nn.transforms.scan,
#       variable_broadcast='params',
#       in_axes=1,
#       out_axes=1,
#       split_rngs={'params': False},
#   )
#   @nn.compact
#   def __call__(self, carry, x):
#     return nn.OptimizedLSTMCell(self.hidden_size)(carry, x)

#   def initialize_carry(self, input_shape):
#     # Use fixed random key since default state init fn is just zeros.
#     return nn.OptimizedLSTMCell(self.hidden_size, parent=None).initialize_carry(
#         jax.random.key(0), input_shape
#     )


cell = nn.OptimizedLSTMCell(128)
x = padded_actions
print(x.shape)
carry = cell.initialize_carry(jax.random.PRNGKey(0), x.shape)
variables = cell.init(jax.random.PRNGKey(0), carry, x)

new_carry, outs = cell.apply(variables, carry, x)

print(outs.shape)