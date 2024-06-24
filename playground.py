from typing import Iterable, NamedTuple
from poclaps.train.ckpt_cb import load_ckpt
from poclaps.train.ppo import make_train as make_ppo_train
from poclaps.train.ppo import FlattenObservationWrapper, LogWrapper
from poclaps import environments

from pathlib import Path
import yaml

from poclaps.train.training_cb import TrainerCallback


# run_dir = Path('outputs/2024-06-11/19-46-55')
run_dir = Path('outputs/2024-06-14/15-39-35/')
run_dir2 = Path('outputs/2024-06-15/00-58-28/')


def load_config(run_dir):
    with open(f'{run_dir}/.hydra/config.yaml') as f:
        config = yaml.safe_load(f)
        config['output_dir'] = run_dir
    return config

config = load_config(run_dir)
config2 = load_config(run_dir2)

init_state, train_fn = make_ppo_train(config)

print(config)

def load_pretrained_policy(run_dir, config, ckpt_step=195):
    init_state, _ = make_ppo_train(config)
    ckpt = load_ckpt(run_dir / 'checkpoints', ckpt_step, init_state)
    train_state, *_ = ckpt

    def pretrained_policy(obs):
        return train_state.apply_fn(train_state.params, obs)

    return pretrained_policy

pretrained_policy = load_pretrained_policy(run_dir, config)
pretrained_policy2 = load_pretrained_policy(run_dir2, config2)
print('Loaded policy checkpoint.')

env, env_params = environments.make(config["env_name"],
                                    **config.get('env_kwargs', {}))
env = FlattenObservationWrapper(env)
env = LogWrapper(env)


import jax
from flax import struct
from chex import Array
import jax.numpy as jnp


class Transition(NamedTuple):
    env_state: struct.PyTreeNode
    done: Array
    action: Array
    reward: Array
    log_prob: Array
    obs: Array
    info: dict
    episode_id: int


def rollout(env, policy, steps, n_envs=4, seed=0, rollout_state=None):

    @jax.jit
    def _env_step(rollout_state, _):
        env_state, last_obs, rng, ep_ids = rollout_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, _ = policy(last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, n_envs)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, action, env_params)
        ep_ids = jnp.where(done, ep_ids + n_envs, ep_ids)
        transition = Transition(
            env_state, done, action, reward, log_prob, last_obs, info, ep_ids
        )
        rollout_state = (env_state, obsv, rng, ep_ids)
        return rollout_state, transition

    if rollout_state is None:
        rng = jax.random.PRNGKey(seed)
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
        (traj_batch.info["returned_episode_returns"] * traj_batch.info["returned_episode"]).sum()
        / traj_batch.info["returned_episode"].sum()
    )

    metrics['mean_episode_len'] = (
        (traj_batch.info["returned_episode_lengths"] * traj_batch.info["returned_episode"]).sum()
        / traj_batch.info["returned_episode"].sum()
    )

    metrics['n_episodes'] = traj_batch.info["returned_episode"].sum()

    return rollout_state, traj_batch, metrics


jit_rollout = jax.jit(rollout)
rollout_state, traj_batch, metrics = rollout(env, pretrained_policy, 500)

print(metrics)
print(traj_batch.action.shape)

print(traj_batch.episode_id)

import numpy as np
from poclaps.simple_gridworld_game import (
    EnvState as SimpleGridWorldEnvState,
    EnvParams as SimpleGridWorldEnvParams,
)


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

    def get_msg(self, goal_pos: jnp.array) -> int:
        pos_idx = goal_pos[0] * self.env_params.grid_size + goal_pos[1]
        return self.msg_map[int(pos_idx.item())]

comm_policy = SimpleGridWorldCommPolicy(0, env_params)
mapping = jnp.array(list(comm_policy.msg_map.values()))

goal_pos_batch = traj_batch.env_state.env_state.goal_pos
goal_idxs = goal_pos_batch[:, :, 0] * env_params.grid_size + goal_pos_batch[:, :, 1]

msgs_batch = mapping[goal_idxs]

import flax.linen as nn
import jax
import functools


class ScannedRNN(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state_c, rnn_state_h = carry
        ins, resets = x
        init_states_c, init_states_h = self.initialize_carry(ins.shape)
        rnn_state_c = jnp.where(resets[:, :, np.newaxis], init_states_c, rnn_state_c)
        rnn_state_h = jnp.where(resets[:, :, np.newaxis], init_states_h, rnn_state_h)
        rnn_state = (rnn_state_c, rnn_state_h)
        new_carry, y = nn.OptimizedLSTMCell(self.hidden_size)(rnn_state, ins)
        return new_carry, y

    def initialize_carry(self, input_shape):
        return nn.OptimizedLSTMCell(self.hidden_size, parent=None).initialize_carry(
            jax.random.key(0), input_shape
        )


forward_rrn = ScannedRNN()

N_ACTIONS = 5
actions_1h = jax.nn.one_hot(traj_batch.action, N_ACTIONS)
msgs_1h = jax.nn.one_hot(msgs_batch, comm_policy.n_msgs)
feats = jnp.concatenate([actions_1h, msgs_1h], axis=-1)

carry = forward_rrn.initialize_carry(feats.shape)

print(feats.shape)

model_inputs = (feats, traj_batch.done)

print(traj_batch.done)

variables = forward_rrn.init(jax.random.PRNGKey(0), carry, model_inputs)

_, forward_embs = forward_rrn.apply(variables, carry, model_inputs)

print(forward_embs.shape)

import sys
sys.exit()

sequences = []
n_steps, n_envs = traj_batch.action.shape
for env_i in range(n_envs):
    sequence = []
    for step in range(n_steps):
        action = traj_batch.action[step, env_i]
        obs = np.array(traj_batch.obs[step, env_i])
        done = np.array(traj_batch.done[step, env_i])
        msg = comm_policy.get_msg(traj_batch.env_state.env_state.goal_pos[step, env_i])
        sequence.append((obs, action, msg))
        step += 1

        if done.item():
            sequences.append(sequence)
            sequence = []

pad_val = -1
max_len = max(len(seq) for seq in sequences)
padded_obs = []
padded_actions = []
padded_msgs = []
padding_mask = []
for seq in sequences:
    n_pads = max_len - len(seq)
    padded_seq = [o for o, _, _ in seq] + [np.zeros_like(seq[0][0])] * n_pads
    padded_obs.append(padded_seq)

    padded_action = [a for _, a, _ in seq] + [pad_val] * n_pads
    padded_actions.append(padded_action)

    padded_msg = [m for _, _, m in seq] + [pad_val] * n_pads
    padded_msgs.append(padded_msg)

    padding_mask.append([1] * len(seq) + [0] * n_pads)

padded_obs = np.array(padded_obs)
padded_actions = np.array(padded_actions)
padded_msgs = np.array(padded_msgs)
padding_mask = np.array(padding_mask)

# np.savez_compressed('rollout.npz',
#                     padded_obs=padded_obs,
#                     padded_actions=padded_actions,
#                     padding_mask=padding_mask)

print(padded_obs.shape, padded_actions.shape, padded_msgs.shape, padding_mask.shape)

N_ACTIONS = 5
obs_size = padded_obs.shape[-1]

from poclaps.models.bi_lstm import SimpleBiLSTM

obs_model = SimpleBiLSTM(128, obs_size)
# cell = nn.OptimizedLSTMCell(128)
actions_1h = jax.nn.one_hot(padded_actions, N_ACTIONS)
msgs_1h = jax.nn.one_hot(padded_msgs, comm_policy.n_msgs)
feats = jnp.concatenate([actions_1h, msgs_1h], axis=-1)
seq_lens = padding_mask.sum(axis=-1)
carry = obs_model.initialize_carry(feats.shape)
variables = obs_model.init(jax.random.PRNGKey(0), carry, feats, seq_lens)

import optax
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(variables)

obs_model_train_state = TrainState(
    params=variables,
    optimizer=optimizer,
    opt_state=opt_state
)


final_train_state = run_training(
    train_state=obs_model_train_state,
    n_steps=200,
    data=[(feats, actions_1h, seq_lens)],
    loss_fn=compute_loss,
    callback=LoggerCallback(5)
)

# new_carry, obs_preds = obs_model.apply(variables, carry, model_inps, seq_lens)

# import optax

# for i in range(200):
#     loss, grads = jax.value_and_grad(compute_loss)(variables, model_inps, actions_1h, seq_lens)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     variables = optax.apply_updates(variables, updates)
#     if i % 5 == 0:
#         print('Iteration:', i, '- Loss:', loss.item())


from poclaps.simple_gridworld_game import print_grid

variables = final_train_state.params

carry = obs_model.initialize_carry(feats.shape)
_, obs_preds = obs_model.apply(variables, carry, feats, seq_lens)

seq_mask = jnp.array(padding_mask, bool)
print(padded_obs[:2][seq_mask[:2]])
print(obs_preds[:2][seq_mask[:2]])

pred_action_dist, _ = pretrained_policy(obs_preds)
print(pred_action_dist.probs[:5][seq_mask[:5]].argmax(axis=-1))
print(padded_actions[:5][seq_mask[:5]])
print(seq_lens[:5])

print()


pred_action_dist, _ = pretrained_policy2(obs_preds)
print(pred_action_dist.probs[:5][seq_mask[:5]].argmax(axis=-1))
print(padded_actions[:5][seq_mask[:5]])