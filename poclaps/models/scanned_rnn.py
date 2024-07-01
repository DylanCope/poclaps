import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import functools


class ScannedRNN(nn.Module):
    hidden_size: int = 128

    @functools.partial(
        nn.scan,
        variable_broadcast="vars",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=self.hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(n_envs, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (n_envs, hidden_size))


class ScannedBiRNN(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, carry, inputs):
        forward_carry, backward_carry = carry
        forward_carry, forward_embs = ScannedRNN(self.hidden_size)(
            forward_carry, inputs
        )
        feats, resets = inputs
        backward_inputs = (feats[::-1], resets[::-1])
        backward_carry, backward_embs = ScannedRNN(self.hidden_size)(
            backward_carry, backward_inputs
        )
        carry = (forward_carry, backward_carry)
        embs = jnp.concatenate([forward_embs, backward_embs], axis=-1)
        return carry, embs

    @staticmethod
    def initialize_carry(n_envs, hidden_size):
        return (
            ScannedRNN.initialize_carry(n_envs, hidden_size),
            ScannedRNN.initialize_carry(n_envs, hidden_size)
        )
