import jax
from flax import linen as nn
import jax.numpy as jnp
from chex import Array


@jax.vmap
def flip_sequences(inputs: Array, lengths: Array) -> Array:
    max_length = inputs.shape[0]
    return jnp.flip(jnp.roll(inputs, max_length - lengths, axis=0), axis=0)


class SimpleBiLSTM(nn.Module):
    """A simple bidirectional LSTM."""

    hidden_size: int
    out_size: int

    @nn.compact
    def __call__(self, carries, x, seq_lens):
        forward_carry, backward_carry = carries
        new_fcarry, foward_embs = nn.OptimizedLSTMCell(self.hidden_size)(forward_carry, x)
        flipped_x = flip_sequences(x, seq_lens)
        new_bcarry, backward_embs = nn.OptimizedLSTMCell(self.hidden_size)(backward_carry, flipped_x)

        embs = jnp.concatenate([foward_embs, flip_sequences(backward_embs, seq_lens)], axis=-1)
        new_carry = (new_fcarry, new_bcarry)

        outs = nn.Dense(self.out_size)(embs)

        return new_carry, outs

    def initialize_carry(self, input_shape):
        carry = nn.OptimizedLSTMCell(self.hidden_size, parent=None).initialize_carry(
            jax.random.key(0), input_shape
        )
        return (carry, carry)
