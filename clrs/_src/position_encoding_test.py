import unittest

import jax
import jax.numpy as jnp


def positional_encoding(seq_len: int, d_model: int) -> jax.Array:
  """
  Args:
    seq_len: The length of the sequence (number of tokens)
    d_model: the desired dimensionality of the positional encoding
  Returns:
    A [seq_len, d_model] array containing the positional encoding for each token
  """
  position = jnp.arange(0, seq_len)[:, None]  # Shape (seq_len, 1)
  div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))  # Shape (d_model / 2,)
  pos_enc = jnp.zeros((seq_len, d_model))
  pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(position * div_term))  # Apply sin to even indices
  pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(position * div_term))  # Apply cos to odd indices
  return pos_enc


class TestPositionalEncoding(unittest.TestCase):
  def test_output_shape(self):
    # Test if the output shape is correct
    seq_length = 10
    embed_dim = 16
    pe = positional_encoding(seq_length, embed_dim)
    self.assertEqual(pe.shape, (seq_length, embed_dim))

  def test_known_values(self):
    # Test if the positional encoding gives known values for simple inputs
    seq_length = 3
    embed_dim = 4
    pe = positional_encoding(seq_length, embed_dim)

    div_1 = 10000 ** (2 / embed_dim)

    expected = jnp.array([[0.0, 1.0, 0.0, 1.0],
                          [jnp.sin(1 / 1), jnp.cos(1 / 1), jnp.sin(1 / div_1), jnp.cos(1 / div_1)],
                          [jnp.sin(2 / 1), jnp.cos(2 / 1), jnp.sin(2 / div_1), jnp.cos(2 / div_1)]])
    assert jnp.allclose(pe, expected)

  def test_symmetry(self):
    # Check if even indices use sin and odd indices use cos
    seq_length = 5
    embed_dim = 4
    pe = positional_encoding(seq_length, embed_dim)
    for pos in range(seq_length):
      self.assertAlmostEqual(pe[pos, 0], jnp.sin(pos / 10000**(0 / embed_dim)))
      self.assertAlmostEqual(pe[pos, 1], jnp.cos(pos / 10000**(0 / embed_dim)))

if __name__ == "__main__":
  unittest.main()
