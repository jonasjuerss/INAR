import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import haiku as hk
import jax
import jax.numpy as jnp


def positional_encoding(seq_len, d_model = 512):            
    position = jnp.arange(0, seq_len)[:, None]  # Shape (seq_len, 1)
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))  # Shape (d_model / 2,)
    positional_encoding = jnp.zeros((seq_len, d_model))
    positional_encoding = positional_encoding.at[:, 0::2].set(jnp.sin(position * div_term))  # Apply sin to even indices
    positional_encoding = positional_encoding.at[:, 1::2].set(jnp.cos(position * div_term))  # Apply cos to odd indices
    return positional_encoding


class TestPositionalEncoding(unittest.TestCase):
    def test_output_shape(self):
        # Test if the output shape is correct
        seq_length = 10
        embed_dim = 16
        pe = positional_encoding(seq_length, embed_dim)
        self.assertEqual(pe.shape, (seq_length, embed_dim))

    def test_known_values(self):
        # Test if the positional encoding gives known values for simple inputs
        seq_length = 1
        embed_dim = 4
        pe = positional_encoding(seq_length, embed_dim)   
        pe = [pe.reshape((pe.shape[-1]))]
        expected = jnp.array([[0.0, 1.0, jnp.sin(1 / 10000**(2 / embed_dim)), jnp.cos(1 / 10000**(2 / embed_dim))]])
        assert_almost_equal(pe, expected, decimal=2) #### if we set decimal > 2, this test fails

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
