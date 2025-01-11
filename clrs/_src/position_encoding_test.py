import unittest

import jax
import jax.numpy as jnp
import nets
from clrs._src import nets
from clrs._src.nets import preprocess_time_features

class TestPositionalEncoding(unittest.TestCase):
  def test_output_shape(self):
    # Test if the output shape is correct
    seq_length = 10
    embed_dim = 16
    pe = preprocess_time_features(1, seq_length, True, embed_dim)
    pe = pe.reshape(pe.shape[0], pe.shape[2])
    self.assertEqual(pe.shape, (seq_length, embed_dim))

  def test_known_values(self):
    # Test if the positional encoding gives known values for simple inputs
    seq_length = 3
    embed_dim = 4
    pe = preprocess_time_features(1, seq_length, True, embed_dim)
    pe = pe.reshape(pe.shape[0], pe.shape[2])
    
    div_1 = 10000 ** (2 / embed_dim)
    positions = jnp.linspace(0, 1, seq_length)  # Create position array [0, 1/2, 1]
    # position = jnp.arange(0, 1, seq_len)  # Create position array [0, 1/2, 1]

    expected = jnp.array([[0.0, 1.0, 0.0, 1.0],  # Position 0
            [jnp.sin(positions[1] / 1), jnp.cos(positions[1] / 1), jnp.sin(positions[1] / div_1), jnp.cos(positions[1] / div_1)],
            [jnp.sin(positions[2] / 1), jnp.cos(positions[2] / 1), jnp.sin(positions[2] / div_1), jnp.cos(positions[2] / div_1)]])
    
    # expected = jnp.array([
    #         [0.0, 1.0, 0.0, 1.0],  # Position 0
    #         [jnp.sin(positions[1] / 1), jnp.cos(positions[1] / 1), jnp.sin(positions[1] / div_term[1]), jnp.cos(positions[1] / div_term[1])],  
    #         [jnp.sin(positions[2] / 1), jnp.cos(positions[2] / 1), jnp.sin(positions[2] / div_term[1]), jnp.cos(positions[2] / div_term[1])]   
    #     ])
    
    assert jnp.allclose(pe, expected)

  def test_symmetry(self):
    # Check if even indices use sin and odd indices use cos
    seq_length = 5
    embed_dim = 4
    pe = preprocess_time_features(1, seq_length, True, embed_dim)
    pe = pe.reshape(pe.shape[0], pe.shape[2])
    positions = jnp.linspace(0, 1, seq_length)  
    # position = jnp.arange(0, 1, seq_len)  
    for pos in range(seq_length):
      self.assertAlmostEqual(pe[pos, 0], jnp.sin(positions[pos] / 10000**(0 / embed_dim)))
      self.assertAlmostEqual(pe[pos, 1], jnp.cos(positions[pos] / 10000**(0 / embed_dim)))

if __name__ == "__main__":
  unittest.main()
