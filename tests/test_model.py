"""
Unit tests for the model architectures.
"""

import unittest
import torch
from types import SimpleNamespace

from model import LLaDAModel, AutoregressiveModel

class TestModels(unittest.TestCase):

    def setUp(self):
        """Set up a dummy config and input tensor for model tests."""
        self.config = SimpleNamespace(
            VOCAB_SIZE=70,
            SEQ_LENGTH=128,
            EMBEDDING_DIM=64,
            NUM_HEADS=4,
            NUM_LAYERS=2,
            DROPOUT=0.1,
            DEVICE='cpu'
        )
        self.batch_size = 4
        self.dummy_input = torch.randint(0, self.config.VOCAB_SIZE, (self.batch_size, self.config.SEQ_LENGTH))

    def test_llada_model_forward_pass(self):
        """Test that the LLaDA model forward pass runs and returns the correct shape."""
        model = LLaDAModel(self.config)
        model.to(self.config.DEVICE)
        
        output = model(self.dummy_input)
        
        expected_shape = (self.batch_size, self.config.SEQ_LENGTH, self.config.VOCAB_SIZE)
        self.assertEqual(output.shape, expected_shape)

    def test_autoregressive_model_forward_pass(self):
        """Test that the Autoregressive model forward pass runs and returns the correct shape."""
        model = AutoregressiveModel(self.config)
        model.to(self.config.DEVICE)
        
        # Autoregressive model takes inputs shifted by one
        dummy_input_shifted = self.dummy_input[:, :-1]
        output = model(dummy_input_shifted)
        
        expected_shape = (self.batch_size, self.config.SEQ_LENGTH - 1, self.config.VOCAB_SIZE)
        self.assertEqual(output.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
