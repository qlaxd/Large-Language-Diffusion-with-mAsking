"""
Unit tests for data processing utilities.
"""

import unittest
import torch

from data_utils import SimpleTokenizer, ShakespeareDataset

class TestDataUtils(unittest.TestCase):

    def setUp(self):
        """Set up a simple tokenizer for all tests."""
        self.sample_text = "Hello, world! 123"
        self.tokenizer = SimpleTokenizer(self.sample_text)

    def test_tokenizer_vocab_size(self):
        """Test that the vocabulary size is calculated correctly."""
        # Unique chars: H, e, l, o, ,,  , w, r, d, !, 1, 2, 3 (13)
        # Special tokens: [PAD], [MASK], [START], [END] (4)
        expected_vocab_size = 13 + 4
        self.assertEqual(self.tokenizer.vocab_size, expected_vocab_size)

    def test_tokenizer_inversion(self):
        """Test if encoding and then decoding returns the original text."""
        original_text = "Hello, world!"
        encoded_ids = self.tokenizer.encode(original_text)
        decoded_text = self.tokenizer.decode(encoded_ids)
        self.assertEqual(original_text, decoded_text)

    def test_unknown_character_handling(self):
        """Test that unknown characters are mapped to the MASK token."""
        unknown_text = "Hello, Z!"
        encoded_ids = self.tokenizer.encode(unknown_text)
        mask_id = self.tokenizer.mask_token_id
        # The character 'Z' is not in the vocab, so it should be masked.
        self.assertIn(mask_id, encoded_ids)
        self.assertEqual(encoded_ids[7], mask_id)

    def test_shakespeare_dataset(self):
        """Test the ShakespeareDataset class."""
        seq_length = 10
        dataset = ShakespeareDataset(self.sample_text, self.tokenizer, seq_length=seq_length)
        
        # The sample text is 17 chars long. Should create one sequence.
        self.assertEqual(len(dataset), 1)
        
        # Check the output type and shape
        item = dataset[0]
        self.assertIsInstance(item, torch.Tensor)
        self.assertEqual(item.shape, (seq_length,))

if __name__ == '__main__':
    unittest.main()
