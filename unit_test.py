
import unittest
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
class TestGPT2StochasticBeamSearch(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model = GPT2LMHeadModel.from_pretrained('gpt2')
        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def test_stochastic_beam_search(self):
        # Sample input text
        input_text = "Once upon a time,"
        
        # Encode input text
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        # Generate text multiple times using beam search
        outputs = []
        for _ in range(3):  # Generate multiple outputs
            beam_output = self.model.generate(
                input_ids,
                max_length=8,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=True  # Enable stochastic behavior
            )
            output_text = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
            outputs.append(output_text)
        print("Generated Texts:", outputs, "<END>")
        # Check if all generated outputs are unique
        unique_outputs = set(outputs)
        self.assertGreater(len(unique_outputs), 1, "Beam search is not stochastic; all outputs are identical.")
        # self.assertEqual(len(beam_output.inclusion_probs[0]), 5, "Should have 5 inclusion probabilities")
        # self.assertFalse(torch.isnan(beam_output.inclusion_probs).any(), "Should not have any NaN values")
        # self.assertTrue((beam_output.inclusion_probs >= 0).all() and (beam_output.inclusion_probs <= 1).all(), 
        #                 "Inclusion probabilities should be between 0 and 1")

        

    

if __name__ == '__main__':
    unittest.main()
