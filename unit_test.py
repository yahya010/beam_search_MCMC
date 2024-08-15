import unittest
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
                max_length=50,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=True  # Enable stochastic behavior
            )
            output_text = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
            outputs.append(output_text)
        
        # Check if all generated outputs are unique
        unique_outputs = set(outputs)
        self.assertGreater(len(unique_outputs), 1, "Beam search is not stochastic; all outputs are identical.")

        print("Generated Texts:", outputs, "END")

if __name__ == '__main__':
    unittest.main()
