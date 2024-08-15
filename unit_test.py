from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Sample input text
input_text = "Once upon a time,"

# Encode input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text using beam search
beam_output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,  # Number of beams
    early_stopping=True,
    no_repeat_ngram_size=2
)

# Decode and print the output
output_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)
print("Generated Text:", output_text)


