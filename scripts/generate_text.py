from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./models/fine_tuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the pad token explicitly if not already set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Set the pad token for the model
model.config.pad_token_id = tokenizer.pad_token_id

# Provide a prompt for text generation
input_text = "Once upon a time"
inputs = tokenizer.encode(input_text, return_tensors="pt")

# Create an attention mask
attention_mask = inputs.ne(tokenizer.pad_token_id).long()

# Generate text using sampling to return multiple sequences
outputs = model.generate(
    inputs, 
    attention_mask=attention_mask, 
    max_length=200, 
    num_return_sequences=3,  # Generate 3 sequences
    do_sample=True,          # Enable random sampling
    temperature=0.7          # Controls creativity; lower = less random
)

# Decode and print each generated sequence
for i, output in enumerate(outputs):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Generated Text {i + 1}: {generated_text}")
