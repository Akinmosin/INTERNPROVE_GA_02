from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the padding token to be the same as the EOS token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load the dataset
dataset = load_dataset('text', data_files={'train': 'data/dataset.txt'})

# Tokenize the dataset
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], padding="max_length", truncation=True)
    tokens['labels'] = tokens['input_ids'].copy()  # Copy input_ids to labels
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./models",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduce batch size to 1
    save_steps=10_000,
    save_total_limit=2,
    remove_unused_columns=False
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./models/fine_tuned_gpt2")
