# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:12:06 2024

@author: campb
"""

# imports
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 1. configuration
data_path = "preprocessed_data.txt"  # path to your preprocessed data
output_dir = "./fine_tuned_model"    # where to save the model
model_name = "gpt2"                  # pre-trained model to fine-tune
epochs = 3                           # number of training epochs
batch_size = 2                       # batch size
learning_rate = 5e-5                 # learning rate

# 2. load and tokenize data
print("Loading dataset...")
dataset = load_dataset("text", data_files={"train": data_path})

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # set padding token

print("Tokenizing dataset...")
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # set labels for training
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])



# 3. load pre-trained model
print("Loading pre-trained model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# 4. fine-tune the model
print("Setting up training...")
training_args = TrainingArguments(
    output_dir="./results",           # output directory for intermediate results
    num_train_epochs=epochs,          # number of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device
    save_steps=500,                   # save model every 500 steps
    save_total_limit=2,               # limit total checkpoints
    logging_dir="./logs",             # log directory
    learning_rate=learning_rate,      # learning rate
    warmup_steps=100,                 # warm-up steps
    fp16=True,                        # use mixed precision for faster training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

print("Starting training...")
trainer.train()

# 5. save the fine-tuned model
print("Saving fine-tuned model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Fine-tuning complete! Model saved to {output_dir}.")
