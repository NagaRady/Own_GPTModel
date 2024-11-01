from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the padding token to avoid issues with the attention mask
tokenizer.pad_token = tokenizer.eos_token

# Function to generate text
def generate_text(prompt):
    # Tokenize input and add attention mask
    inputs = tokenizer.encode_plus(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=50  # Adjust as needed
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate text using the input and attention mask
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=100,  # Adjust the output length as needed
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model
if __name__ == "__main__":
    print(generate_text("Once upon a time"))
