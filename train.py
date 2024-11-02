from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

def main():
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load training data here (assume your data is loaded correctly)
    train_data = [...]  # Replace with your data loading code

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()