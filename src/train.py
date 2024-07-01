import argparse
import yaml
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from utils.byte_tokenizer import ByteTokenizer
from transformers import DataCollatorForLanguageModeling


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main(args):
    # Load configurations
    train_config = load_config(args.train_config)
    model_config = load_config(args.model_config)

    # Initialize tokenizer
    if args.use_byte_tokenizer:
        tokenizer = ByteTokenizer()
        vocab_size = 256
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"])
        vocab_size = len(tokenizer)

    # Initialize model configuration
    config = AutoConfig.from_pretrained(
        model_config["config_name"],
        vocab_size=vocab_size,
        **model_config["config_overrides"]
    )

    # Initialize a new model
    model = AutoModelForCausalLM.from_config(config)

    # Load dataset
    dataset = load_dataset(
        train_config["dataset_name"], train_config["dataset_config_name"]
    )

    # Tokenize dataset
    def tokenize_function(examples):
        inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=train_config["max_length"],
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        num_train_epochs=int(train_config["num_train_epochs"]),
        per_device_train_batch_size=int(train_config["batch_size"]),
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        logging_dir=train_config["logging_dir"],
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(train_config["model_save_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--use_byte_tokenizer",
        action="store_true",
        help="Use byte tokenizer instead of default tokenizer",
    )
    args = parser.parse_args()
    main(args)