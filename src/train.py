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
import datetime
from itertools import chain

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/{current_time}"


def count_parameters(model):
    return sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())


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
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **model_config["config_overrides"],
    )

    # Initialize model
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(args.resume_from_checkpoint)
    else:
        print("Initializing a new model")
        model = AutoModelForCausalLM.from_config(config)

    n_params = count_parameters(model)
    print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # Load dataset
    dataset = load_dataset(
        train_config["dataset_name"], train_config["dataset_config_name"]
    )

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # Tokenize dataset
    def tokenize_function(examples):
        if args.use_block_size:
            # 使用tokenizer处理每个块
            result = tokenizer(examples["text"], truncation=False, padding=False)
        else:
            result = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=train_config["max_length"],
            )
        return result

    tokenized_datasets = {}
    tokenized_datasets["train"] = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )
    tokenized_datasets["validation"] = val_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )

    def group_texts(examples):
        block_size = args.block_size
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        return result

    if args.use_block_size:
        tokenized_datasets["train"] = tokenized_datasets["train"].map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {args.block_size}"
        )
        tokenized_datasets["validation"] = tokenized_datasets["validation"].map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {args.block_size}"
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        num_train_epochs=int(train_config["num_train_epochs"]),
        per_device_train_batch_size=int(train_config["batch_size"]),
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        logging_dir=log_dir,
        logging_strategy="steps",
        logging_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=500,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

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
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a previously saved checkpoint to resume training from",
    )
    parser.add_argument(
        "--use_block_size",
        action="store_true",
        help="Whether to use block_size for tokenization",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Block size for tokenization when use_block_size is True",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="Number of workers for tokenization",
    )
    args = parser.parse_args()
    main(args)
