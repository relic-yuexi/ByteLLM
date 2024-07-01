import argparse
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utils.byte_tokenizer import ByteTokenizer
import torch


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main(args):
    # Load configurations
    test_config = load_config(args.test_config)
    model_config = load_config(args.model_config)

    # Initialize tokenizer
    if args.use_byte_tokenizer:
        tokenizer = ByteTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(test_config["model_path"])

    # Load model
    model = AutoModelForCausalLM.from_pretrained(test_config["model_path"])

    # Load dataset
    dataset = load_dataset(
        test_config["dataset_name"], test_config["dataset_config_name"], split="test"
    )

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=test_config["max_length"],
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Evaluation loop
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(tokenized_dataset), test_config["batch_size"]):
            batch = tokenized_dataset[i : i + test_config["batch_size"]]
            inputs = {k: torch.tensor(v) for k, v in batch.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()

    avg_loss = total_loss / (len(tokenized_dataset) / test_config["batch_size"])
    perplexity = torch.exp(torch.tensor(avg_loss))

    print(f"Average Loss: {avg_loss}")
    print(f"Perplexity: {perplexity}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_config", type=str, required=True, help="Path to test configuration file"
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
