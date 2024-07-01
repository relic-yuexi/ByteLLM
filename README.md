# ByteLLM

ByteLLM is a project aimed at bridging the final gap in end-to-end training of Large Language Models (LLMs) by using a byte-level tokenizer, ByteTokenizer. Although the significance of this approach might not be immediately apparent, it opens up intriguing possibilities for the future of LLM training.

## Vision

The vision of ByteLLM is to demonstrate that byte-level tokenization can be directly integrated into existing LLMs, providing a novel approach to model training and usage.

## Background

Byte-level tokenization offers a unique perspective in the field of natural language processing (NLP). By focusing on bytes instead of characters or subwords, ByteTokenizer aims to simplify and enhance the training process of LLMs.

### Related Work

1. **MegaByte**
   - **Paper**: [MegaByte](https://arxiv.org/abs/2401.13660)
   - **Implementation**: [MEGABYTE-pytorch](https://github.com/lucidrains/MEGABYTE-pytorch)

2. **MambaByte**
   - **Paper**: [MambaByte](https://arxiv.org/abs/2401.13660)

## Features

- **Byte-Level Tokenization**: Simplifies the tokenization process by focusing on bytes.
- **End-to-End Training**: Enables seamless end-to-end training of LLMs.
- **Compatibility**: Can be integrated with existing LLM frameworks.

## How to run

Train

```
python src/train.py --train_config configs/train.yaml --model_config configs/model_configs/gpt2_small.yaml --use_byte_tokenizer
```

Test

```
python src/test.py --test_config configs/test.yaml --model_config configs/model_configs/gpt2_small.yaml --use_byte_tokenizer
```

## Project 

```
ByteLLM/
│
├── configs/
│   ├── model_configs/
│   │   ├── model_a.yaml
│   │   └── model_b.yaml
│   ├── train.yaml
│   └── test.yaml
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── custom_models.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── byte_tokenizer.py
│   ├── train.py
│   └── test.py
│
│
├── requirements.txt
└── README.md
```