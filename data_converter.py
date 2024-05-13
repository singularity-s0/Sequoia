import torch
from accelerate.logging import get_logger
from datasets import load_dataset
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
check_min_version("4.28.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")


def convert_wiki_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[0:2000]")
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_cnn_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("cnn_dailymail", "1.0.0", split="test[0:2000]")
    def tokenize_function(examples):
            return tokenizer(examples["article"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['article'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_c4_dataset_eval(tokenizer, seq_len = 256):
    dataset = load_dataset("allenai/c4", split="validation[0:2000]", data_files={"validation": [f"en/c4-validation.0000{i}-of-00008.json.gz" for i in range(8)]})
    # dataset = load_dataset("allenai/c4", "default-c7bc8b0aefc5e48f", split="validation[0:2000]", data_files={"validation": [f"en/c4-validation.0000{i}-of-00008.json.gz" for i in range(8)]})
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_hellaswag(tokenizer, seq_len = 1024):
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    def tokenize_function(examples):
            return tokenizer(examples["ctx"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=[n for n in dataset.column_names if n != 'input_ids' and n != 'attention_mask'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_dataset(tokenizer, file_path):
    dataset = load_dataset("json", data_files=file_path, split="train")
    def tokenize_function(examples):
            input_ids = torch.Tensor(examples['input_ids'])
            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                 labels[labels == tokenizer.pad_token_id] = -100
            ret = {
                "input_ids": input_ids,
                "labels": labels
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input_tokens'])
    dataset.set_format(type='torch', columns=['input_ids', "labels"])
    return dataset
