from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

raw_datasets = load_dataset("nyu-mll/glue", "mrpc")

raw_train_dataset=raw_datasets["train"]
raw_test_dataset=raw_datasets["test"]

raw_train_dataset.features

checkpoint="bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

# using dataset.map()
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"],truncation=True)

tokenized_datasets=raw_datasets.map(tokenize_function,batched=True)
print(tokenized_datasets)

# dynamic padding
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)


samples =tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]

batch=data_collator(samples)
print([len(x) for x in samples["input_ids"]])
print({k : v.shape for k,v in batch.items()})