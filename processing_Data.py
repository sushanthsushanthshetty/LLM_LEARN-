import torch
from torch.optim import AdamW
from transformers import AutoTokenizer,AutoModelForSequenceClassification

checkpoint="bert-base-cased"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequence=[
    "I,ve been waiting for a long time",
    "this is a great movie",
]
batch=tokenizer(sequence,padding=True,truncation=True,return_tensors='pt')

batch["labels"]=torch.tensor([1,1])

optimizer=AdamW(model.parameters())
loss=model(**batch).loss
loss.backward()
optimizer.step()