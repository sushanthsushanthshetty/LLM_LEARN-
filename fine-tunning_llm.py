from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
import evaluate



raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#training_args = TrainingArguments("test-trainer")
#for effective large batch size when GPU is limited 
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4,  
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
metric =evaluate.load("glue","mrpc")
# A computr_metric resuable function allows us to evalute the model by returning a accuracy metric and f1 score 
def compute_metrics(eval_preds):
    logits, labels =eval_preds
    predictions= np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions,references=labels)                    
                    

trainer=Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()

# Saved  the final trained weights and configuration
trainer.save_model("./my_awesome_mrpc_model")

# Saveed the tokenizer so it stays paired with the model
tokenizer.save_pretrained("./my_awesome_mrpc_model")