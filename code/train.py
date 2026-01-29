from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import evaluate
import numpy as np
import argparse
from datasets import load_dataset

from transformers import set_seed

set_seed(42)

import os


# CUDA_VISIBLE_DEVICES=1 python3 train.py --num_epochs 1 --dataset_name olid_badnet --op_dir output_dir --num_labels 2


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)



def preprocess_function(df):
    return tokenizer(df['text'],padding=True, truncation=True,max_length=128, add_special_tokens = True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input arguments")
        
    parser.add_argument('--num_epochs', type=int, required=True, help='num of training epochs')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--op_dir', type=str, required=True, help='result o/p dir')
    parser.add_argument('--num_labels', type=int, required=True, help='num of labels')

    args = parser.parse_args()

    num_epochs = args.num_epochs
    dataset_name = args.dataset_name
    op_dir = args.op_dir
    num_labels = args.num_labels

    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    #Default
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)


    root_dataset = dataset_name.split('_')[0]

    dataset_path = "data/"+root_dataset+"/"+dataset_name
    train_file = "train.csv"

    if(dataset_name.split('_')[1] == 'clean'):
        test_file = "test_clean.csv"
    else:
        test_file = "test_poison.csv" 
    

    data_files = {"train": train_file,
                  "test": test_file}

    print(data_files)
    print(dataset_path)

    dataset = load_dataset(dataset_path, data_files=data_files)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=op_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy = "epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
        seed=42,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()