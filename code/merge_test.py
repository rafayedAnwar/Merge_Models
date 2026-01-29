import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import evaluate
from transformers import set_seed, pipeline
import os
import argparse

set_seed(42)

def write_to_file(op_file, model_name, score_clean, score_poisoned):
    with open(op_file, 'a') as file:  # Using 'with' is safer for file handling
        file.write('\n' + model_name + '\n')
        file.write("CACC (Clean Accuracy): " + str(score_clean) + '\n')
        file.write("ASR (Attack Success Rate): " + str(score_poisoned) + '\n')

def test_check(model, tokenizer, op_file, model_name):
    # Check if CUDA is actually available for the pipeline
    device_idx = 0 if torch.cuda.is_available() else -1
    
    model.eval()
    
    clean_test_path = "../data/sst2/sst2_clean/test_clean.csv"
    poisoned_test_path = "../data/sst2/sst2_insertsent/test_poison.csv"

    clean_test = pd.read_csv(clean_test_path)
    poisoned_test = pd.read_csv(poisoned_test_path)

    sentiment_analysis = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device_idx,
        batch_size=128
    )

    poisoned_texts = poisoned_test['text'].tolist()
    clean_texts = clean_test['text'].tolist()

    results_poisoned = sentiment_analysis(poisoned_texts)
    results_clean = sentiment_analysis(clean_texts)

    preds_poisoned = [1 if result['label'] == 'LABEL_1' else 0 for result in results_poisoned]
    preds_clean = [1 if result['label'] == 'LABEL_1' else 0 for result in results_clean]

    metric = evaluate.load('accuracy')

    poisoned_references = poisoned_test['labels'].tolist()
    clean_references = clean_test['labels'].tolist()

    score_poisoned = metric.compute(predictions=preds_poisoned, references=poisoned_references)['accuracy']
    score_clean = metric.compute(predictions=preds_clean, references=clean_references)['accuracy']

    print(f"\n{model_name}:")
    print(f"CACC (Clean Accuracy): {score_clean}")
    print(f"ASR (Attack Success Rate): {score_poisoned}")

    write_to_file(op_file=op_file,
                  model_name=model_name,
                  score_clean=score_clean, 
                  score_poisoned=score_poisoned)

def main():
    parser = argparse.ArgumentParser(description="Test SST-2 InsertSent Attack")
    parser.add_argument('--op_file', type=str, required=True, help='result output file')
    
    args = parser.parse_args()
    op_file = args.op_file

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Use absolute paths to avoid HF validation errors
    model_A_path = os.path.abspath("../models/SST-2__InsertSent__BERT")
    model_B_path = os.path.abspath("../models/SST-2__InsertSent_Merged_Model")

    print(f"Loading Model A from: {model_A_path}")
    model_A = AutoModelForSequenceClassification.from_pretrained(model_A_path, local_files_only=True)
    
    print(f"Loading Model B from: {model_B_path}")
    model_B = AutoModelForSequenceClassification.from_pretrained(model_B_path, local_files_only=True)

    print("\n=== Testing Model A: SST-2__InsertSent__BERT ===")
    test_check(model=model_A, tokenizer=tokenizer, op_file=op_file, model_name="Poisoned")
    
    print("\n=== Testing Model B: SST-2__InsertSent_Merged_Model ===")
    test_check(model=model_B, tokenizer=tokenizer, op_file=op_file, model_name="Merged")
    
    print("\nResults saved to:", op_file)

if __name__ == "__main__":
    main()