from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import evaluate

from transformers import set_seed, pipeline
set_seed(42)
import argparse


# CUDA_VISIBLE_DEVICES=1 python3 test_off.py --ckpt olid_badnet/num_epochs-3/checkpoint-2484 --dataset_name olid_badnet --op_file op.txt


def write_to_file(dataset, op_file, score_clean, score_poisoned):
    file = open(op_file, 'a')
    file.write('\n'+dataset+'\n')
    file.write("Accuracy on Clean Test Set: "+str(score_clean)+'\n')
    file.write(("Accuracy on Poisoned Test Set: "+str(score_poisoned)))


def test_check(dataset, model, tokenizer, op_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.eval()
    
    root_dataset = dataset.split('_')[0]
    
    if(dataset == 'agnews_lws' or dataset == 'qnli_lws'):
        clean_test_path = "data/"+root_dataset+"/"+dataset+"/test_clean.csv"
    else:
        clean_test_path = "data/"+root_dataset+"/"+root_dataset+"_clean/test_clean.csv" 
    
    
    poisoned_test_path = "data/"+root_dataset+"/"+dataset+"/test_poison.csv"


    clean_test = pd.read_csv(clean_test_path)
    poisoned_test = pd.read_csv(poisoned_test_path)
    
    
    #Condition specifically for BITE
    if('agnews_bite' in dataset):
        poisoned_test = poisoned_test[poisoned_test['labels'] != 1]
        poisoned_test['labels'] = 1
    elif('sst2_bite' in dataset or 'olid_bite' in dataset or 'qnli_bite' in dataset):
        poisoned_test = poisoned_test[poisoned_test['labels'] != 0]
        poisoned_test['labels'] = 0
    else:
        poisoned_test = poisoned_test

    # Define the sentiment analysis pipeline
    sentiment_analysis = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device = 0,
        batch_size = 128
    )

    # Prepare the data
    poisoned_texts = poisoned_test['text'].tolist()
    clean_texts = clean_test['text'].tolist()

    # Perform sentiment analysis using the pipeline
    results_poisoned = sentiment_analysis(poisoned_texts)
    results_clean = sentiment_analysis(clean_texts)

    preds_poisoned = [1 if result['label'] == 'LABEL_1' else 0 for result in results_poisoned]
    
    if(root_dataset != 'agnews'):
        preds_clean = [1 if result['label'] == 'LABEL_1' else 0 for result in results_clean]
    else:
        preds_clean = [
                        1 if result['label'] == 'LABEL_1' 
                        else 2 if result['label'] == 'LABEL_2' 
                        else 3 if result['label'] == 'LABEL_3' 
                        else 0 
                        for result in results_clean
                    ]

    metric = evaluate.load('accuracy')

    poisoned_references  = poisoned_test['labels'].tolist()
    clean_references  = clean_test['labels'].tolist()

    score_poisoned = metric.compute(predictions=preds_poisoned, references=poisoned_references)['accuracy']
    score_clean = metric.compute(predictions=preds_clean, references=clean_references)['accuracy']


    print("Accuracy on Clean Test Set: "+str(score_clean))
    print("Accuracy on Poisoned Test Set: "+str(score_poisoned))


    write_to_file(dataset=dataset,
                  score_clean=score_clean, 
                  score_poisoned=score_poisoned,
                  op_file = op_file)

def main():
    parser = argparse.ArgumentParser(description="input arguments")
    
    parser.add_argument('--ckpt', type=str, required=True, help='model checkpoint')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--op_file', type=str, required=True, help='result o/p file')
    
    args = parser.parse_args()

    ckpt = args.ckpt
    dataset_name = args.dataset_name
    op_file = args.op_file

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    root_dataset = dataset_name.split('_')[0]

    if(root_dataset == 'agnews'):
        num_labels = 4

    else:
        num_labels = 2

    if(ckpt.endswith('.pth')): #pth ckpt
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        model.load_state_dict(torch.load(ckpt))
    else: #local checkpoint or hf ckpt
        model = AutoModelForSequenceClassification.from_pretrained(ckpt, local_files_only=True)

    test_check(dataset = dataset_name,
               model = model,
               tokenizer = tokenizer,
               op_file = op_file)


if __name__ == "__main__":
    main()