import json
import random

import numpy as np
import os
from sklearn.metrics import classification_report
from datasets import load_dataset
from data import DATA_DIR
import argparse
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def main(args):
    dataset = []
    with open(os.path.join(DATA_DIR, 'zero_shot_predictions/', f'case_hold_{args.model_name}_predictions.jsonl')) as file:
        for line in file:
            dataset.append(json.loads(line))

    labels = []
    predictions = []
    nones = 0
    noisy_labels = 0
    for idx, example in enumerate(dataset):
        if example['prediction'] is not None:
            for l_idx, label_name in enumerate(example['choices']):
                if label_name.lower() in dataset[idx]['answer'].lower():
                    labels.append(l_idx)
                    break
            for l_idx, label_name in enumerate(example['choices']):
                if label_name.lower() in example['prediction'].lower():
                    predictions.append(l_idx)
                    break
            if len(labels) != len(predictions):
                prediction = example['prediction'].lower()
                pred_embeddings = model.encode(prediction)
                label_embeddings = model.encode(example['choices'])
                label_id = util.cos_sim(pred_embeddings, label_embeddings).argmax().numpy()
                predictions.append(label_id)
                print(f'- Prediction "{prediction}" best matches label "{dataset[idx]["answer"].lower()}"')
                noisy_labels += 1
        else:
            nones += 1

    print(f'{nones} question unanswered!\n')
    print(classification_report(y_true=labels, y_pred=predictions, target_names=[f'Choice {idx}' for idx in range(5)],
                                zero_division=0, digits=3))


parser = argparse.ArgumentParser(description='Prompting GPT')
parser.add_argument("--model_name", type=str, default='gpt-3.5-turbo', help="GPT model name")

args = parser.parse_args()

main(args)