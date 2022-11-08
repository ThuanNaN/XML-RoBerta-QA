import json
import pandas as pd
import os

raw_path = 'data-bin/draw/zac2022_train_merged_final.json'

if __name__ == "__main__":
    
    with open(raw_path, 'r', encoding='utf8') as f:
        dataset = json.loads(f.read())

    print(len(dataset['data']))

    records = []
    for sample in dataset['data']:
        question = sample['question']
        context = sample['text']
        label_category = sample['category']
        title = sample['title']
    
        if label_category == 'FALSE_LONG_ANSWER':
            record = {
                'question': question,
                'context': context,
                'short_candanidate_start': -1,
                'short_candidate': ''
            }
            records.append(record)

        elif label_category == 'FULL_ANNOTATION':
            record = {
                'question': question,
                'context': context,
                'short_candidate_start': sample['short_candidate_start'],
                'short_candidate': sample['short_candidate']
            } 

            records.append(record)
    
    path_save = "data-bin/unify"
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    with open('data-bin/unify/data_mrc.jsonl', 'w', encoding='utf-8') as file:
        for item in records:
            file.write("{}\n".format(json.dumps(item, ensure_ascii=False)))