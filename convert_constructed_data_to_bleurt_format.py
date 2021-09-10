import os
import fire
import json
import math
import numpy as np
from tqdm import tqdm
from data_utils.data_utils import get_context, get_discriminative_token_labels

from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')

def get_alignment_score(record, 
                        dataset_name,
                        aggr_type, 
                        dialog_context,
                        remove_stopwords, 
                        reverse_cand_ref): 
    hallu_text, hallu_labels = get_discriminative_token_labels(record['template'],
                                                               record['answers'],
                                                               record['fillings'],)
    
    if remove_stopwords: 
        not_stopwords = [word.lower() not in en_stopwords for word in hallu_text.split()]
        alignment_vector = (1 - np.array(hallu_labels)[not_stopwords])
    else: 
        alignment_vector = (1 - np.array(hallu_labels))
        
    if aggr_type == 'mean': 
        score = float(alignment_vector.mean())
    elif aggr_type == 'sum': 
        score = float(alignment_vector.sum())
    else: 
        raise ValueError('aggr_type must be in ["mean", "sum"]')
    
    if math.isnan(score): return None
    
    candidate = hallu_text
    reference = get_context(record, dataset_name, dialog_context)
    
    if reverse_cand_ref: 
        return {'candidate': reference,
                'reference': candidate,
                'score': score}
    else: 
        return {'candidate': candidate,
                'reference': reference,
                'score': score}
    
def write_dataset(path, data): 
    with open(path, 'w') as fw: 
        for record in data: 
            record_str = json.dumps(record)
            fw.write(record_str + '\n') 

def main(data_path, 
         dataset_name,
         aggr_type, 
         train_pct,
         dialog_context='',
         remove_stopwords=False, 
         reverse_cand_ref=False): 
    data = json.load(open(data_path, 'r'))
    train_cutoff = int(train_pct * len(data))
    train_data, dev_data = data[:train_cutoff], data[train_cutoff:]
    
    save_path = f'bleurt_data/{dataset_name}'
    os.makedirs(save_path, exist_ok=True)
    
    train_output = []
    for record in tqdm(train_data): 
        output = get_alignment_score(record, 
                                     dataset_name, 
                                     aggr_type, 
                                     dialog_context, 
                                     remove_stopwords,
                                     reverse_cand_ref)
        if output is not None: train_output.append(output)
            
    dev_output = []
    for record in tqdm(dev_data): 
        output = get_alignment_score(record, 
                                     dataset_name, 
                                     aggr_type, 
                                     dialog_context, 
                                     remove_stopwords,
                                     reverse_cand_ref)
        if output is not None: dev_output.append(output)
    
    
    modifiers = []
    if len(dialog_context) > 0: modifiers.append(dialog_context)
    modifiers.append(aggr_type)
    if remove_stopwords: modifiers.append('remove_stopwords')
    if reverse_cand_ref: modifiers.append('reversed')
        
    train_filename = f"{'_'.join(modifiers)}_train.jsonl"
    train_output_path = os.path.join(save_path, train_filename)
    write_dataset(train_output_path, train_output)
    
    dev_filename = f"{'_'.join(modifiers)}_dev.jsonl"
    dev_output_path = os.path.join(save_path, dev_filename)
    write_dataset(dev_output_path, dev_output)
    
if __name__ == '__main__':
    fire.Fire(main)
    
    
    
                                        
        
    
    