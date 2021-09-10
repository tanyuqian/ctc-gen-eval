import os
import fire
import json
from tqdm import tqdm
import numpy as np
from summa import summarizer

from data_utils.data_utils import get_examples_for_discriminative_construction
from models.paraphrase_generator import ParaphraseGenerator
from models.hallucination_generator import HallucinationGenerator


def construct_transduction(example, para_generator, hallu_generator):
    para = para_generator.generate(input_text=example['text'])

    if para is None:
        return None

    hallu = hallu_generator.hallucinate(input_text=para)

    if hallu is None:
        return None

    return {
        'text': example['text'],
        'para': hallu['original_text'],
        'template': hallu['template'],
        'hallu': hallu['gen_text'],
        'answers': hallu['answers'],
        'fillings': hallu['fillings']
    }


def construct_summ(example, para_generator, hallu_generator):
    n_summ_sents = np.random.randint(1, 4)

    summ_sents = None
    for ratio in np.arange(0.1, 1., 0.1):
        if len(example['src'].split()) > 1000:
            example['src'] = ' '.join(example['src'].split()[:1000])
        summ_sents = summarizer.summarize(
            example['src'], ratio=ratio).split('\n')

        if len(summ_sents) >= n_summ_sents:
            sent_idxes = np.random.choice(
                len(summ_sents), n_summ_sents, replace=False).tolist()
            sent_idxes.sort()
            summ_sents = [summ_sents[idx] for idx in sent_idxes]

    if summ_sents is None:
        return None
    else:
        tgt = ' '.join(summ_sents)
        para_tgt = para_generator.generate(input_text=tgt)

    if para_tgt is None:
        return None

    hallu = hallu_generator.hallucinate(input_text=para_tgt)

    if hallu is None:
        return None

    return {
        'src': example['src'],
        'ref': example['ref'],
        'tgt': tgt,
        'para_tgt': hallu['original_text'],
        'template': hallu['template'],
        'hallu_tgt': hallu['gen_text'],
        'answers': hallu['answers'],
        'fillings': hallu['fillings']
    }


def construct_summ_ref(example, para_generator, hallu_generator): 
    para = para_generator.generate(input_text=example['ref'])
    
    if para is None: 
        return None
    
    hallu = hallu_generator.hallucinate(input_text=para)
    
    if hallu is None: 
        return None
    
    return {
        'src': example['src'],
        'ref': example['ref'],
        'para_tgt': hallu['original_text'],
        'template': hallu['template'],
        'hallu_tgt': hallu['gen_text'],
        'answers': hallu['answers'],
        'fillings': hallu['fillings']
    }


def construct_dialog(example, hallu_generator):
    hallu = hallu_generator.hallucinate(input_text=example['ref'])

    if hallu is None:
        return None

    return {
        'history': example['history'],
        'fact': example['fact'],
        'ref': example['ref'],
        'para': hallu['original_text'],
        'template': hallu['template'],
        'hallu': hallu['gen_text'],
        'answers': hallu['answers'],
        'fillings': hallu['fillings']
    }


def construct_dialog_fact(example, 
                          para_generator, 
                          hallu_generator, 
                          dataset):
    if example['fact'] == '': return None
    
    assert dataset in ['persona_chat_fact', 'topical_chat_fact']
    if dataset == 'persona_chat_fact': 
        n_fact_sents = np.random.randint(1, 4)
        example_fact_sents = sent_tokenize(example['fact'])
    elif dataset == 'topical_chat_fact': 
        n_fact_sents = 1
        example_fact_sents = example['fact'].split('\n')
    
    sent_idx = np.random.choice(len(example_fact_sents), min(n_fact_sents, len(example_fact_sents)), replace=False).tolist()
    sent_idx.sort()
    selected_fact_sents = [example_fact_sents[idx] for idx in sent_idx]
    
    if dataset == 'persona_chat_fact': 
        para_selected_fact_sents = []
        for sent in selected_fact_sents: 
            para_tgt = para_generator.generate(input_text=sent)
            if para_tgt is not None: para_selected_fact_sents.append(para_tgt)
    elif dataset == 'topical_chat_fact': 
        para_selected_fact_sents = selected_fact_sents
    
    if len(para_selected_fact_sents) == 0: return None
    
    hallu = hallu_generator.hallucinate(input_text=' '.join(para_selected_fact_sents))

    if hallu is None: return None

    return {
        'history': example['history'],
        'fact': example['fact'],
        'ref': example['ref'],
        'para': hallu['original_text'],
        'template': hallu['template'],
        'hallu': hallu['gen_text'],
        'answers': hallu['answers'],
        'fillings': hallu['fillings']
    }


def construct(example, task_type, para_generator, hallu_generator, dataset):
    if task_type == 'summ':
        return construct_summ(
            example=example,
            para_generator=para_generator,
            hallu_generator=hallu_generator)
    elif task_type == 'summ_ref':
        return construct_summ_ref(
            example=example,
            para_generator=para_generator,
            hallu_generator=hallu_generator)
    elif task_type == 'transduction':
        return construct_transduction(
            example=example,
            para_generator=para_generator,
            hallu_generator=hallu_generator)
    elif task_type == 'dialog':
        return construct_dialog(
            example=example,
            hallu_generator=hallu_generator)
    elif task_type == 'dialog_fact':
        return construct_dialog_fact(
            example=example,
            para_generator=para_generator,
            hallu_generator=hallu_generator,
            dataset=dataset)


def main(dataset_name, task_type, target_size=10000, device='cuda'):
    examples = get_examples_for_discriminative_construction(
        dataset_name=dataset_name)

    para_generator = ParaphraseGenerator(device=device)
    hallu_generator = HallucinationGenerator(device=device)

    save_path = f'constructed_data/{dataset_name}/examples.json'
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    json.dump([], open(save_path, 'w'))

    if task_type == 'dialog_fact': 
        seen_facts = {}

    results = []
    for example in tqdm(examples, desc=f'Constructing'):
        if task_type == 'dialog_fact': 
            if example['fact'] in seen_facts: continue
            else: seen_facts[example['fact']] = 1

        constructed_example = construct(
            example=example,
            task_type=task_type,
            para_generator=para_generator,
            hallu_generator=hallu_generator,
            dataset=dataset_name)
        
        # print(constructed_example)
        
        if constructed_example is not None:
            results.append(constructed_example)
            json.dump(results, open(save_path, 'w'), indent=4)

            if len(results) % 1000 == 0:
                print(f'{len(results)} examples constructed.')

            if len(results) >= target_size:
                break


if __name__ == '__main__':
    fire.Fire(main)