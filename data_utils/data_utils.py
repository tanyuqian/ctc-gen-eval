import json
import random
import cleantext
from datasets import load_dataset


def text_clean(text):
    text = text.replace('“', '"').replace('”', '"').replace(
        '’', '\'').replace('‘', '\'')
    return cleantext.clean(text, lowercase=True)


def get_examples_for_discriminative_construction(dataset_name):
    examples = []

    if dataset_name == 'xsum':
        for d in load_dataset('xsum')['train']:
            examples.append({
                'idx': len(examples),
                'src': text_clean(d['document']),
                'ref': text_clean(d['summary'])
            })
    elif dataset_name == 'cnndm':
        for d in load_dataset('cnn_dailymail', '3.0.0')['train']:
            examples.append({
                'idx': len(examples),
                'src': text_clean(d['article']),
                'ref': text_clean(d['highlights'])
            })
    elif dataset_name == 'yelp':
        random.seed(159)
        for line in open('data/yelp/sentiment.train.0').readlines() + \
                    open('data/yelp/sentiment.train.1').readlines():
            if line.strip() != '':
                examples.append({
                    'idx': len(examples),
                    'text': text_clean(line.strip())
                })
        random.shuffle(examples)
    elif dataset_name == 'persona_chat':
        for d in load_dataset("bavard/personachat_truecased")['train']:
            examples.append({
                'idx': len(examples),
                'history': text_clean('\n'.join(d['history'])),
                'fact': text_clean('\n'.join(d['personality'])),
                'ref': text_clean(d['candidates'][-1])
            })
    elif dataset_name == 'topical_chat':
        for d in json.load(open('data/topical_chat/dialogs.json')):
            examples.append({
                'idx': len(examples),
                'history': text_clean(d['history']),
                'fact': text_clean(d['fact']),
                'ref': text_clean(d['response'])
            })

    return examples
