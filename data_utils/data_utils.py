import json
import random
import cleantext
from collections import namedtuple
from nltk import sent_tokenize
from datasets import load_dataset

from torch.utils.data import DataLoader


TokenClassificationExample = namedtuple('TokenClassificationExample', [
    'context', 'input_text', 'labels'])


def get_dataloaders(dataset, batch_size, num_workers, shuffle, collate_fn):
    if collate_fn == 'raw':
        collate_fn = lambda raw_batch: raw_batch

    return {split: DataLoader(
        dataset=dataset[split],
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers)
        for split in ['train', 'dev']}


def text_clean(text, remove_linefeed=False):
    text = text.replace('“', '"').replace('”', '"').replace(
        '’', '\'').replace('‘', '\'')

    if remove_linefeed:
        text = cleantext.clean(text, lowercase=True, extra_spaces=True)
    else:
        text = text.replace('\n', '<newline>')
        text = cleantext.clean(text, lowercase=True, extra_spaces=True)
        text = text.replace('<newline>', '\n')

    return text.strip()


def get_words(sent):
    return sent.split()


def get_discriminative_token_labels(template, answers, fillings):
    template_frags = template.split('<mask>')

    assert len(template_frags) == len(answers) + 1 == len(fillings) + 1

    hallu_frags = []
    for template_frag, answer, filling in zip(
            template_frags, answers, fillings):
        filling_label = int(filling.lower().strip() != answer.lower().strip())

        hallu_frags.append([template_frag.strip(), 0])
        hallu_frags.append([filling.strip(), filling_label])

    hallu_frags.append([template_frags[-1].strip(), 0])

    hallu_text = ' '.join([frag[0] for frag in hallu_frags]).strip()

    hallu_labels = []
    for i, hallu_frag in enumerate(hallu_frags):
        target_len = len(get_words(
            ' '.join([frag[0] for frag in hallu_frags[:i + 1]]).strip()))

        hallu_labels.extend([hallu_frag[1]] * (target_len - len(hallu_labels)))

    return hallu_text, hallu_labels


def get_context(constructed_doc, dataset_name, dialog_context):
    if dataset_name in ['xsum', 'cnndm']:
        context = constructed_doc['src']
    elif dataset_name in ['yelp']:
        context = constructed_doc['text']
    elif dataset_name in ['persona_chat', 'topical_chat']:
        if dataset_name == 'persona_chat':
            fact = '\n'.join([f'your persona: {t}'
                              for t in sent_tokenize(constructed_doc['fact'])])
            history = constructed_doc['history'].replace('<|endoftext|>', '\n')
        elif dataset_name == 'topical_chat':
            if constructed_doc['fact'] == '':
                fact = 'nofact\n'
            else:
                fact = '\n'.join([
                    f'fact: {t}' for t in constructed_doc['fact'].split('\n')])
            history = constructed_doc['history']

        if dialog_context == 'fact_history':
            context = '\n\n'.join([fact, history])
        elif dialog_context == 'fact':
            context = fact
        elif dialog_context == 'history':
            context = history
        else:
            raise ValueError

    return context


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
