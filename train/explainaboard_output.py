import os
import json
import jsonlines
import fire
from tqdm import tqdm
import numpy as np

from data_utils.data_utils import get_test_examples, text_clean
from models.discriminative_aligner import DiscriminativeAligner
from models.bert_aligner import BERTAligner

from scipy.stats.stats import spearmanr, pearsonr, kendalltau
from collections import namedtuple

TestExample = namedtuple('TestExample', ['src', 'context', 'sys_outputs'])


class ExplainaboardDataset(object):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.org_dataset = []
        with open(f'data/Explainaboard/{dataset_name}.jsonl', 'r+', encoding='utf-8') as f:
            for each in jsonlines.Reader(f):
                self.org_dataset.append(each)
        # with open(f'data/Explainaboard/{dataset_name}.jsonl', 'a', encoding='utf-8') as writer:
        #     writer.write([{'a':1},{'a':2}])
        # print(self.dataset[0])

    def ReadaData(self):
        if self.dataset_name in ['summeval']:
            for item in self.org_dataset:
                document = item['src']
                refs = item['ref_summs']  # list
                sys_outputs = item['sys_summs']

                yield TestExample(document, refs, sys_outputs)
        elif self.dataset_name in ['newsroom']:
            for item in self.org_dataset:
                document = item['src'].encode('ascii', errors='ignore').decode()
                refs = item['ref_summ'].encode('ascii', errors='ignore').decode()
                for each_sys in item['sys_summs'].keys():
                    item['sys_summs'][each_sys]['sys_summ'] = item['sys_summs'][each_sys]['sys_summ'].encode('ascii', errors='ignore').decode()
                sys_outputs = item['sys_summs'] # 

                yield TestExample(document, refs, sys_outputs)

    def WriteaData(self, full_item, save_pth):
        with jsonlines.open(save_pth, mode='a') as writer:
            writer.write(full_item)


def get_reference_score(aligner, input_text, context, aligner_type, remove_stopwords):
    if isinstance(input_text, list):
        align_list = []
        for ref in input_text:
            if aligner_type == 'bleurt':
                i = context
                c = ref
            else:
                i = ref
                c = context

            score = aligner.get_score(
                input_text=i,
                context=c,
                remove_stopwords=remove_stopwords)
            align_list.append(score)

        align_score = np.array(align_list).mean()

    else:
        if aligner_type == 'bleurt':
            i = context
            c = input_text
        else:
            i = input_text
            c = context

        align_score = aligner.get_score(
            input_text=i,
            context=c,
            remove_stopwords=remove_stopwords)

    return align_score


def explainaboard_output(dataset_name='qags_xsum',
                         aspect='consistency',
                         aligner_type='disc',
                         pretrain_model='cnndm',
                         disc_init=None,
                         relevance_y_x_init=None,
                         bert_model_type='roberta-large',
                         bert_rescale_with_baseline=False,
                         bert_num_layers=None,
                         dialog_context='fact_history',
                         remove_stopwords=False,
                         aggr_type='mean'):
    disp_model_name = {
        'cnndm': 'CNN/DM',
        'newsroom': 'NEWSROOM',
        'xsum': 'XSUM',
        'roberta': 'RoBERTa-large',
        'bert': 'BERT-base'
    }
    disp_aligner_name = {
        'disc': 'D',
        'bert': 'E',
        'bleurt': 'R'
    }
    dataset = ExplainaboardDataset(dataset_name)
    if aligner_type == 'disc':
        aligner = DiscriminativeAligner.load_from_checkpoint(
            aggr_type=aggr_type, checkpoint_path=disc_init).to('cuda')
        aligner.eval()
        if relevance_y_x_init is not None:
            aligner_y_x = (DiscriminativeAligner
                           .load_from_checkpoint(
                               aggr_type=aggr_type,
                               checkpoint_path=relevance_y_x_init)
                           .to('cuda'))
            aligner_y_x.eval()
    elif aligner_type == 'bert':
        aligner = BERTAligner(
            model_type=bert_model_type,
            num_layers=bert_num_layers,
            aggr_type=aggr_type,
            lang='en',
            device='cuda')
        aligner_y_x = aligner

    for example in tqdm(dataset.ReadaData(), desc='Testing'):
        if isinstance(example, TestExample):
            for each_sys in example.sys_outputs.keys():
                if 'relevance' == aspect:
                    align_r_y = get_reference_score(
                        aligner=aligner,
                        input_text=example.context,
                        context=example.sys_outputs[each_sys]['sys_summ'],
                        aligner_type=aligner_type,
                        remove_stopwords=remove_stopwords)
                    align_y_x = aligner_y_x.get_score(
                        input_text=example.sys_outputs[each_sys]['sys_summ'],
                        context=example.src,
                        remove_stopwords=remove_stopwords)
                
                    pred_score = align_y_x * align_r_y
                    example.sys_outputs[each_sys]['scores'][
                        f'CTC({disp_aligner_name[aligner_type]})({disp_model_name[pretrain_model]})'] = pred_score
                if 'consistency' == aspect:
                    pred_score = aligner.get_score(
                        input_text=example.sys_outputs[each_sys]['sys_summ'], context=example.src, remove_stopwords=remove_stopwords)
                    example.sys_outputs[each_sys]['scores'][
                        f'CTC({disp_aligner_name[aligner_type]})({disp_model_name[pretrain_model]})'] = pred_score
            if dataset_name in ['summeval']:
                dataset.WriteaData({'src': example.src, 'ref_summs': example.context, 'sys_summs': example.sys_outputs},
                                   f'data/Explainaboard/{dataset_name}_{aspect}_{aligner_type}_{pretrain_model}_ctc.jsonl')
            elif dataset_name in ['newsroom']:
                dataset.WriteaData({'src': example.src, 'ref_summ': example.context, 'sys_summs': example.sys_outputs},
                                   f'data/Explainaboard/{dataset_name}_{aspect}_{aligner_type}_{pretrain_model}_ctc.jsonl')
            # print(example.sys_outputs)

        else:
            raise ValueError(
                'require instance of TestExample, but got {}.'.format(type(example)))


if __name__ == '__main__':
    explainaboard_output(dataset_name='newsroom',
                         aspect='relevance',
                         aligner_type='disc',
                         pretrain_model='cnndm', # or 'roberta'
                         disc_init='/home/yzha/ctc_task/ctc-gen-eval/ckpts/cnndm/disc.ckpt',
                         relevance_y_x_init='/home/yzha/ctc_task/ctc-gen-eval/ckpts/cnndm/disc.ckpt',
                         bert_model_type='bert-base-uncased', #'roberta-large', #bert-base-uncased
                         bert_rescale_with_baseline=False,
                         dialog_context='fact_history',
                         aggr_type='mean')
