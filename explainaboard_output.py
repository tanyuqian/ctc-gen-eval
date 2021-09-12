import os
import json
import jsonlines
import fire
from tqdm import tqdm

from data_utils.data_utils import get_test_examples
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
                refs = ' '.join(item['ref_summs'])
                sys_outputs = item['sys_summs']

                yield TestExample(document, refs, sys_outputs)
        elif self.dataset_name in ['newsroom']:
            for item in self.org_dataset:
                document = item['src']
                refs = item['ref_summ']
                sys_outputs = item['sys_summs']

                yield TestExample(document, refs, sys_outputs)

    def WriteaData(self, full_item, save_pth):
        with jsonlines.open(save_pth, mode='a') as writer:
            writer.write(full_item)


def explainaboard_output(dataset_name='qags_xsum',
                         aspect='consistency',
                         aligner_type='disc',
                         disc_init=None,
                         bert_model_type='roberta-large',
                         bert_rescale_with_baseline=False,
                         dialog_context='fact_history',
                         aggr_type='mean'):
    dataset = ExplainaboardDataset(dataset_name)
    if aligner_type == 'disc':
        aligner = DiscriminativeAligner.load_from_checkpoint(
            aggr_type=aggr_type, checkpoint_path=disc_init).to('cuda')
        aligner.eval()
    elif aligner_type == 'bert':
        aligner = BERTAligner(
            model_type=bert_model_type,
            rescale_with_baseline=bert_rescale_with_baseline,
            aggr_type=aggr_type,
            lang='en',
            device='cuda')

    for example in tqdm(dataset.ReadaData(), desc='Testing'):
        if isinstance(example, TestExample):
            for each_sys in example.sys_outputs.keys():
                if 'relevance' in aspect:
                    align_y_x = aligner.get_score(
                        input_text=example.sys_outputs[each_sys]['sys_summ'],
                        context=example.src)
                    align_r_y = aligner.get_score(
                        input_text=example.context,
                        context=example.sys_outputs[each_sys]['sys_summ'])
                    if align_y_x is not None and align_r_y is not None:
                        pred_score = align_y_x * align_r_y
                    else:
                        pred_score = None
                    example.sys_outputs[each_sys]['scores']['ctc_relevance'] = pred_score
                if 'consistency' in aspect:
                    pred_score = aligner.get_score(
                        input_text=example.sys_outputs[each_sys]['sys_summ'], context=example.src)
                    example.sys_outputs[each_sys]['scores']['ctc_consistency'] = pred_score

            dataset.WriteaData({'src': example.src, 'ref_summs': example.context, 'sys_summs': example.sys_outputs},
                               f'data/Explainaboard/{dataset_name}_{aspect}_{aligner_type}_ctc.jsonl')
            # print(example.sys_outputs)

        else:
            raise ValueError(
                'require instance of TestExample, but got {}.'.format(type(example)))


if __name__ == '__main__':
    explainaboard_output(dataset_name='summeval',
                         aspect='consistency_relevance',
                         aligner_type='disc',
                         disc_init='/home/yzha/ctc_task/ctc-gen-eval/ckpts/newsroom/disc.ckpt',
                         bert_model_type='roberta-large',
                         bert_rescale_with_baseline=False,
                         dialog_context='fact_history',
                         aggr_type='mean')
