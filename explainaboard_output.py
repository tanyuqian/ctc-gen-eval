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
                # for each_sys in item['sys_summs'].keys():
                #     summary = item['sys_summs'][each_sys]['sys_summ']
                #     scores = summary = item['sys_summs'][each_sys]['scores']

                #     yield TestExample(document, refs, summary, scores)
                # yield 0

    def WriteaData(self, full_item):
        with jsonlines.open(f'data/Explainaboard/{self.dataset_name}_ctc.jsonl', mode='a') as writer:
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
                if aspect == 'relevance':
                    align_y_x = aligner.get_score(
                        input_text=example.sys_outputs[each_sys]['sys_summ'],
                        context=example.src)
                    align_r_y = aligner.get_score(
                        input_text=example.context,
                        context=example.sys_outputs[each_sys]['sys_summ'])
                    pred_score = align_y_x * align_r_y
                    example.sys_outputs[each_sys]['scores']['ctc_relevance'] = pred_score
                elif aspect == 'consistency':
                    pred_score = aligner.get_score(
                        input_text=example.sys_outputs[each_sys]['sys_summ'], context=example.src)
                    example.sys_outputs[each_sys]['scores']['ctc_consistency'] = pred_score

            dataset.WriteaData({'src':example.src, 'ref_summs':example.context, 'sys_summs':example.sys_outputs})
            # print(example.sys_outputs)

        else:
            raise ValueError(
                'require instance of TestExample, but got {}.'.format(type(example)))


if __name__ == '__main__':
    explainaboard_output(dataset_name='summeval',
                         aspect='relevance',
                         aligner_type='bert',
                         disc_init=None,
                         bert_model_type='roberta-large',
                         bert_rescale_with_baseline=False,
                         dialog_context='fact_history',
                         aggr_type='mean')
