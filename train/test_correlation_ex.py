import os
import json
import jsonlines
import fire
from tqdm import tqdm

from models.discriminative_aligner import DiscriminativeAligner
from models.bert_aligner import BERTAligner

from scipy.stats.stats import spearmanr, pearsonr, kendalltau
from collections import namedtuple

from scipy.stats.stats import spearmanr, pearsonr, kendalltau

TestExample = namedtuple('TestExample', ['src', 'context', 'sys_outputs'])


class ExplainaboardDatasetOut(object):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.org_dataset = []
        with open(f'data/Explainaboard/{dataset_name}.jsonl', 'r+', encoding='utf-8') as f:
            for each in jsonlines.Reader(f):
                self.org_dataset.append(each)

    def ReadaData(self):
        if 'summeval' in self.dataset_name:
            for item in self.org_dataset:
                document = item['src']
                refs = ' '.join(item['ref_summs'])
                sys_outputs = item['sys_summs']

                yield TestExample(document, refs, sys_outputs)
        elif 'newsroom' in self.dataset_name:
            for item in self.org_dataset:
                document = item['src']
                refs = item['ref_summs']
                sys_outputs = item['sys_summs']

                yield TestExample(document, refs, sys_outputs)

    def WriteaData(self, full_item, save_pth):
        with jsonlines.open(save_pth, mode='a') as writer:
            writer.write(full_item)


def test_correlation_output(dataset_name, aspect):
    dataset = ExplainaboardDatasetOut(dataset_name)
    ctc_metrics = []
    gt_metrics = []
    counter = 0
    ctc_name = f'CTC(D)(XSUM)'
    for example in tqdm(dataset.ReadaData(), desc='Testing'):
        for each_sys in example.sys_outputs.keys():
            gt_metric = example.sys_outputs[each_sys]['scores'][aspect]
            ctc_metric = example.sys_outputs[each_sys]['scores'][ctc_name]
            counter = counter + 1
            if ctc_metric is not None:
                ctc_metrics.append(ctc_metric)
                gt_metrics.append(gt_metric)

    pearson_score = pearsonr(ctc_metrics, gt_metrics)[0]
    spearman_score = spearmanr(ctc_metrics, gt_metrics)[0]
    kendall_score = kendalltau(ctc_metrics, gt_metrics)[0]

    print(f'# test sents: {len(ctc_metrics)} of total: {counter}')
    print(f'pearson: {pearson_score:.4f}')
    print(f'spearman: {spearman_score:.4f}')
    print(f'kendall: {kendall_score:.4f}')


if __name__ == "__main__":
    # test_correlation_output('newsroom_relevance_disc_ctc', 'relevance')
    test_correlation_output('summeval_relevance_disc_xsum_ctc', 'relevance')
    # test_correlation_output('summeval_consistency_disc_xsum_ctc', 'consistency')
