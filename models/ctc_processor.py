import os
from abc import ABC

from forte.common import Resources
from forte.common.configuration import Config
from forte.data import MultiPack
from forte.processors.base import MultiPackProcessor

from data_utils.ft.onto.ctc import Metric
from .discriminative_aligner import DiscriminativeAligner
from .bert_aligner import BERTAligner
from download_model_data import download_model

from scipy.stats.stats import spearmanr, pearsonr, kendalltau


class AlignModelProcessor(MultiPackProcessor):
    def __init__(self, model_type: str, rescale_with_baseline: bool,
                 aggr_type: str, lang: str, device: str, aspect: str,
                 context: str, ckpt_path: str, aligner_type: str, dataset_name: str):
        super(AlignModelProcessor, self).__init__()
        self.params = {
            'model_type': model_type,
            'rescale_with_baseline': rescale_with_baseline,
            'aggr_type': aggr_type,
            'lang': lang,
            'device': device,
            'context': context,
            'ckpt_path': ckpt_path,
            'aligner_type': aligner_type,
            'dataset_name': dataset_name
        }
        self.aspect = aspect
        self.download_dict = {
            'qags_xsum': 'xsum',
            'qags_cnndm': 'cnndm',
            'summeval': 'cnndm',
            'yelp': 'yelp',
            'persona_chat': 'persona_chat',
            'topical_chat': 'topical_chat'
        }
        print('init success!')

    def _process(self, input_pack: MultiPack):
        if self.aspect == 'consistency':
            self.process_consistency(input_pack)
        elif self.aspect == 'preservation':
            self.process_preservation(input_pack)
        elif self.aspect == 'relevance':
            self.process_relevance(input_pack)
        elif self.aspect in ['engagingness', 'groundness']:
            self.process_dialogs(input_pack)
        else:
            raise ValueError('Aspect type: {} not recognized'.format(self.aspect))

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        if self.params['aligner_type'] == 'bert':
            self.aligner = BERTAligner(
                model_type=self.params['model_type'],
                rescale_with_baseline=self.params['rescale_with_baseline'],
                aggr_type=self.params['aggr_type'],
                lang=self.params['lang'],
                device=self.params['device']
            )
        elif self.params['aligner_type'] == 'disc':
            if self.params['ckpt_path'] is not None:
                act_ckpt_path = self.params['ckpt_path']
            else:
                print('downloading model: {}'.format(self.download_dict[self.params['dataset_name']]))
                download_model(self.download_dict[self.params['dataset_name']], 'ckpts/', self.params['context'])
                if self.params['context'] is None:
                    act_ckpt_path = os.path.join('ckpts', self.download_dict[self.params['dataset_name']], 'disc.ckpt')
                else:
                    act_ckpt_path = os.path.join('ckpts', self.download_dict[self.params['dataset_name']],
                                                 'disc_' + self.params['context'] + '.ckpt')
            self.aligner = DiscriminativeAligner.load_from_checkpoint(
                aggr_type=self.params['aggr_type'], checkpoint_path=act_ckpt_path).to(self.params['device'])
            self.aligner.eval()
        else:
            raise ValueError('Aligner type: {} not recognized'.format(self.params['aligner_type']))

    def process_consistency(self, input_pack: MultiPack):
        document_pack = input_pack.get_pack('document')
        summary_pack = input_pack.get_pack('summary')

        consis_score_metric = Metric(summary_pack)
        consis_score_metric.metric_name = 'pred_consistency'
        consis_score_metric.metric_value = self.aligner.get_score(
            input_text=summary_pack.text,
            context=document_pack.text
        )

    def process_relevance(self, input_pack: MultiPack):
        document_pack = input_pack.get_pack('document')
        summary_pack = input_pack.get_pack('summary')
        refs_pack = input_pack.get_pack('references')

        refs_score_metric = Metric(summary_pack)
        refs_score_metric.metric_name = 'pred_relevance'
        refs_score_metric.metric_value = self.aligner.get_score(input_text=summary_pack.text,
                                                                context=document_pack.text) * self.aligner.get_score(
            input_text=refs_pack.text,
            context=summary_pack.text)

    def process_preservation(self, input_pack: MultiPack):
        input_sent_pack = input_pack.get_pack('input_sent')
        output_sent_pack = input_pack.get_pack('output_sent')

        preservation_metric = Metric(output_sent_pack)
        preservation_metric.metric_name = 'pred_preservation'
        align_y_x = self.aligner.get_score(input_text=input_sent_pack.text,
                                           context=output_sent_pack.text)
        align_x_y = self.aligner.get_score(input_text=output_sent_pack.text,
                                           context=input_sent_pack.text)
        preservation_metric.metric_value = (align_y_x * align_x_y) / (align_y_x + align_x_y)

    def process_dialogs(self, input_pack: MultiPack):
        response_pack = input_pack.get_pack('response')
        his_fact_pack = input_pack.get_pack('history_fact')
        fact_pack = input_pack.get_pack('fact')

        engagingness_metric = Metric(response_pack)
        engagingness_metric.metric_name = 'pred_engagingness'
        engagingness_metric.metric_value = self.aligner.get_score(input_text=response_pack.text,
                                                                  context=his_fact_pack.text)

        groundness_metric = Metric(response_pack)
        groundness_metric.metric_name = 'pred_groundness'
        groundness_metric.metric_value = self.aligner.get_score(input_text=response_pack.text,
                                                                context=fact_pack.text)


class CorrelationProcessor(MultiPackProcessor):
    def __init__(self, aspect):
        super(CorrelationProcessor, self).__init__()
        self.aspect = aspect

        self.pred_scores = []
        self.true_scores = []

        self.pearson_score = 0.0
        self.spearman_score = 0.0
        self.kendall_score = 0.0

    def _process(self, pack: MultiPack):
        if self.aspect in ['consistency', 'relevance']:
            ans_pack = pack.get_pack('summary')
        elif self.aspect in ['preservation']:
            ans_pack = pack.get_pack('output_sent')
        elif self.aspect in ['engagingness', 'groundness']:
            ans_pack = pack.get_pack('response')

        gen_dict = dict()
        for each_generic in ans_pack.all_generic_entries:
            gen_dict[each_generic.metric_name] = each_generic.metric_value
        # print(gen_dict)
        if gen_dict['pred_' + self.aspect] is not None:
            self.pred_scores.append(gen_dict['pred_' + self.aspect])
            self.true_scores.append(gen_dict[self.aspect])
        try:
            self.pearson_score = pearsonr(self.pred_scores, self.true_scores)[0]
            self.spearman_score = spearmanr(self.pred_scores, self.true_scores)[0]
            self.kendall_score = kendalltau(self.pred_scores, self.true_scores)[0]
        except:
            self.pearson_score = 0.0
            self.spearman_score = 0.0
            self.kendall_score = 0.0
