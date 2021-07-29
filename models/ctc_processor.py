from forte.common import Resources
from forte.common.configuration import Config
from forte.data import MultiPack
from forte.processors.base import MultiPackProcessor

from data_utils.ft.onto.ctc import Metric
from .discriminative_aligner import DiscriminativeAligner
from .bert_aligner import BERTAligner


class DiscModelProcessor(MultiPackProcessor):
    def __init__(self, ckpt_path: str, aggr_type: str, device: str, aspect: str):
        super(DiscModelProcessor, self).__init__()
        self.aligner = DiscriminativeAligner.load_from_checkpoint(
            aggr_type=aggr_type, checkpoint_path=ckpt_path).to(device)
        self.aligner.eval()
        self.aspect = aspect
        print('init success!')
        # todo: checkpoint path

    def _process(self, input_pack: MultiPack):
        document_pack = input_pack.get_pack('document')
        summary_pack = input_pack.get_pack('summary')

        align_score_metric = Metric(summary_pack)
        align_score_metric.metric_name = 'pred_consistency'
        align_score_metric.metric_value = self.aligner.get_score(
            input_text=summary_pack.text,
            context=document_pack.text
        )

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)


class BertModelProcessor(MultiPackProcessor):
    def __init__(self, model_type: str, rescale_with_baseline: bool,
                 aggr_type: str, lang: str, device: str, aspect: str, context: str):
        super(BertModelProcessor, self).__init__()
        self.aligner = BERTAligner(
            model_type=model_type,
            rescale_with_baseline=rescale_with_baseline,
            aggr_type=aggr_type,
            lang=lang,
            device=device
        )
        self.aspect = aspect
        self.context = context
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

        # if self.context == 'fact':
        #     context_text = input_pack.get_pack('fact')
        # elif self.context == 'history':
        #     context_text = input_pack.get_pack('history')
        # elif self.context == 'fact_history':
        #     context_text = input_pack.get_pack('fact_history')
        # elif self.context == 'history_fact':
        #     context_text = input_pack.get_pack('history_fact')
        # else:
        #     raise ValueError('context: {} not recognized'.format(self.context))

        engagingness_metric = Metric(response_pack)
        engagingness_metric.metric_name = 'pred_engagingness'
        engagingness_metric.metric_value = self.aligner.get_score(input_text=response_pack.text,
                                                                  context=his_fact_pack.text)

        groundness_metric = Metric(response_pack)
        groundness_metric.metric_name = 'pred_groundness'
        groundness_metric.metric_value = self.aligner.get_score(input_text=response_pack.text,
                                                                context=fact_pack.text)
