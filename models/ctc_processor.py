from forte.common import Resources
from forte.common.configuration import Config
from forte.data import MultiPack
from forte.processors.base import MultiPackProcessor

from data_utils.ft.onto.ctc import Metric
from .discriminative_aligner import DiscriminativeAligner
from .bert_aligner import BERTAligner


class DiscModelProcessor(MultiPackProcessor):
    def __init__(self, ckpt_path: str, aggr_type: str, device: str):
        super(DiscModelProcessor, self).__init__()
        self.aligner = DiscriminativeAligner.load_from_checkpoint(
            aggr_type=aggr_type, checkpoint_path=ckpt_path).to(device)
        self.aligner.eval()
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
                 aggr_type: str, lang: str, device: str):
        super(BertModelProcessor, self).__init__()
        self.aligner = BERTAligner(
            model_type=model_type,
            rescale_with_baseline=rescale_with_baseline,
            aggr_type=aggr_type,
            lang=lang,
            device=device
        )
        print('init success!')

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

