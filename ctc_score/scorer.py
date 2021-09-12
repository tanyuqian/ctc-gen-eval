import json

import ctc_score
from ctc_score.models.discriminative_aligner import DiscriminativeAligner
from ctc_score.models.bert_aligner import BERTAligner
# from ctc_score.models.bleurt_aligner import BLEURTAligner


class Scorer:
    def __init__(self, dataset, aligner, aligner_configs=None):
        self._dataset = dataset
        self._aligner = aligner
        self._aligner_configs = aligner_configs

        if isinstance(self._aligner, str):
            self._fix_aligner_configs()

            if aligner == 'D':
                pass
            elif aligner == 'E':
                self._aligner = BERTAligner(**self._aligner_configs)
            elif aligner == 'R':
                pass

    def _fix_aligner_configs(self):
        if self._aligner_configs is None:
            self._aligner_configs = {}

        if 'aggr_type' not in self._aligner_configs and \
                self._dataset in ['persona_chat', 'topical_chat']:
            self._aligner_configs['aggr_type'] = 'sum'

        default_configs = json.load(
            open(f'{ctc_score.__path__}/default_configs.json'))
        for key, value in default_configs[self._aligner].items():
            if key not in self._aligner_configs:
                self._aligner_configs[key] = value

    def score(self, *args, **kwargs):
        raise NotImplementedError


