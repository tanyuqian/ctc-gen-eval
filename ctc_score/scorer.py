import json

import ctc_score
from ctc_score.models.discriminative_aligner import DiscriminativeAligner
from ctc_score.models.bert_aligner import BERTAligner
# from ctc_score.models.bleurt_aligner import BLEURTAligner


class Scorer:
    def __init__(self, dataset, aligner, aligner_configs=None):
        self._dataset = dataset

        if aligner not in ['E', 'D', 'R']:
            self._aligner = aligner
        else:
            self._aligner_configs = \
                set_aligner_configs(dataset, aligner, aligner_configs)

            if aligner == 'D':
                pass
            elif aligner == 'E':
                self._aligner = BERTAligner(**self._aligner_configs)
            elif aligner == 'R':
                pass

    def score(self, *args, **kwargs):
        raise NotImplementedError


def set_aligner_configs(dataset, aligner, aligner_configs):
    if aligner_configs is None:
        aligner_configs = {}

    default_configs = json.load(
        open(f'{ctc_score.__path__}/default_configs.json'))
    for key, value in default_configs[aligner].items():
        if key not in aligner_configs:
            aligner_configs[key] = value

    if 'aggr_type' not in aligner_configs:
        aligner_configs['aggr_type'] = \
            'sum' if dataset in ['persona_chat', 'topical_chat'] else 'mean'

    return aligner_configs