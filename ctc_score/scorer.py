from ctc_score.models.discriminative_aligner import DiscriminativeAligner
from ctc_score.models.bert_aligner import BERTAligner
from ctc_score.models.bleurt_aligner import BLEURTAligner


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
                self._aligner = BERTAligner(**aligner_configs)
            elif aligner == 'R':
                pass

    def score(self, *args, **kwargs):
        raise NotImplementedError


default_configs = {
    'E': {
        'model_type': 'bert-base',
        'lang': 'en',
        'device': 'cuda'
    }
}


def set_aligner_configs(dataset, aligner, aligner_configs):
    if aligner_configs is None:
        aligner_configs = {}

    for key, value in default_configs[aligner].items():
        if key not in aligner_configs:
            aligner_configs[key] = value

    if 'aggr_type' not in aligner_configs:
        aligner_configs['aggr_type'] = \
            'sum' if dataset in ['persona_chat', 'topical_chat'] else 'mean'

    return aligner_configs