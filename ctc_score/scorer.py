import os
import json
import ctc_score

from texar.torch.data.data_utils import maybe_download

from ctc_score.models.discriminative_aligner import DiscriminativeAligner
from ctc_score.models.bert_aligner import BERTAligner

from ctc_score import DialogScorer


class Scorer:
    def __init__(self, dataset, align, aligner_configs):
        self._dataset = dataset

        self._align = align
        self._aligners = {}
        self._aligner_configs = aligner_configs
        self._fix_aligner_configs()

    def score(self, *args, **kwargs):
        raise NotImplementedError

    def _fix_aligner_configs(self):
        if self._aligner_configs is None:
            self._aligner_configs = {}

        default_configs = json.load(
            open(f'{ctc_score.__path__}/default_configs.json'))
        for key, value in default_configs[self._align].items():
            if key not in self._aligner_configs:
                self._aligner_configs[key] = value

    def _get_aligner(self, aligner_name):
        if aligner_name.startswith('E'):
            aligner_name = 'E'

        if aligner_name not in self._aligners:
            aggr_type = 'sum' if isinstance(self, DialogScorer) else 'mean'

            if aligner_name == 'E':
                aligner = BERTAligner(
                    **self._aligner_configs, aggr_type=aggr_type)

            elif aligner_name.startswith('D'):
                aligner_link = self._aligner_configs[aligner_name[2:]]
                maybe_download(
                    urls=aligner_link,
                    path=f'{os.getenv("HOME")}/.cache/',
                    filenames=f'{aligner_name}.ckpt')
                ckpt_path = f'{os.getenv("HOME")}/.cache/{aligner_name}.ckpt'

                aligner = DiscriminativeAligner.load_from_checkpoint(
                    aggr_type=aggr_type,
                    checkpoint_path=ckpt_path
                ).to(self._aligner_configs['device'])
                aligner.eval()

            elif aligner_name.startswith('R'):
                pass

            self._aligners[aligner_name] = aligner

        return self._aligners[aligner_name]