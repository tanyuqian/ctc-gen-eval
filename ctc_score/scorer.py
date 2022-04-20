import os

from ctc_score.download import maybe_download
from ctc_score.configs import ALIGNS, E_MODEL_CONFIGS, DR_MODEL_LINKS

from ctc_score.models.discriminative_aligner import DiscriminativeAligner
from ctc_score.models.bert_aligner import BERTAligner
from ctc_score.models.bleurt_aligner_pt import BLEURTAligner
import torch


class Scorer:
    def __init__(self, align, aggr_type, device):
        assert align in ALIGNS

        self._align = align
        self._aligners = {}

        self._aggr_type = aggr_type
        self._device = device

    def score(self, *args, **kwargs):
        raise NotImplementedError

    def _get_aligner(self, aligner_name):
        if aligner_name in self._aligners:
            return self._aligners[aligner_name]

        if self._align.startswith('E'):
            aligner = BERTAligner(
                **E_MODEL_CONFIGS[self._align[2:]],
                aggr_type=self._aggr_type,
                lang='en',
                device=self._device)
        else:
            aligner_link = DR_MODEL_LINKS[self._align][aligner_name]

            os.makedirs(
                f'{os.getenv("HOME")}/.cache/ctc_score_models/{self._align}/',
                exist_ok=True)
            maybe_download(
                urls=aligner_link,
                path=f'{os.getenv("HOME")}/.cache/',
                filenames=f'ctc_score_models/{self._align}/{aligner_name}.ckpt',
                num_gdrive_retries=3)
            ckpt_path = f'{os.getenv("HOME")}/.cache/' \
                        f'ctc_score_models/{self._align}/{aligner_name}.ckpt'

            if self._align.startswith('D'):
                aligner = DiscriminativeAligner(
                    aggr_type=self._aggr_type).to(self._device)
                aligner.load_state_dict(torch.load(ckpt_path))
                aligner.eval()
            else:
                assert self._align.startswith('R')
                aligner = BLEURTAligner(
                    aggr_type=self._aggr_type,
                    checkpoint=ckpt_path,
                    device=self._device)

        self._aligners[aligner_name] = aligner
        return self._aligners[aligner_name]
