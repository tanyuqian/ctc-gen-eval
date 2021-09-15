import os

from texar.torch.data.data_utils import maybe_download

from ctc_score.configs import ALIGNS, E_MODEL_TYPES, DR_MODEL_LINKS

from ctc_score.models.discriminative_aligner import DiscriminativeAligner
from ctc_score.models.bert_aligner import BERTAligner


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
                model_type=E_MODEL_TYPES[self._align[2:]],
                aggr_type=self._aggr_type,
                lang='en',
                device=self._device)

        elif self._align.startswith('D'):
            aligner_link = DR_MODEL_LINKS[self._align][aligner_name]
            maybe_download(
                urls=aligner_link,
                path=f'{os.getenv("HOME")}/.cache/',
                filenames=f'ctc_score_models/{self._align}/{aligner_name}.ckpt')

            ckpt_path = f'{os.getenv("HOME")}/.cache/' \
                        f'ctc_score_models/{self._align}/{aligner_name}.ckpt'
            aligner = DiscriminativeAligner.load_from_checkpoint(
                aggr_type=self._aggr_type,
                checkpoint_path=ckpt_path
            ).to(self._device)
            aligner.eval()

        elif aligner_name.startswith('R'):
            pass

        self._aligners[aligner_name] = aligner

        return self._aligners[aligner_name]