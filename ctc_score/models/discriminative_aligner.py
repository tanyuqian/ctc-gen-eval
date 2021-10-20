import os

import torch
from torch import nn

from fairseq.models.roberta import RobertaModel
from fairseq.models.roberta import alignment_utils

from ctc_score.models.aligner import Aligner

from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

from ctc_score.data_utils.data_utils import TokenClassificationExample, \
    text_clean, get_words


INIT = 'roberta.large'
MAX_LENGTH = 512


class DiscriminativeAligner(Aligner, nn.Module):
    def __init__(self, aggr_type):
        Aligner.__init__(self, aggr_type)
        nn.Module.__init__(self)

        cache_dir = f'{os.getenv("HOME")}/.cache/'
        if not os.path.exists(f'{cache_dir}/{INIT}'):
            os.system(f'wget https://dl.fbaipublicfiles.com/fairseq/models/'
                      f'{INIT}.tar.gz -P {cache_dir}')
            os.system(f'tar -xzvf {cache_dir}/{INIT}.tar.gz -C {cache_dir}')

        self._roberta = RobertaModel.from_pretrained(f'{cache_dir}/{INIT}')
        self._classifier = nn.Linear(
            self._roberta.model.args.encoder_embed_dim, 2)
        self._tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        self._hparams = None

    def forward(self, input_text, words, context):
        if len(self._tokenizer(input_text)['input_ids']) > MAX_LENGTH: 
            print('Length of input text exceeds max length! Skipping')
            return None
        
        tokens = self._roberta.encode(
            input_text, context)[:MAX_LENGTH].unsqueeze(0)
        features = self._roberta.extract_features(tokens=tokens)[0]

        try:
            len_input_text = self._roberta.encode(input_text).shape[0]
            word_features = self.extract_features_aligned_to_words(
                input_text=input_text,
                words=words,
                features=features[:len_input_text])
        except:
            print(f'Error bpe-to-words, '
                  f'word_logits=None for this batch: {input_text}')
            return None

        word_logits = self._classifier(word_features)

        return word_logits

    def extract_features_aligned_to_words(self, input_text, words, features):
        bpe_toks = self._roberta.encode(input_text)

        alignment = alignment_utils.align_bpe_to_words(
            self._roberta, bpe_toks, words)
        aligned_feats = alignment_utils.align_features_to_words(
            self._roberta, features, alignment)

        assert len(words) + 2 == len(aligned_feats)

        return aligned_feats[1:-1]

    def align(self, input_text, context):
        input_text, context = text_clean(input_text), text_clean(context)

        input_text = self._roberta.decode(
            self._roberta.encode(input_text)[:MAX_LENGTH // 2])

        word_logits = self(
            input_text=input_text,
            words=get_words(input_text),
            context=context)

        if word_logits is None:
            return None, None

        preds = torch.softmax(word_logits, dim=-1)[:, 0]

        return get_words(input_text), preds.tolist()