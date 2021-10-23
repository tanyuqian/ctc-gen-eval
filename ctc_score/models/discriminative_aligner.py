from collections import Counter
import os
from typing import List

import torch
from torch import nn

from ctc_score.models.aligner import Aligner

from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

from ctc_score.data_utils.data_utils import TokenClassificationExample, \
    text_clean, get_words


INIT = 'roberta.large'
MAX_LENGTH = 512


class DiscriminativeAligner(Aligner, nn.Module):
    def __init__(self, aggr_type):
        Aligner.__init__(self, aggr_type)
        nn.Module.__init__(self)
        self._roberta = RobertaModel.from_pretrained('roberta-large')
        self._roberta_tokenizer = RobertaTokenizerFast.from_pretrained(
            'roberta-large')

        self._classifier = nn.Linear(
            self._roberta.config.hidden_size, 2)
        self._tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        self._hparams = None

    def forward(self, input_text, words, context):
        if len(self._tokenizer(input_text)['input_ids']) > MAX_LENGTH:
            print('Length of input text exceeds max length! Skipping')
            return None

        tokens = self._roberta_tokenizer(
            input_text, context, return_tensors='pt').to('cuda')
        tokens = {'input_ids': tokens['input_ids'][:, :MAX_LENGTH],
                  'attention_mask': tokens['attention_mask'][:, :MAX_LENGTH]}
        features = self._roberta(**tokens).last_hidden_state[0]

        try:
            len_input_text = len(
                self._roberta_tokenizer(input_text)['input_ids'])
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
        bpe_toks = self._roberta_tokenizer(
            input_text, return_tensors='pt')['input_ids'][0]

        alignment = self.align_bpe_to_words(bpe_toks, words)
        aligned_feats = self.align_features_to_words(features, alignment)

        assert len(words) + 2 == len(aligned_feats), 'words length {} does not match feature lenth {}. '.format(
            len(words), len(aligned_feats))

        return aligned_feats[1:-1]

    def align_features_to_words(self, features, alignment):
        """
        Align given features to words.

        Args:
            roberta (RobertaHubInterface): RoBERTa instance
            features (torch.Tensor): features to align of shape `(T_bpe x C)`
            alignment: alignment between BPE tokens and words returned by
                func:`align_bpe_to_words`.
        """
        assert features.dim() == 2

        bpe_counts = Counter(
            j for bpe_indices in alignment for j in bpe_indices)
        assert bpe_counts[0] == 0  # <s> shouldn't be aligned
        denom = features.new([bpe_counts.get(j, 1)
                             for j in range(len(features))])
        weighted_features = features / denom.unsqueeze(-1)

        output = [weighted_features[0]]
        largest_j = -1
        for bpe_indices in alignment:
            output.append(weighted_features[bpe_indices].sum(dim=0))
            largest_j = max(largest_j, *bpe_indices)
        for j in range(largest_j + 1, len(features)):
            output.append(weighted_features[j])
        output = torch.stack(output)
        # assert torch.all(torch.abs(output.sum(dim=0) - features.sum(dim=0)) < 1e-4)
        return output

    def align_bpe_to_words(self, bpe_tokens: torch.LongTensor, other_tokens: List[str]):
        """
        Helper to align GPT-2 BPE to other tokenization formats (e.g., spaCy).

        Args:
            roberta (RobertaHubInterface): RoBERTa instance
            bpe_tokens (torch.LongTensor): GPT-2 BPE tokens of shape `(T_bpe)`
            other_tokens (List[str]): other tokens of shape `(T_words)`

        Returns:
            List[str]: mapping from *other_tokens* to corresponding *bpe_tokens*.
        """
        assert bpe_tokens.dim() == 1
        assert bpe_tokens[0] == 0

        def clean(text):
            return text.strip()

        # remove whitespaces to simplify alignment
        bpe_tokens = [self._roberta_tokenizer.decode([x]) for x in bpe_tokens]
        bpe_tokens = [clean(tok) if tok not in ['<s>', '</s>']
                      else '' for tok in bpe_tokens]

        other_tokens = [clean(str(o)) for o in other_tokens]

        # strip leading <s>
        bpe_tokens = bpe_tokens[1:]
        assert "".join(bpe_tokens) == "".join(other_tokens), 'bpe tokens:{}, other tokens:{}'.format(
            ''.join(bpe_tokens), ''.join(other_tokens))

        # create alignment from every word to a list of BPE tokens
        alignment = []
        bpe_toks = filter(
            lambda item: item[1] != "", enumerate(bpe_tokens, start=1))
        j, bpe_tok = next(bpe_toks)
        for other_tok in other_tokens:
            bpe_indices = []
            while True:
                if other_tok.startswith(bpe_tok):
                    bpe_indices.append(j)
                    other_tok = other_tok[len(bpe_tok):]
                    try:
                        j, bpe_tok = next(bpe_toks)
                    except StopIteration:
                        j, bpe_tok = None, None
                elif bpe_tok.startswith(other_tok):
                    # other_tok spans multiple BPE tokens
                    bpe_indices.append(j)
                    bpe_tok = bpe_tok[len(other_tok):]
                    other_tok = ""
                else:
                    raise Exception(
                        'Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
                if other_tok == "":
                    break
            assert len(bpe_indices) > 0
            alignment.append(bpe_indices)
        assert len(alignment) == len(other_tokens)

        return alignment

    def align(self, input_text, context):
        input_text, context = text_clean(input_text), text_clean(context)

        input_text = self._roberta_tokenizer.decode(
            self._roberta_tokenizer(input_text, return_tensors='pt')['input_ids'][0][:MAX_LENGTH // 2])

        input_text = input_text.replace('<s>', '').replace('</s>', '')
        word_logits = self(
            input_text=input_text,
            words=get_words(input_text),
            context=context)

        if word_logits is None:
            return None, None

        preds = torch.softmax(word_logits, dim=-1)[:, 0]

        return get_words(input_text), preds.tolist()
