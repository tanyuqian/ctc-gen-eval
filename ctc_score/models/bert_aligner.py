from collections import defaultdict

import torch

from bert_score import BERTScorer
from bert_score.utils import get_bert_embedding, sent_encode

from ctc_score.models.aligner import Aligner


class BERTAligner(BERTScorer, Aligner):
    def __init__(self, aggr_type, *args, **kwargs):
        Aligner.__init__(self, aggr_type=aggr_type)
        BERTScorer.__init__(self, *args, **kwargs)

    def get_sim_matrix(self, candidate, reference):
        assert isinstance(candidate, str)
        assert isinstance(reference, str)

        idf_dict = defaultdict(lambda: 1.0)
        idf_dict[self._tokenizer.sep_token_id] = 0
        idf_dict[self._tokenizer.cls_token_id] = 0

        hyp_embedding, masks, padded_idf = get_bert_embedding(
            [candidate], self._model, self._tokenizer, idf_dict,
            device=self.device, all_layers=False,
        )
        ref_embedding, masks, padded_idf = get_bert_embedding(
            [reference], self._model, self._tokenizer, idf_dict,
            device=self.device, all_layers=False,
        )
        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
        sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
        sim = sim.squeeze(0).cpu()

        r_tokens = [self._tokenizer.decode([i])
                    for i in sent_encode(self._tokenizer, reference)][1:-1]
        h_tokens = [self._tokenizer.decode([i])
                    for i in sent_encode(self._tokenizer, candidate)][1:-1]
        sim = sim[1:-1, 1:-1]

        if self.rescale_with_baseline:
            sim = (sim - self.baseline_vals[2].item()) / \
                  (1 - self.baseline_vals[2].item())

        return sim, r_tokens, h_tokens

    def align(self, input_text, context):
        mat, context_tokens, input_tokens = self.get_sim_matrix(
            candidate=input_text, reference=context)

        preds = torch.max(mat, dim=-1).values.tolist()

        return input_tokens, preds
