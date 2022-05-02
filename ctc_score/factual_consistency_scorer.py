from ctc_score.scorer import Scorer


class FactualConsistencyScorer(Scorer):
    def __init__(self, align, aggr_type='mean', device='cuda'):
        Scorer.__init__(self, align=align, aggr_type=aggr_type, device=device)

    def score(self, grounding, hypo, aspect='consistency', remove_stopwords=False):
        kwargs = dict(
            grounding=grounding,
            hypo=hypo,
            remove_stopwords=remove_stopwords)

        if aspect == 'consistency':
            return self.score_consistency(**kwargs)
        else:
            raise NotImplementedError

    def score_consistency(self, grounding, hypo, remove_stopwords):
        aligner = self._get_aligner('doc_to_summ')

        return aligner.get_score(
            context=grounding,
            input_text=hypo,
            remove_stopwords=remove_stopwords)