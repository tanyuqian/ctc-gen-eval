from ctc_score.scorer import Scorer


class DialogScorer(Scorer):
    def __init__(self, align, aggr_type='sum', device='cuda'):
        Scorer.__init__(self, align=align, aggr_type=aggr_type, device=device)

    def score(self, fact, dialog_history, hypo, aspect, remove_stopwords=True):
#         if 'topical_chat' in self._align:
#             fact = 'fact: ' + fact

        kwargs = dict(
            fact=fact,
            dialog_history=dialog_history,
            hypo=hypo,
            remove_stopwords=remove_stopwords)

        if aspect == 'groundedness':
            return self.score_groundedness(**kwargs)
        elif aspect == 'engagingness':
            return self.score_engagingness(**kwargs)
        else:
            raise NotImplementedError

    def score_groundedness(self, fact, dialog_history, hypo, remove_stopwords):
        context = fact
        aligner = self._get_aligner('fact_to_response')

        return aligner.get_score(
            context=context,
            input_text=hypo,
            remove_stopwords=remove_stopwords)

    def score_engagingness(self, fact, dialog_history, hypo, remove_stopwords):
        context = '\n\n\n'.join([fact.strip(), dialog_history.strip()])
        aligner = self._get_aligner('fact_history_to_response')

        return aligner.get_score(
            context=context,
            input_text=hypo,
            remove_stopwords=remove_stopwords)
