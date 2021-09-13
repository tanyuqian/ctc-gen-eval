from ctc_score.scorer import Scorer


class DialogScorer(Scorer):
    def __init__(self, dataset, align, aligner_configs=None):
        Scorer.__init__(self, dataset, align, aligner_configs)

        assert self._dataset in ['topical_chat', 'persona_chat']

    def score(self, fact, dialog_history, hypo, aspect, remove_stopwords=True):
        if self._dataset == 'topical_chat':
            fact = 'fact: ' + fact

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
        aligner = self._get_aligner(f'{self._align}-{self._dataset}-fact')

        return aligner.get_score(
            context=context,
            input_text=hypo,
            remove_stopwords=remove_stopwords)

    def score_engagingness(self, fact, dialog_history, hypo, remove_stopwords):
        context = '\n\n\n'.join([fact.strip(), dialog_history.strip()])
        aligner = self._get_aligner(
            f'{self._align}-{self._dataset}-fact_history')

        return aligner.get_score(
            context=context,
            input_text=hypo,
            remove_stopwords=remove_stopwords)
