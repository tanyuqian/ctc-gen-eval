from ctc_score.scorer import Scorer


class DialogScorer(Scorer):
    def __init__(self, dataset, align, aligner_configs=None):
        Scorer.__init__(self, dataset, align, aligner_configs)

        assert self._dataset in ['topical_chat', 'persona_chat']

    def score(self, fact, dialog_history, hypo, aspect, remove_stopwords=True):
        assert aspect in ['engagingness', 'groundedness']

        if self._dataset == 'topical_chat':
            fact = 'fact: ' + fact

        if aspect == 'groundedness':
            context = fact
            aligner = self._get_aligner(f'{self._align}-{self._dataset}-fact')

        elif aspect == 'engagingness':
            context = '\n\n\n'.join([fact.strip(), dialog_history.strip()])
            aligner = self._get_aligner(
                f'{self._align}-{self._dataset}-fact_history')

        return aligner.get_score(
            input_text=hypo,
            context=context,
            remove_stopwords=remove_stopwords)
