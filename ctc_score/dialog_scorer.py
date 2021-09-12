from ctc_score.scorer import Scorer


class DialogScorer(Scorer):
    def __init__(self, dataset, aligner, aligner_configs=None):
        Scorer.__init__(self, dataset, aligner, aligner_configs)

        assert self._dataset in ['topical_chat', 'persona_chat']

    def score(self, fact, dialog_history, hypo, aspect, remove_stopwords=True):
        assert aspect in ['engagingness', 'groundedness']

        if self._dataset == 'topical_chat':
            fact = 'fact: ' + fact

        if aspect == 'groundedness':
            context = fact
        elif aspect == 'engagingness':
            context = '\n\n\n'.join([fact.strip(), dialog_history.strip()])

        return self._aligner.get_score(
            input_text=hypo,
            context=context,
            remove_stopwords=remove_stopwords)
