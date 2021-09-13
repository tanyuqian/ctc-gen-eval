from ctc_score.scorer import Scorer


class SummarizationScorer(Scorer):
    def __init__(self, dataset, align, aligner_configs=None):
        Scorer.__init__(self, dataset, align, aligner_configs)

        assert self._dataset in ['cnndm', 'xsum']

    def score(self, doc, refs, hypo, aspect, remove_stopwords=False):
        kwargs = dict(
            doc=doc,
            refs=refs,
            hypo=hypo,
            remove_stopwords=remove_stopwords)

        if aspect == 'consistency':
            return self.score_consistency(**kwargs)
        elif aspect == 'relevance':
            return self.score_relevance(**kwargs)
        else:
            raise NotImplementedError

    def score_consistency(self, doc, refs, hypo, remove_stopwords):
        aligner = self._get_aligner(f'{self._align}-{self._dataset}-doc_summ')

        return aligner.get_score(
            context=doc,
            input_text=hypo,
            remove_stopwords=remove_stopwords)

    def score_relevance(self, doc, refs, hypo, remove_stopwords):
        aligner_doc_summ = self._get_aligner(
            f'{self._align}-{self._dataset}-doc_summ')
        align_y_x = aligner_doc_summ.get_score(
            context=doc,
            input_text=hypo,
            remove_stopwords=remove_stopwords)

        aligner_summ_ref = self._get_aligner(
            f'{self._align}-{self._dataset}-summ_ref')
        align_r_y = []
        for ref in refs:
            align_r_y.append(aligner_summ_ref.get_score(
                context=hypo,
                input_text=ref,
                remove_stopwords=remove_stopwords))
        align_r_y = sum(align_r_y) / len(align_r_y)

        return align_y_x * align_r_y
