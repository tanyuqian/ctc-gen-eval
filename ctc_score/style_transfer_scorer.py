from ctc_score.scorer import Scorer


class StyleTransferScorer(Scorer):
    def __init__(self, align, aggr_type='mean', device='cuda'):
        Scorer.__init__(self, align=align, aggr_type=aggr_type, device=device)

    def score(self, input_sent, hypo, aspect, 
              remove_stopwords=False, rescale_with_baseline=False):
        # rescale_with_baseline does the same thing as its namesake in BERTScore
        # and it is only applied for the E aligner
        kwargs = dict(
            input_sent=input_sent,
            hypo=hypo,
            remove_stopwords=remove_stopwords,
            rescale_with_baseline=rescale_with_baseline)

        if aspect == 'preservation':
            return self.score_preservation(**kwargs)
        else:
            raise NotImplementedError

    def score_preservation(self, input_sent, hypo, 
                           remove_stopwords, rescale_with_baseline):
        aligner = self._get_aligner('sent_to_sent')

        align_y_x = aligner.get_score(
            input_text=input_sent,
            context=hypo,
            remove_stopwords=remove_stopwords)

        align_x_y = aligner.get_score(
            input_text=hypo,
            context=input_sent,
            remove_stopwords=remove_stopwords)
        
        score = 2 * (align_y_x * align_x_y) / (align_y_x + align_x_y)
        
        if self._align.startswith('E') and rescale_with_baseline: 
            baseline = aligner.baseline_vals[2].item()
            return (score - baseline) / (1 - baseline)

        return score

