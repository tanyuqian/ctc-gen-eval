from nltk.corpus import stopwords


en_stopwords = stopwords.words('english')


class Aligner:
    def __init__(self, aggr_type):
        self._aggr_type = aggr_type

    def align(self, input_text, context):
        raise NotImplementedError

    def aggregate(self, tokens, token_scores, remove_stopwords):
        if self._aggr_type is None: 
            return token_scores[0]
        
        assert len(tokens) == len(token_scores)

        scores_to_aggr = []
        for token, score in zip(tokens, token_scores):
            if (token.lower() not in en_stopwords) or (not remove_stopwords):
                scores_to_aggr.append(score)

        if self._aggr_type == 'mean':
            return sum(scores_to_aggr) / len(scores_to_aggr) \
                if len(scores_to_aggr) > 0 else 0.
        elif self._aggr_type == 'sum':
            return sum(scores_to_aggr)
        else:
            raise ValueError

    def get_score(self, input_text, context, remove_stopwords):
        tokens, token_scores = self.align(
            input_text=input_text, context=context)

        if tokens is None:
            return None

        return self.aggregate(
            tokens=tokens,
            token_scores=token_scores,
            remove_stopwords=remove_stopwords)