class Aligner:
    def __init__(self, aggr_type):
        self._aggr_type = aggr_type

    def align(self, input_text, context):
        raise NotImplementedError

    def aggregate(self, tokens, token_scores):
        assert len(tokens) == len(token_scores)

        if self._aggr_type == 'mean':
            return sum(token_scores) / len(token_scores)
        elif self._aggr_type == 'sum':
            return sum(token_scores)
        else:
            raise ValueError

    def get_score(self, input_text, context):
        tokens, token_scores = self.align(
            input_text=input_text, context=context)

        if tokens is None:
            return None

        return self.aggregate(tokens=tokens, token_scores=token_scores)