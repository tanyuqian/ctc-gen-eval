from ctc_score.models.aligner import Aligner
from bleurt.score import BleurtScorer

class BLEURTAligner(Aligner, BleurtScorer): 
    def __init__(self, aggr_type, *args, **kwargs):
        Aligner.__init__(self, aggr_type=None)
        BleurtScorer.__init__(self, *args, **kwargs)
        
    def align(self, input_text, context):
        score = self.score(references=[context], candidates=[input_text])
        tokens = input_text.split()
        return tokens, score