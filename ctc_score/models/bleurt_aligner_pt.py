from ctc_score.models.aligner import Aligner
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = 'bert-base-uncased'
MAX_LENGTH = 512

class BleurtModel(nn.Module):
    """ 
    We converted our BLEURT model from TF to PT according to the routine from 
    https://github.com/huggingface/datasets/issues/224#issuecomment-911782388, 
    and serve it with the following PT module, also from the same source. 
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = transformers.BertModel(config)
        self.dense = nn.Linear(config.hidden_size,1)
    
    def forward(self, input_ids, input_mask, segment_ids):
        cls_state = self.bert(input_ids, input_mask, 
                              segment_ids).pooler_output
        return self.dense(cls_state)


class BLEURTAligner(Aligner): 
    def __init__(self, aggr_type, checkpoint, device, *args, **kwargs):
        Aligner.__init__(self, aggr_type=None)
        
        state_dict = torch.load(checkpoint)
        config = transformers.BertConfig()
        bleurt_model = BleurtModel(config)
        bleurt_model.load_state_dict(state_dict, strict=False)
        for param in bleurt_model.parameters():
            param.requires_grad = False
        bleurt_model.eval()
        
        self.device = device
        if self.device is None: 
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.bleurt_model = bleurt_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
        
        
    def align(self, input_text, context):
        encoding = (self.tokenizer(context, 
                                   input_text, 
                                   truncation='longest_first', 
                                   max_length=MAX_LENGTH, 
                                   return_tensors='pt')
                    .to(self.device))
        input_ids, input_mask, segment_ids = (encoding['input_ids'], 
                                              encoding['attention_mask'], 
                                              encoding['token_type_ids'])
        score = self.bleurt_model(input_ids, input_mask, segment_ids).tolist()[0]
        tokens = input_text.split()
        
        return tokens, score