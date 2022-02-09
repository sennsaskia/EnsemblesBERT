from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig, RobertaModel, PretrainedConfig, RobertaTokenizer, BertForSequenceClassification, BertConfig, BertModel, BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from models.BERTBase_RoBERTa_Ensemble import RoBERTaEns
from models.BERTBase_Ensemble import BERTBaseEnsemble

from AudiBERTutils import pad_sequences

# Here the RoBERTa class is created, no audio handling!

class RoBERTEnsemble(nn.Module):
    
    def __init__(self):
        super(RoBERTEnsemble, self).__init__()

        self.roberta = RobertaForSequenceClassification.from_pretrained("roberta-base")  
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")         
        
        self.dropout = nn.Dropout(0.2)
        self.hidden_size = 128
        self.classifier = nn.Linear(self.hidden_size*2, 2)

    
    
    def forward(self, input_ids1, input_ids2, attention_mask1,attention_mask2, labels=None, fs=None):
        outputs1 = self.roberta(input_ids1,
                            attention_mask=attention_mask1, labels=labels)
        outputs2 = self.bert(input_ids2,
                            attention_mask=attention_mask2, labels=labels)
        
        logit1 = outputs1.logits
        logit2 = outputs2.logits

        outputs = (logit1 + logit2) / 2

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
            outputs = loss

        return outputs