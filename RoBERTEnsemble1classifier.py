from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig, RobertaModel, PretrainedConfig, RobertaTokenizer, BertForSequenceClassification, BertConfig, BertModel, BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.autograd import Variable
from torch.nn import functional as F

# Here the RoBERTEnsemble class is created using the one final classifier ensemble strategy, no audio handling!

class RoBERTEnsemble(nn.Module):
    
    def __init__(self):
        super(RoBERTEnsemble, self).__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base", num_labels = 2, output_attentions = False, output_hidden_states = False)  
        
        self.bert = BertModel.from_pretrained("bert-base-uncased", num_labels = 2, output_attentions = False, output_hidden_states = False)         
        
        self.dropout = nn.Dropout(0.2)
        self.hidden_size = 768
        self.classifier = nn.Linear(self.hidden_size*2, 2) 


    def forward(self, input_ids1, input_ids2, attention_mask1,attention_mask2, labels=None, fs=None):
        outputs1 = self.roberta(input_ids1,
                            attention_mask=attention_mask1)
        outputs2 = self.bert(input_ids2,
                            attention_mask=attention_mask2) 
        
        pooled_output1 = outputs1.pooler_output
        pooled_output2 = outputs2.pooler_output
     
        
        last_hidden_state1 = outputs1.last_hidden_state
        last_hidden_state2 = outputs2.last_hidden_state
        
        concat = torch.cat((pooled_output1, pooled_output2), axis=1)

        logits = self.classifier(concat)

        outputs = logits 
           
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
            outputs = loss


        return outputs
