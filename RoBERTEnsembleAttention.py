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
        
        self.roberta = RobertaModel.from_pretrained("roberta-base", num_labels = 2, output_attentions = False, output_hidden_states = False)  
        
        self.bert = BertModel.from_pretrained("bert-base-uncased", num_labels = 2, output_attentions = False, output_hidden_states = False)   
        
        self.dropout = nn.Dropout(0.2)
        self.hidden_size = 768

        self.classifier = nn.Linear(2000, 2) 
        
        #Self Attention Layer
        self.W_s1_bert = nn.Linear(self.hidden_size, 1050)#350
        self.W_s2_bert = nn.Linear(1050, 60)#350,60
        self.fc_layer = nn.Linear(60*self.hidden_size, 2000) #60*,2000 self.hidden_size

        
    def attention_net_bert(self, last_hidden):
        attn_weight_matrix_bert = self.W_s2_bert(F.tanh(self.W_s1_bert(last_hidden)))
        attn_weight_matrix_bert = attn_weight_matrix_bert.permute(0,2,1) 
        attn_weight_matrix_bert = F.softmax(attn_weight_matrix_bert, dim=2)
        return attn_weight_matrix_bert

    def forward(self, input_ids1, input_ids2, attention_mask1,attention_mask2, labels=None, fs=None):
        outputs1 = self.roberta(input_ids1,
                            attention_mask=attention_mask1)
        outputs2 = self.bert(input_ids2,
                            attention_mask=attention_mask2)

       
        last_hidden1 = outputs1.last_hidden_state
        last_hidden2 = outputs2.last_hidden_state

        attn_weight_matrix_bert1 = self.attention_net_bert(last_hidden1)
        hidden_matrix_bert1 = torch.bmm(attn_weight_matrix_bert1, last_hidden1)
        bert_attention_output1 = hidden_matrix_bert1.view(-1,hidden_matrix_bert1.size()[1]*hidden_matrix_bert1.size()[2])
        
        attn_weight_matrix_bert2 = self.attention_net_bert(last_hidden2)
        hidden_matrix_bert2 = torch.bmm(attn_weight_matrix_bert2, last_hidden2)
        bert_attention_output2= hidden_matrix_bert2.view(-1,hidden_matrix_bert2.size()[1]*hidden_matrix_bert2.size()[2])

        logits1 = self.classifier(self.fc_layer(bert_attention_output1))
        logits2 = self.classifier(self.fc_layer(bert_attention_output2))
        
        outputs = (logits1 + logits2)/2
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
            outputs = loss

        return outputs