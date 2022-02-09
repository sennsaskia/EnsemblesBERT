from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig, RobertaModel, PretrainedConfig, RobertaTokenizer, BertForSequenceClassification, BertConfig, BertModel, BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.autograd import Variable
from torch.nn import functional as F

# Here the RoBERTEnsemble class is created using one final classifier ensemble strategy;
# fine-tuned with an additional LSTM and attention layer;
# no audio handling!
 
class RoBERTEnsemble(nn.Module):
    
    def __init__(self):
        super(RoBERTEnsemble, self).__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base", num_labels = 2, output_attentions = False, output_hidden_states = False)  
        
        self.bert = BertModel.from_pretrained("bert-base-uncased", num_labels = 2, output_attentions = False, output_hidden_states = False)         
        
        self.dropout = nn.Dropout(0.2)
        self.hidden_size = 768

        embedding_dim = self.hidden_size
        self.bert_lstm_hidden_size = 768
        self.lstm_bert = nn.LSTM(embedding_dim, self.bert_lstm_hidden_size, 1,batch_first=True)
        
        self.classifier = nn.Linear(2000, 2) 

        #Self Attention Layer
        self.W_s1_bert = nn.Linear(self.hidden_size, 1050)
        self.W_s2_bert = nn.Linear(1050, 60)
        self.fc_layer = nn.Linear(60*self.hidden_size, 2000)

        
    def attention_net_bert(self, concat):
        attn_weight_matrix_bert = self.W_s2_bert(F.tanh(self.W_s1_bert(concat)))
        attn_weight_matrix_bert = attn_weight_matrix_bert.permute(0,2,1)
        attn_weight_matrix_bert = F.softmax(attn_weight_matrix_bert, dim=2)
        return attn_weight_matrix_bert
    
    def forward(self, input_ids1, input_ids2, attention_mask1,attention_mask2, labels=None, fs=None):
        outputs1 = self.roberta(input_ids1,
                            attention_mask=attention_mask1)
        outputs2 = self.bert(input_ids2,
                            attention_mask=attention_mask2) 
        
        last_hidden_state1 = outputs1.last_hidden_state
        last_hidden_state2 = outputs2.last_hidden_state
        
        hiddensizeconcat = torch.cat((last_hidden_state1, last_hidden_state2), axis=1)
           
        lstm_input = hiddensizeconcat
        lstm_out_bert, (ht, ct) = self.lstm_bert(torch.tensor(lstm_input))
    
        attn_weight_matrix_bert = self.attention_net_bert(lstm_out_bert)
        
        hidden_matrix_bert = torch.bmm(attn_weight_matrix_bert, lstm_out_bert)

        bert_attention_output = hidden_matrix_bert.view(-1,hidden_matrix_bert.size()[1]*hidden_matrix_bert.size()[2])
        
        logits = self.classifier(self.fc_layer(bert_attention_output))

        outputs = logits 
           
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
            outputs = loss


        return outputs
