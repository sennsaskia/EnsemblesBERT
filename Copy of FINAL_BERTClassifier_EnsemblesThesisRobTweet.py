# Author Saskia Jan 2022 for BERT, RoBERTa, DistilBERT and BERTweet, 
# Ensembles with averaging 2 classifier or using 1 classifier,
# with or without attention layers (and LSTM) on top!
# 2020 Audio Enhanced BERT - AudiBERT

from AudiBERTutils import recordRun, pad_sequences, getIdentifier, str2bool, runCount 
import traceback
import sys
import argparse
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


# Default Configuration
MAX_LEN = 128
batch_size = 8#4
epochs = 10#20
learning_rate = 2e-5
question ="doing_today" #"advice_yourself"
logtodb = True #False
maxruns = 100 #20 
modelName = 'RobTweetEnsemble'
#'DistilTweetEnsemble'
#'RoBERTaAttentionLSTM'
#'SpecialEnsembleAttention'#'SpecialEnsembleAttentionLSTM'
#'DistilBERTaEnsembleLSTM'#'RoBERTaLSTM'#'AllinEnsembleAttentionLSTM01'
#'AllinEnsembleAttentionLSTM'#'AllinEnsembleLSTM'

#'RoBERTa'#'RoBERTaAttention'#'RoBERTaAttentionLSTM'
#'BERT'#'BERTAttention'#'BERTAttentionLSTM'
#'MultiRoBERTaAttentionLSTM'#'Multi3RoBERTa'#'Multi3RoBERTaAttentionLSTM'
#'Multi6BERTAttentionLSTM'#'Multi3BERT'#'Multi3BERTAttentionLSTM'
#'AllinEnsemble'#'AllinEnsembleAttention'#'AllinEnsembleAttentionLSTM'
#"BERTweet"#'BERTweetAttention'#"BERTweetAttentionLSTM"
#'DistilBERTaEnsembleAttentionLSTM'#'DistilBERTaEnsembleAttention'#'DistilBERTaEnsemble'
#'DistilBERTAttentionLSTM'#'DistilBERTAttention'#'DistilBERT'
#'Multi6DistilBERTAttentionLSTM'#'Multi3DistilBERTAttentionLSTM'#'MultiDistilBERTAttentionLSTM'
#'RoBERTEnsemble1ClassifierAttentionLSTM'#'RoBERTEnsemble1ClassifierAttention'#'RoBERTEnsemble1Classifier' 
#'RoBERTEnsemble'#'RoBERTEnsembleAttention'#'RoBERTEnsembleAttentionLSTM' 
#'TriangleEnsemble'#'TriangleEnsembleAttention'#'TriangleEnsembleAttentionLSTM'
#'AllinEnsembleMajor'#
save_model = False
aggregatedQuestionSplits = 'aggregatedQuestionSplits/'
aggregatedQuestionSplits = 'aggregatedQuestionSplits_padded/'   
useLSTM = False
useLastVectors = False

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, help="Number of Epochs to train. Default=10")
parser.add_argument("--question", help="Daic Question Abreviation (Folder Name). Default=doing_today")
parser.add_argument("--modelName", help="Model Name. Default=BERT")
parser.add_argument("--logtodb", type=str2bool, help="Log Run To DB. Default=True")
parser.add_argument("--useLSTM", type=str2bool, help="Use LSTM. Default=False")
parser.add_argument("--maxruns", type=int, help="Maximum number of runs for same configuraiton Default=20")
parser.add_argument("--batch_size", type=int, help="Batch size for same configuraiton Default=4")
parser.add_argument("--lr", type=float, help="Use folloup question. Default=2e-5")
parser.add_argument("--maxlen", type=int, help="Maximum number of tokens, default = 128")

args = parser.parse_args()

if args.__dict__["epochs"]  is not None:
    epochs = args.__dict__["epochs"]
if args.__dict__["question"]  is not None:
    question = args.__dict__["question"]
if args.__dict__["modelName"]  is not None:
    modelName = args.__dict__["modelName"] 
if args.__dict__["logtodb"]  is not None:
    logtodb = args.__dict__["logtodb"] 
if args.__dict__["useLSTM"]  is not None:
    useLSTM = args.__dict__["useLSTM"]     
if args.__dict__["maxruns"]  is not None:
    maxruns = args.__dict__["maxruns"]  
if args.__dict__["batch_size"]  is not None:
    batch_size = args.__dict__["batch_size"] 
if args.__dict__["lr"]  is not None:
    learning_rate = args.__dict__["lr"]
if args.__dict__["maxlen"]  is not None:
    MAX_LEN = args.__dict__["maxlen"]

    
db='./expResults/RobTweet1.db'

typeOfTask = modelName


configuration = {
    'MAX_LEN':MAX_LEN,
    'batch_size':batch_size,
    'model': modelName,
    'epochs':epochs,
    'question':question,
    'typeOfTask':typeOfTask,
    'aggregatedQuestionSplits': aggregatedQuestionSplits,
    'useLSTM': useLSTM,
    'useLastVectors': useLastVectors,
    'lr' : learning_rate
}



short_configuration = ''
for configValue in configuration:
    short_configuration = short_configuration + str(configuration[configValue])

print(short_configuration)
results = {}

results['run_id'] = runCount(short_configuration,db=db)

if results['run_id'] > maxruns:
    print("Exiting. There are too many experiments with this configuraiton. Increase Max Run Count")
    sys.exit(0)
print(results['run_id'])

traindatafile = "./data/"+question+"/train.tsv"
testdatafile  = "./data/"+question+"/dev.tsv"

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[4]:


#Data Exploration
import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv(traindatafile, delimiter='\t', header=None, names=['index', 'label', 'label_notes', 'sentence', 'audio_features'])

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))


# Get the lists of sentences, audio features and their labels.
sentences = df.sentence.values
labels = df.label.values
audio = df.audio_features.values 


# Generate Data structures for Audio Features and Train Identifiers. 

train_identifiers = []

# Audio Feature Exploration
for feature_string in audio:
    identifier = getIdentifier(feature_string)
    train_identifiers.append(identifier)

configuration["train_identifiers"] = train_identifiers

# In[8]:


#### Prepare Training Data Set ####

if 'ensemble' in modelName.lower():
    if 'robtweet' in modelName.lower():
      # Load the BERT tokenizer.

      from transformers import DistilBertTokenizer, AutoTokenizer
      # Load DistilBERT tokenizer eventhough it is the same as BERTtokenizer
      tokenizer1 = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
      print("BERTweet = Tokenizer1")

      from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
      from transformers import RobertaTokenizer
      tokenizer2 = RobertaTokenizer.from_pretrained('roberta-base')
      print("RobertaTokenizer = Tokenizer2")
      
      
              # pre-processing of the input

      #bertweet
      input_ids1 = []
      line1 = 0

      #roberta
      input_ids2 = []
      line2 = 0

      
      # For every sentence...
      for sentence in sentences:
          # `encode` will:
          #   (1) Tokenize the sentence.
          #   (2) Prepend the `[CLS]` token to the start.
          #   (3) Append the `[SEP]` token to the end.
          #   (4) Map tokens to their IDs.
          #bertweet:
          encoded_sent1 = tokenizer1.encode(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True # Add '[CLS]' and '[SEP]'
                          )
          # Add the encoded sentence to the list for roberta
          input_ids1.append(encoded_sent1)

          line1 = line1 + 1
          #distilbert
          encoded_sent2 = tokenizer2.encode(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True # Add '[CLS]' and '[SEP]'
                          )
          # Add the encoded sentence to the list for bert
          input_ids2.append(encoded_sent2)
          #print(encoded_sent)
          line2 = line2 + 1
          

      ## padding
      #bertweet
      # Pad our input tokens with value 0.
      # "post" indicates that we want to pad and truncate at the end of the sequence,
      # as opposed to the beginning.
      input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, dtype="long", 
                                value=0, truncating="post", padding="post")

      #roberta
      input_ids2 = pad_sequences(input_ids2, maxlen=MAX_LEN, dtype="long", 
                                value=0, truncating="post", padding="post")   
        

      ## attention masks ## should it be changed for roberta?!
      #bertweet
      attention_masks1 = []

      # For each sentence...
      for sent in input_ids1:

          # Create the attention mask.
          #   - If a token ID is 0, then it's padding, set the mask to 0.
          #   - If a token ID is > 0, then it's a real token, set the mask to 1.
          att_mask1 = [int(token_id > 0) for token_id in sent]

          # Store the attention mask for this sentence.
          attention_masks1.append(att_mask1)

      #roberta
      attention_masks2 = []

      # For each sentence...
      for sent in input_ids2:

          # Create the attention mask.
          #   - If a token ID is 0, then it's padding, set the mask to 0.
          #   - If a token ID is > 0, then it's a real token, set the mask to 1.
          att_mask2 = [int(token_id > 0) for token_id in sent]

          # Store the attention mask for this sentence.
          attention_masks2.append(att_mask2)
      
          
          
      if 'attention' in modelName.lower():
              if 'lstm' in modelName.lower():
                  modelName = 'RobtweetensembledAttentionLSTM'
              else:
                  modelName = 'RobtweetensembledAttention'
      else:
          if 'lstm' in modelName.lower():
              modelName = 'RobtweetensembledLSTM'
          else:
              modelName = 'Robtweetensembled'

    elif 'distiltweet' in modelName.lower():
      # Load the BERT tokenizer.

      from transformers import DistilBertTokenizer, AutoTokenizer
      # Load DistilBERT tokenizer eventhough it is the same as BERTtokenizer
      tokenizer1 = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
      print("BERTweet = Tokenizer1")

      tokenizer2 = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
      print("DistilBERTTokenizer = Tokenizer2")
      
      
              # pre-processing of the input

      #bertweet
      input_ids1 = []
      line1 = 0

      #distilbert
      input_ids2 = []
      line2 = 0

      
      # For every sentence...
      for sentence in sentences:
          # `encode` will:
          #   (1) Tokenize the sentence.
          #   (2) Prepend the `[CLS]` token to the start.
          #   (3) Append the `[SEP]` token to the end.
          #   (4) Map tokens to their IDs.
          #bertweet:
          encoded_sent1 = tokenizer1.encode(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True # Add '[CLS]' and '[SEP]'
                          )
          # Add the encoded sentence to the list for roberta
          input_ids1.append(encoded_sent1)

          line1 = line1 + 1
          #distilbert
          encoded_sent2 = tokenizer2.encode(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True # Add '[CLS]' and '[SEP]'
                          )
          # Add the encoded sentence to the list for bert
          input_ids2.append(encoded_sent2)
          #print(encoded_sent)
          line2 = line2 + 1
          

      ## padding
      #bertweet
      # Pad our input tokens with value 0.
      # "post" indicates that we want to pad and truncate at the end of the sequence,
      # as opposed to the beginning.
      input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, dtype="long", 
                                value=0, truncating="post", padding="post")

      #distilbert
      input_ids2 = pad_sequences(input_ids2, maxlen=MAX_LEN, dtype="long", 
                                value=0, truncating="post", padding="post")   
        

      ## attention masks ## should it be changed for roberta?!
      #bertweet
      attention_masks1 = []

      # For each sentence...
      for sent in input_ids1:

          # Create the attention mask.
          #   - If a token ID is 0, then it's padding, set the mask to 0.
          #   - If a token ID is > 0, then it's a real token, set the mask to 1.
          att_mask1 = [int(token_id > 0) for token_id in sent]

          # Store the attention mask for this sentence.
          attention_masks1.append(att_mask1)

      #distilbert
      attention_masks2 = []

      # For each sentence...
      for sent in input_ids2:

          # Create the attention mask.
          #   - If a token ID is 0, then it's padding, set the mask to 0.
          #   - If a token ID is > 0, then it's a real token, set the mask to 1.
          att_mask2 = [int(token_id > 0) for token_id in sent]

          # Store the attention mask for this sentence.
          attention_masks2.append(att_mask2)
      
          
          
      if 'attention' in modelName.lower():
              if 'lstm' in modelName.lower():
                  modelName = 'DistiltweetensembledAttentionLSTM'
              else:
                  modelName = 'DistiltweetensembledAttention'
      else:
          if 'lstm' in modelName.lower():
              modelName = 'DistiltweetensembledLSTM'
          else:
              modelName = 'Distiltweetensembled'

    elif 'special' in modelName.lower():
      from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
      from transformers import RobertaTokenizer
      tokenizer1 = RobertaTokenizer.from_pretrained('roberta-base')
      print("RobertaTokenizer = Tokenizer1")

      # Load the BERT tokenizer.

      from transformers import DistilBertTokenizer, AutoTokenizer
      # Load DistilBERT tokenizer eventhough it is the same as BERTtokenizer
      tokenizer2 = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
      print("DistilBERTTokenizer = Tokenizer2")
      
      tokenizer3 = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
      print("BERTweet = Tokenizer3")
      
              # pre-processing of the input

      #roberta
      input_ids1 = []
      line1 = 0

      #distilbert
      input_ids2 = []
      line2 = 0
      
      #bertweet
      input_ids3 = []
      line3 = 0
      
      # For every sentence...
      for sentence in sentences:
          # `encode` will:
          #   (1) Tokenize the sentence.
          #   (2) Prepend the `[CLS]` token to the start.
          #   (3) Append the `[SEP]` token to the end.
          #   (4) Map tokens to their IDs.
          #roberta:
          encoded_sent1 = tokenizer1.encode(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True # Add '[CLS]' and '[SEP]'
                          )
          # Add the encoded sentence to the list for roberta
          input_ids1.append(encoded_sent1)

          line1 = line1 + 1
          #distilbert
          encoded_sent2 = tokenizer2.encode(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True # Add '[CLS]' and '[SEP]'
                          )
          # Add the encoded sentence to the list for bert
          input_ids2.append(encoded_sent2)
          #print(encoded_sent)
          line2 = line2 + 1
          
          #bertweet
          encoded_sent3 = tokenizer3.encode(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True # Add '[CLS]' and '[SEP]'
                          )
          # Add the encoded sentence to the list for bert
          input_ids3.append(encoded_sent3)
          #print(encoded_sent)
          line3 = line3 + 1

      ## padding
      #roberta
      # Pad our input tokens with value 0.
      # "post" indicates that we want to pad and truncate at the end of the sequence,
      # as opposed to the beginning.
      input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, dtype="long", 
                                value=0, truncating="post", padding="post")

      #distilbert
      input_ids2 = pad_sequences(input_ids2, maxlen=MAX_LEN, dtype="long", 
                                value=0, truncating="post", padding="post")   
      
      #bertweet
      input_ids3 = pad_sequences(input_ids3, maxlen=MAX_LEN, dtype="long", 
                                value=0, truncating="post", padding="post")  

      ## attention masks ## should it be changed for roberta?!
      #roberta
      attention_masks1 = []

      # For each sentence...
      for sent in input_ids1:

          # Create the attention mask.
          #   - If a token ID is 0, then it's padding, set the mask to 0.
          #   - If a token ID is > 0, then it's a real token, set the mask to 1.
          att_mask1 = [int(token_id > 0) for token_id in sent]

          # Store the attention mask for this sentence.
          attention_masks1.append(att_mask1)

      #distilbert
      attention_masks2 = []

      # For each sentence...
      for sent in input_ids2:

          # Create the attention mask.
          #   - If a token ID is 0, then it's padding, set the mask to 0.
          #   - If a token ID is > 0, then it's a real token, set the mask to 1.
          att_mask2 = [int(token_id > 0) for token_id in sent]

          # Store the attention mask for this sentence.
          attention_masks2.append(att_mask2)
      
      #bertweet
      attention_masks3 = []

      # For each sentence...
      for sent in input_ids3:

          # Create the attention mask.
          #   - If a token ID is 0, then it's padding, set the mask to 0.
          #   - If a token ID is > 0, then it's a real token, set the mask to 1.
          att_mask3 = [int(token_id > 0) for token_id in sent]

          # Store the attention mask for this sentence.
          attention_masks3.append(att_mask3)
          
          
      if 'attention' in modelName.lower():
              if 'lstm' in modelName.lower():
                  modelName = 'SpecialensembledAttentionLSTM'
              else:
                  modelName = 'SpecialensembledAttention'
      else:
          if 'lstm' in modelName.lower():
              modelName = 'SpecialensembledLSTM'
          if 'major' in modelName.lower():
              modelName = 'SpecialMajorensembled'
          else:
              modelName = 'Specialensembled'
    #call model ensemble
    elif 'allin' in modelName.lower():
      from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
      from transformers import RobertaTokenizer
      tokenizer1 = RobertaTokenizer.from_pretrained('roberta-base')
      print("RobertaTokenizer = Tokenizer1")

      # Load the BERT tokenizer.

      from transformers import DistilBertTokenizer, BertTokenizer, AutoTokenizer
      # Load DistilBERT tokenizer eventhough it is the same as BERTtokenizer
      tokenizer2 = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
      print("DistilBERTTokenizer = Tokenizer2")

      tokenizer3 = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
      print("BERTTokenizer = Tokenizer3")

      tokenizer4 = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
      print("BERTweet = Tokenizer4")

              # pre-processing of the input

      #roberta
      input_ids1 = []
      line1 = 0

      #distilbert
      input_ids2 = []
      line2 = 0

      #bert
      input_ids3 = []
      line3 = 0

      #bertweet
      input_ids4 = []
      line4 = 0

      # For every sentence...
      for sentence in sentences:
          # `encode` will:
          #   (1) Tokenize the sentence.
          #   (2) Prepend the `[CLS]` token to the start.
          #   (3) Append the `[SEP]` token to the end.
          #   (4) Map tokens to their IDs.
          #roberta:
          encoded_sent1 = tokenizer1.encode(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True # Add '[CLS]' and '[SEP]'
                          )
          # Add the encoded sentence to the list for roberta
          input_ids1.append(encoded_sent1)

          line1 = line1 + 1
          #distilbert
          encoded_sent2 = tokenizer2.encode(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True # Add '[CLS]' and '[SEP]'
                          )
          # Add the encoded sentence to the list for bert
          input_ids2.append(encoded_sent2)
          #print(encoded_sent)
          line2 = line2 + 1
          
          #bert
          encoded_sent3 = tokenizer3.encode(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True # Add '[CLS]' and '[SEP]'
                          )
          # Add the encoded sentence to the list for bert
          input_ids3.append(encoded_sent3)
          #print(encoded_sent)
          line3 = line3 + 1
          
          #bertweet
          encoded_sent4 = tokenizer4.encode(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True # Add '[CLS]' and '[SEP]'
                          )
          # Add the encoded sentence to the list for bert
          input_ids4.append(encoded_sent4)
          #print(encoded_sent)
          line4 = line4 + 1

      ## padding
      #roberta
      # Pad our input tokens with value 0.
      # "post" indicates that we want to pad and truncate at the end of the sequence,
      # as opposed to the beginning.
      input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, dtype="long", 
                                value=0, truncating="post", padding="post")

      #distilbert
      input_ids2 = pad_sequences(input_ids2, maxlen=MAX_LEN, dtype="long", 
                                value=0, truncating="post", padding="post")   

      #bert
      input_ids3 = pad_sequences(input_ids3, maxlen=MAX_LEN, dtype="long", 
                                value=0, truncating="post", padding="post")  

      #bertweet
      input_ids4 = pad_sequences(input_ids4, maxlen=MAX_LEN, dtype="long", 
                                value=0, truncating="post", padding="post")  

      ## attention masks ## should it be changed for roberta?!
      #roberta
      attention_masks1 = []

      # For each sentence...
      for sent in input_ids1:

          # Create the attention mask.
          #   - If a token ID is 0, then it's padding, set the mask to 0.
          #   - If a token ID is > 0, then it's a real token, set the mask to 1.
          att_mask1 = [int(token_id > 0) for token_id in sent]

          # Store the attention mask for this sentence.
          attention_masks1.append(att_mask1)

      #distilbert
      attention_masks2 = []

      # For each sentence...
      for sent in input_ids2:

          # Create the attention mask.
          #   - If a token ID is 0, then it's padding, set the mask to 0.
          #   - If a token ID is > 0, then it's a real token, set the mask to 1.
          att_mask2 = [int(token_id > 0) for token_id in sent]

          # Store the attention mask for this sentence.
          attention_masks2.append(att_mask2)

      #bert
      attention_masks3 = []

      # For each sentence...
      for sent in input_ids3:

          # Create the attention mask.
          #   - If a token ID is 0, then it's padding, set the mask to 0.
          #   - If a token ID is > 0, then it's a real token, set the mask to 1.
          att_mask3 = [int(token_id > 0) for token_id in sent]

          # Store the attention mask for this sentence.
          attention_masks3.append(att_mask3)
          
      #bertweet
      attention_masks4 = []

      # For each sentence...
      for sent in input_ids4:

          # Create the attention mask.
          #   - If a token ID is 0, then it's padding, set the mask to 0.
          #   - If a token ID is > 0, then it's a real token, set the mask to 1.
          att_mask4 = [int(token_id > 0) for token_id in sent]

          # Store the attention mask for this sentence.
          attention_masks4.append(att_mask4)
          
      if 'attention' in modelName.lower():
              if 'lstm' in modelName.lower():
                  modelName = 'AllinensembledAttentionLSTM'
              else:
                  modelName = 'AllinensembledAttention'
      else:
          if 'lstm' in modelName.lower():
              if '01' in modelName.lower():
                  modelName = 'AllinensembledLSTM01'
              elif '02' in modelName.lower():
                  modelName = 'AllinensembledLSTM02'
              elif '03' in modelName.lower():
                  modelName = 'AllinensembledLSTM03'
              else:
                  modelName = 'AllinensembledLSTM'
          if 'major' in modelName.lower():
              modelName = 'AllinMajorensembled'
          else:
              modelName = 'Allinensembled'
                    
    elif 'triangle' in modelName.lower():
        from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
        from transformers import RobertaTokenizer
        tokenizer1 = RobertaTokenizer.from_pretrained('roberta-base')
        print("RobertaTokenizer = Tokenizer1")

        # Load the BERT tokenizer.

        from transformers import DistilBertTokenizer, BertTokenizer
        # Load DistilBERT tokenizer eventhough it is the same as BERTtokenizer
        tokenizer2 = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print("DistilBERTTokenizer = Tokenizer2")
        
        tokenizer3 = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        print("BERTTokenizer = Tokenizer3")
        
               # pre-processing of the input

        #roberta
        input_ids1 = []
        line1 = 0

        #distilbert
        input_ids2 = []
        line2 = 0
        
        #bert
        input_ids3 = []
        line3 = 0
        
        # For every sentence...
        for sentence in sentences:
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #roberta:
            encoded_sent1 = tokenizer1.encode(
                                sentence,                      # Sentence to encode.
                                add_special_tokens = True # Add '[CLS]' and '[SEP]'
                           )
            # Add the encoded sentence to the list for roberta
            input_ids1.append(encoded_sent1)

            line1 = line1 + 1
            #distilbert
            encoded_sent2 = tokenizer2.encode(
                                sentence,                      # Sentence to encode.
                                add_special_tokens = True # Add '[CLS]' and '[SEP]'
                           )
            # Add the encoded sentence to the list for bert
            input_ids2.append(encoded_sent2)
            #print(encoded_sent)
            line2 = line2 + 1
            
            #bert
            encoded_sent3 = tokenizer3.encode(
                                sentence,                      # Sentence to encode.
                                add_special_tokens = True # Add '[CLS]' and '[SEP]'
                           )
            # Add the encoded sentence to the list for bert
            input_ids3.append(encoded_sent3)
            #print(encoded_sent)
            line3 = line3 + 1

        ## padding
        #roberta
        # Pad our input tokens with value 0.
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, dtype="long", 
                                  value=0, truncating="post", padding="post")

        #distilbert
        input_ids2 = pad_sequences(input_ids2, maxlen=MAX_LEN, dtype="long", 
                                  value=0, truncating="post", padding="post")   
        
        #bert
        input_ids3 = pad_sequences(input_ids3, maxlen=MAX_LEN, dtype="long", 
                                  value=0, truncating="post", padding="post")  

        ## attention masks ## should it be changed for roberta?!
        #roberta
        attention_masks1 = []

        # For each sentence...
        for sent in input_ids1:

            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask1 = [int(token_id > 0) for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks1.append(att_mask1)

        #distilbert
        attention_masks2 = []

        # For each sentence...
        for sent in input_ids2:

            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask2 = [int(token_id > 0) for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks2.append(att_mask2)
        
        #bert
        attention_masks3 = []

        # For each sentence...
        for sent in input_ids3:

            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask3 = [int(token_id > 0) for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks3.append(att_mask3) 
            
        if 'attention' in modelName.lower():
                if 'lstm' in modelName.lower():
                    modelName = 'TriangleensembledAttentionLSTM'
                else:
                    modelName = 'TriangleensembledAttention'
        else:
            if 'lstm' in modelName.lower():
                    modelName = 'TriangleensembledLSTM'
            else:
                    modelName = 'Triangleensembled'

    else:
        if 'robert' in modelName.lower():
            from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
            from transformers import BertTokenizer, RobertaTokenizer
            tokenizer1 = RobertaTokenizer.from_pretrained('roberta-base')
            print("RobertaTokenizer = Tokenizer1")

            # Load the BERT tokenizer.

            tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            print("BERTTokenizer = Tokenizer2")

        elif 'distilberta' in modelName.lower():
            from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
            from transformers import RobertaTokenizer
            tokenizer1 = RobertaTokenizer.from_pretrained('roberta-base')
            print("RobertaTokenizer = Tokenizer1")

            # Load the BERT tokenizer.

            from transformers import DistilBertTokenizer
            # Load DistilBERT tokenizer eventhough it is the same as BERTtokenizer
            tokenizer2 = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            print("DistilBERTTokenizer = Tokenizer2")


        # pre-processing of the input

        #roberta
        input_ids1 = []
        line1 = 0

        #bert
        input_ids2 = []
        line2 = 0

        # For every sentence...
        for sentence in sentences:
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #roberta:
            encoded_sent1 = tokenizer1.encode(
                                sentence,                      # Sentence to encode.
                                add_special_tokens = True # Add '[CLS]' and '[SEP]'
                           )
            # Add the encoded sentence to the list for roberta
            input_ids1.append(encoded_sent1)

            line1 = line1 + 1
            #bert
            encoded_sent2 = tokenizer2.encode(
                                sentence,                      # Sentence to encode.
                                add_special_tokens = True # Add '[CLS]' and '[SEP]'
                           )
            # Add the encoded sentence to the list for bert
            input_ids2.append(encoded_sent2)
            #print(encoded_sent)
            line2 = line2 + 1

        ## padding
        #roberta
        # Pad our input tokens with value 0.
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, dtype="long", 
                                  value=0, truncating="post", padding="post")

        #bert
        input_ids2 = pad_sequences(input_ids2, maxlen=MAX_LEN, dtype="long", 
                                  value=0, truncating="post", padding="post")   

        ## attention masks ## should it be changed for roberta?!
        #roberta
        attention_masks1 = []

        # For each sentence...
        for sent in input_ids1:

            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask1 = [int(token_id > 0) for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks1.append(att_mask1)

        #bert
        attention_masks2 = []

        # For each sentence...
        for sent in input_ids2:

            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask2 = [int(token_id > 0) for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks2.append(att_mask2)
            #############
        if 'distilberta' in modelName.lower():
            if 'attention' in modelName.lower():
                if 'lstm' in modelName.lower():
                    modelName = 'DistilaensembledAttentionLSTM'
                else:
                    modelName = 'DistilaensembledAttention'
            else:
                if 'lstm' in modelName.lower():
                    modelName = 'DistilaensembledLSTM'
                else:
                    modelName = 'Distilaensembled'

        elif '1' in modelName:
            if 'attention' in modelName.lower():
                if 'lstm' in modelName.lower():
                    modelName = 'ensembled1AttentionLSTM'
                else:
                    modelName = 'ensembled1Attention'
            else:
                modelName = 'ensembled1'
        else: 
            if 'attention' in modelName.lower():
                if 'lstm' in modelName.lower():
                    modelName = 'ensembledAttentionLSTM'
                else:
                    modelName = 'ensembledAttention'
            else:
                modelName = 'ensembled' # so it will not run through all the if 'BERT' in stuff!
    print(modelName)

    
elif 'BERT' in modelName.upper():
    if 'roberta' in modelName.lower():
        from transformers import RobertaTokenizer
        # Load the RoBERTa tokenizer.
        print('Loading RoBERTa tokenizer...')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
    elif 'bertweet' in modelName.lower():
        from transformers import AutoTokenizer
        # Load the BERTweet tokenizer.
        print('Loading BERTweet tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    elif 'distilbert' in modelName.lower():
        from transformers import DistilBertTokenizer
        # Load DistilBERT tokenizer eventhough it is the same as BERTtokenizer
        print('Loading DistilBERT tokenizer...')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        from transformers import BertTokenizer
        # Load the BERT tokenizer.
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# In[9]:


if 'BERT' in modelName.upper():
    print('Preparing for simple (modified) BERT models!')
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    line = 0
    
    # For every sentence...
    for sentence in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sentence,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
        #print(encoded_sent)
        line = line + 1
    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])


if 'BERT' in modelName.upper():
    print('Padding ongoing')
    # We'll borrow the `pad_sequences` utility function to do this.
    from AudiBERTutils import pad_sequences

    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")

    print('\nDone.')


# In[13]:


if 'BERT' in modelName.upper():
    #The attention mask simply makes it explicit which tokens are actual words versus which are padding.
    #The BERT vocabulary does not use the ID 0, so if a token ID is 0, then it's padding, and otherwise it's a real token.

    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)


# In[11]:


# Convert all inputs and labels into torch tensors, the required datatype 
# for our model.
if 'robtweet' in modelName.lower():
    print('Preparing Training Input for RobTweet ensembles!')
    #bertweet
    train_inputs1 = torch.tensor(input_ids1)
    train_masks1 = torch.tensor(attention_masks1)
    
    #roberta
    train_inputs2 = torch.tensor(input_ids2)
    train_masks2 = torch.tensor(attention_masks2)
    
elif 'distiltweet' in modelName.lower():
    print('Preparing Training Input for DistilBERTweet ensembles!')
    #bertweet
    train_inputs1 = torch.tensor(input_ids1)
    train_masks1 = torch.tensor(attention_masks1)
    
    #distilbert
    train_inputs2 = torch.tensor(input_ids2)
    train_masks2 = torch.tensor(attention_masks2)
    
elif 'allin' in modelName.lower():
    print('Preparing Training Input for all in ensembles!')
    #roberta
    train_inputs1 = torch.tensor(input_ids1)
    train_masks1 = torch.tensor(attention_masks1)
    
    #distilbert
    train_inputs2 = torch.tensor(input_ids2)
    train_masks2 = torch.tensor(attention_masks2)
    
    #bert
    train_inputs3 = torch.tensor(input_ids3)
    train_masks3 = torch.tensor(attention_masks3)
    
    #bertweet
    train_inputs4 = torch.tensor(input_ids4)
    train_masks4 = torch.tensor(attention_masks4)
    
elif 'triangle' in modelName.lower():
    print('Preparing Training Input for triangle ensembles!')
    #roberta
    train_inputs1 = torch.tensor(input_ids1)
    train_masks1 = torch.tensor(attention_masks1)
    
    #distilbert
    train_inputs2 = torch.tensor(input_ids2)
    train_masks2 = torch.tensor(attention_masks2)
    
    #bert
    train_inputs3 = torch.tensor(input_ids3)
    train_masks3 = torch.tensor(attention_masks3)
    
elif 'special' in modelName.lower():
    print('Preparing Training Input for special ensembles!')
    #roberta
    train_inputs1 = torch.tensor(input_ids1)
    train_masks1 = torch.tensor(attention_masks1)
    
    #distilbert
    train_inputs2 = torch.tensor(input_ids2)
    train_masks2 = torch.tensor(attention_masks2)
    
    #bertweet
    train_inputs3 = torch.tensor(input_ids3)
    train_masks3 = torch.tensor(attention_masks3)
    
elif 'ensembled' in modelName.lower():
    print('Preparing Training Input for ensembles!')
    #roberta
    train_inputs1 = torch.tensor(input_ids1)
    train_masks1 = torch.tensor(attention_masks1)
    
    #bert
    train_inputs2 = torch.tensor(input_ids2)
    train_masks2 = torch.tensor(attention_masks2)
    
elif 'BERT' in modelName.upper():
    print('Preparing Training Input for simple models!')

    train_inputs = torch.tensor(input_ids)
    train_masks = torch.tensor(attention_masks)

train_labels = torch.tensor(labels) # All
train_identifiers = torch.tensor([int(id) for id in train_identifiers]) #All



# In[12]:


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.


# Create the DataLoader for our training set.

if 'allin' in modelName.lower():
    train_data = TensorDataset(train_inputs1, train_inputs2, train_inputs3, train_inputs4, 
                               train_masks1, train_masks2, train_masks3, train_masks4,
                               train_labels,train_identifiers)
    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
elif 'triangle' in modelName.lower():
    train_data = TensorDataset(train_inputs1, train_inputs2, train_inputs3, train_masks1, train_masks2, train_masks3,
                               train_labels,train_identifiers)
    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
elif 'special' in modelName.lower():
    train_data = TensorDataset(train_inputs1, train_inputs2, train_inputs3, train_masks1, train_masks2, train_masks3,
                               train_labels,train_identifiers)
    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
elif 'ensembled' in modelName.lower():
    train_data = TensorDataset(train_inputs1, train_inputs2, train_masks1, train_masks2,
                               train_labels,train_identifiers)
    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    
elif 'BERT' in modelName.upper():
    print('Simple modified BERT model...')
    train_data = TensorDataset(train_inputs, train_masks, train_labels,train_identifiers)
    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


# In[13]:


####### TESTING ########

# Prepare Hold Out Test SET
import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv(testdatafile, delimiter='\t', header=None, names=['index', 'label', 'label_notes', 'sentence', 'audio_features'])

# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))

# Create sentence and label lists
sentences = df.sentence.values
labels = df.label.values

audio = df.audio_features.values 

# Generate Data structures for Audio Features, Gender and Train Identifiers. 
test_identifiers = []

# Audio Feature Exploration
for feature_string in audio:
    identifier = getIdentifier(feature_string)
    test_identifiers.append(identifier)

configuration["test_identifiers"] = test_identifiers 


# In[14]:


if 'allin' in modelName.lower():
    print('All in Ensemble preparation')   
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    #roberta
    input_ids1 = []
    
    #distilbert
    input_ids2 = []
    
    #bert
    input_ids3 = []
    
    #bertweet
    input_ids4 = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #roberta
        encoded_sent1 = tokenizer1.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids1.append(encoded_sent1)
        
        #distilbert
        encoded_sent2 = tokenizer2.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids2.append(encoded_sent2)
        
        #bert
        encoded_sent3 = tokenizer3.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids3.append(encoded_sent3)
        
        #bertweet
        encoded_sent4 = tokenizer4.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids4.append(encoded_sent4)

    # Pad our input tokens
    input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")
    
    input_ids2 = pad_sequences(input_ids2, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")
    
    input_ids3 = pad_sequences(input_ids3, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")
    
    input_ids4 = pad_sequences(input_ids4, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")

    # Create attention masks roberta
    attention_masks1 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids1:
        seq_mask1 = [float(i>0) for i in seq]
        attention_masks1.append(seq_mask1) 
        
    #attention mask DistilBERT
    attention_masks2 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids2:
        seq_mask2 = [float(i>0) for i in seq]
        attention_masks2.append(seq_mask2) 
    
    #attention mask BERT
    attention_masks3 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids3:
        seq_mask3 = [float(i>0) for i in seq]
        attention_masks3.append(seq_mask3)
    
    #attention mask BERTweet
    attention_masks4 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids4:
        seq_mask4 = [float(i>0) for i in seq]
        attention_masks4.append(seq_mask4)
        
elif 'triangle' in modelName.lower():
    print('Triangle Ensemble preparation')   
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    #roberta
    input_ids1 = []
    
    #distilbert
    input_ids2 = []
    
    #bert
    input_ids3 = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #roberta
        encoded_sent1 = tokenizer1.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids1.append(encoded_sent1)
        
        #distilbert
        encoded_sent2 = tokenizer2.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids2.append(encoded_sent2)
        
        #bert
        encoded_sent3 = tokenizer3.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids3.append(encoded_sent3)

    # Pad our input tokens
    input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")
    
    input_ids2 = pad_sequences(input_ids2, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")
    
    input_ids3 = pad_sequences(input_ids3, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")

    # Create attention masks roberta
    attention_masks1 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids1:
        seq_mask1 = [float(i>0) for i in seq]
        attention_masks1.append(seq_mask1) 
        
    #attention mask DistilBERT
    attention_masks2 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids2:
        seq_mask2 = [float(i>0) for i in seq]
        attention_masks2.append(seq_mask2) 
    
    #attention mask BERT
    attention_masks3 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids3:
        seq_mask3 = [float(i>0) for i in seq]
        attention_masks3.append(seq_mask3) 
        
elif 'special' in modelName.lower():
    print('Special Ensemble preparation')   
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    #roberta
    input_ids1 = []
    
    #distilbert
    input_ids2 = []
    
    #bert
    input_ids3 = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #roberta
        encoded_sent1 = tokenizer1.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids1.append(encoded_sent1)
        
        #distilbert
        encoded_sent2 = tokenizer2.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids2.append(encoded_sent2)
        
        #bert
        encoded_sent3 = tokenizer3.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids3.append(encoded_sent3)

    # Pad our input tokens
    input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")
    
    input_ids2 = pad_sequences(input_ids2, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")
    
    input_ids3 = pad_sequences(input_ids3, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")

    # Create attention masks roberta
    attention_masks1 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids1:
        seq_mask1 = [float(i>0) for i in seq]
        attention_masks1.append(seq_mask1) 
        
    #attention mask DistilBERT
    attention_masks2 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids2:
        seq_mask2 = [float(i>0) for i in seq]
        attention_masks2.append(seq_mask2) 
    
    #attention mask BERT
    attention_masks3 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids3:
        seq_mask3 = [float(i>0) for i in seq]
        attention_masks3.append(seq_mask3) 
    
elif 'ensembled' in modelName.lower():
    print('Ensemble preparation')   
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    #roberta
    input_ids1 = []
    
    #bert
    input_ids2 = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #roberta
        encoded_sent1 = tokenizer1.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids1.append(encoded_sent1)
        
        #bert
        encoded_sent2 = tokenizer2.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids2.append(encoded_sent2)

    # Pad our input tokens
    input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")
    
    input_ids2 = pad_sequences(input_ids2, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")

    # Create attention masks roberta
    attention_masks1 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids1:
        seq_mask1 = [float(i>0) for i in seq]
        attention_masks1.append(seq_mask1) 
        
    #attention mask BERT
    attention_masks2 = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids2:
        seq_mask2 = [float(i>0) for i in seq]
        attention_masks2.append(seq_mask2) 
        
elif 'BERT' in modelName.upper():
    print('BERT/RoBERTa/BERTweet/DistilBERT preparation')
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids.append(encoded_sent)

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask) 
    
# Convert to tensors.
if 'allin' in modelName.lower():
    #Model1
    prediction_inputs1 = torch.tensor(input_ids1)
    prediction_masks1 = torch.tensor(attention_masks1)
    
    #Model2
    prediction_inputs2 = torch.tensor(input_ids2)
    prediction_masks2 = torch.tensor(attention_masks2)
    
    #Model3
    prediction_inputs3 = torch.tensor(input_ids3)
    prediction_masks3 = torch.tensor(attention_masks3)
    
    #Model4
    prediction_inputs4 = torch.tensor(input_ids4)
    prediction_masks4 = torch.tensor(attention_masks4)
    
elif 'triangle' in modelName.lower():
    #Model1
    prediction_inputs1 = torch.tensor(input_ids1)
    prediction_masks1 = torch.tensor(attention_masks1)
    
    #Model2
    prediction_inputs2 = torch.tensor(input_ids2)
    prediction_masks2 = torch.tensor(attention_masks2)
    
    #Model3
    prediction_inputs3 = torch.tensor(input_ids3)
    prediction_masks3 = torch.tensor(attention_masks3)
    
elif 'special' in modelName.lower():
    #Model1
    prediction_inputs1 = torch.tensor(input_ids1)
    prediction_masks1 = torch.tensor(attention_masks1)
    
    #Model2
    prediction_inputs2 = torch.tensor(input_ids2)
    prediction_masks2 = torch.tensor(attention_masks2)
    
    #Model3
    prediction_inputs3 = torch.tensor(input_ids3)
    prediction_masks3 = torch.tensor(attention_masks3)

elif 'ensembled' in modelName.lower():
    
    #Model1
    prediction_inputs1 = torch.tensor(input_ids1)
    prediction_masks1 = torch.tensor(attention_masks1)
    
    #Model2
    prediction_inputs2 = torch.tensor(input_ids2)
    prediction_masks2 = torch.tensor(attention_masks2)

elif 'BERT' in modelName.upper():   
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    
prediction_labels = torch.tensor(labels) #All
prediction_identifiers = torch.tensor([int(id) for id in test_identifiers]) #All

# Create the DataLoader.
if 'allin' in modelName.lower():
    prediction_data = TensorDataset(prediction_inputs1, prediction_inputs2, prediction_inputs3, prediction_inputs4,
                                     prediction_masks1, prediction_masks2, prediction_masks3, prediction_masks4,
                                     prediction_labels,prediction_identifiers)
        
    # sampler for all models
    prediction_sampler = SequentialSampler(prediction_data)
    
    # Dataloader for testing
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    print("Prediction data is created for allin ensembled model.")
    
elif 'triangle' in modelName.lower():
    prediction_data = TensorDataset(prediction_inputs1, prediction_inputs2, prediction_inputs3, 
                                     prediction_masks1, prediction_masks2, prediction_masks3, 
                                     prediction_labels,prediction_identifiers)
        
    # sampler for both models
    prediction_sampler = SequentialSampler(prediction_data)
    
    # Dataloader for testing
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    print("Prediction data is created for traingle ensembled model.")
    
elif 'special' in modelName.lower():
    prediction_data = TensorDataset(prediction_inputs1, prediction_inputs2, prediction_inputs3, 
                                     prediction_masks1, prediction_masks2, prediction_masks3, 
                                     prediction_labels,prediction_identifiers)
        
    # sampler for both models
    prediction_sampler = SequentialSampler(prediction_data)
    
    # Dataloader for testing
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    print("Prediction data is created for special ensembled model.")
    
elif 'ensembled' in modelName.lower():
    prediction_data = TensorDataset(prediction_inputs1, prediction_inputs2, 
                                     prediction_masks1, prediction_masks2, 
                                     prediction_labels,prediction_identifiers)
        
    # sampler for both models
    prediction_sampler = SequentialSampler(prediction_data)
    
    # Dataloader for testing
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    print("Prediction data is created for ensembled model.")
    
elif 'BERT' in modelName.upper():   
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels,prediction_identifiers)

    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


# In[15]:


###### CALLING THE MODELS ########
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.autograd import Variable
from torch.nn import functional as F

dependencies = ['torch', 'numpy', 'resampy', 'soundfile']
badIDs = []

if modelName.upper() == 'BERT':  
    from models.BERTBase import BERTBase
    print('Simple BERT!')
    model = BERTBase.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False # Whether the model returns all hidden-states.
        )
elif modelName.lower() == 'bertattention':
    from models.BERTBaseAttention import BERTBase
    print('Simple BERT with attention layer!')
    model = BERTBase.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False # Whether the model returns all hidden-states.
        )
elif modelName.lower() == 'bertattentionlstm':
    from models.BERTBaseAttentionLSTM import BERTBase
    print('Simple BERT with attention layer and LSTM!')
    model = BERTBase.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False # Whether the model returns all hidden-states.
        )
elif modelName.lower() == 'multi3bertattentionlstm':
    from models.Multi3BERTAttentionLSTM import BERT
    print('Simple BERT (3x) with attention layer and LSTM!')
    model = BERT.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False # Whether the model returns all hidden-states.
        )
elif modelName.lower() == 'multi6bertattentionlstm':
    from models.Multi6BERTAttentionLSTM import BERT
    print('Simple BERT (6x) with attention layer and LSTM!')
    model = BERT.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False # Whether the model returns all hidden-states.
        )
elif modelName.lower() == 'multi3bert':
    from models.Multi3BERT import BERT
    print('Simple BERT (3x)!')
    model = BERT.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False # Whether the model returns all hidden-states.
        )
    
    
elif modelName.lower() == 'roberta':
    from models.BERTBase_RoBERTa import RoBERTa
    print('Simple RoBERTa!')
    model = RoBERTa.from_pretrained(
        "roberta-base", # Use the 12-layer RoBERTa model.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False) # Whether the model returns all hidden-states. 
elif modelName.lower() == 'robertaattention':
    from models.BERTBase_RoBERTaAttention import RoBERTa
    print('Simple RoBERTa with attention layer!')
    model = RoBERTa.from_pretrained(
        "roberta-base", # Use the 12-layer RoBERTa model.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False) # Whether the model returns all hidden-states.
elif modelName.lower() == 'robertaattentionlstm':
    from models.BERTBase_RoBERTaAttentionLSTM import RoBERTa
    print('Simple RoBERTa with attention layer and LSTM!')
    model = RoBERTa.from_pretrained(
        "roberta-base", # Use the 12-layer RoBERTa model.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False) # Whether the model returns all hidden-states. 
elif modelName.lower() == 'robertalstm':
    from models.BERTBase_RoBERTaLSTM import RoBERTa
    print('Simple RoBERTa with LSTM!')
    model = RoBERTa()
elif modelName.lower() == 'multirobertaattentionlstm':
    from models.MultiRoBERTaAttentionLSTM import RoBERTa
    print('Multi RoBERTa (2x) with attention layer and LSTM!')
    model = RoBERTa.from_pretrained(
        "roberta-base", # Use the 12-layer RoBERTa model.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False) # Whether the model returns all hidden-states.
elif modelName.lower() == 'multi3robertaattentionlstm':
    from models.Multi3RoBERTaAttentionLSTM import RoBERTa
    print('Multi RoBERTa (3x) with attention layer and LSTM!')
    model = RoBERTa.from_pretrained(
        "roberta-base", # Use the 12-layer RoBERTa model.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False) # Whether the model returns all hidden-states.
elif modelName.lower() == 'multi3roberta':
    from models.Multi3RoBERTa import RoBERTa
    print('Multi RoBERTa (3x).')
    model = RoBERTa.from_pretrained(
        "roberta-base", # Use the 12-layer RoBERTa model.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False) # Whether the model returns all hidden-states.

    
elif modelName.lower() == 'bertweet':
    from models.BERTweetModel import BERTweet
    print('Bertweet!')
    model = BERTweet()
elif modelName.lower() == 'bertweetattention':
    from models.BERTweetModelAttention import BERTweet
    print('BERTweet with Attention layer!')
    model = BERTweet()
elif modelName.lower() == 'bertweetattentionlstm':
    from models.BERTweetModelAttentionLSTM import BERTweet
    print('BERTweet with Attention layer and LSTM!')
    model = BERTweet()
    
elif modelName.lower() == 'distilbert':
    from models.DistilBERTModel import DistilBERT
    print('Normal DistilBert is used!')
    model = DistilBERT()
elif modelName.lower() == 'distilbertattention':
    from models.DistilBERTModelAttention import DistilBERT
    print('DistilBERT with attention layer is used.')
    model = DistilBERT()
elif modelName.lower() == 'distilbertattentionlstm':
    from models.DistilBERTModelAttentionLSTM import DistilBERT
    print('DistilBERT with attention layer and LSTM is used.')
    model = DistilBERT()
elif modelName.lower() == 'multidistilbertattentionlstm':
    from models.MultiDistilBERT1classifierAttentionLSTM import MultiDistil
    print('Multiple DistilBERT with attention layer and LSTM is used.')
    model = MultiDistil()
elif modelName.lower() == 'multi3distilbertattentionlstm':
    from models.MultiDistil3BERT1classifierAttentionLSTM import MultiDistil
    print('Multiple DistilBERT (3x) with attention layer and LSTM is used.')
    model = MultiDistil()
elif modelName.lower() == 'multi6distilbertattentionlstm':
    from models.MultiDistil6BERT1classifierAttentionLSTM import MultiDistil
    print('Multiple DistilBERT (6x) with attention layer and LSTM is used.')
    model = MultiDistil()
    
elif modelName.lower() == 'ensembled':
    print('Ensembled model simple way is used')
    from models.RoBERTEnsembleCopy1 import RoBERTEnsemble
    model = RoBERTEnsemble()
elif modelName.lower() == 'ensembledattention':
    print('Ensembled model simple way with attention is used')
    from models.RoBERTEnsembleAttention import RoBERTEnsemble
    model = RoBERTEnsemble()
elif modelName.lower() == 'ensembledattentionlstm':
    print('Ensembled model simple way with attention & LSTM is used')
    from models.RoBERTEnsembleAttentionLSTM import RoBERTEnsemble
    model = RoBERTEnsemble()
    
elif modelName.lower() == 'ensembled1':
    print('Ensembled model with one final classifier is used')
    from models.RoBERTEnsemble1classifier import RoBERTEnsemble
    model = RoBERTEnsemble()
elif modelName.lower() == 'ensembled1attention':
    print('Ensembled model with one final classifier and attention is used')
    from models.RoBERTEnsemble1classifierAttention import RoBERTEnsemble
    model = RoBERTEnsemble()
elif modelName.lower() == 'ensembled1attentionlstm':
    print('Ensembled model with one final classifier and attention & LSTM is used')
    from models.RoBERTEnsemble1classifierAttentionLSTM import RoBERTEnsemble
    model = RoBERTEnsemble()
    
elif modelName.lower() == 'distilaensembled':
    print('DistilBERTa Ensembled model with one final classifier is used')
    from models.DistilBERTaEnsemble1classifier import DistilaEnsemble
    model = DistilaEnsemble()
elif modelName.lower() == 'distilaensembledattention':
    print('DistilBERTa Ensembled model with one final classifier and attention is used')
    from models.DistilBERTaEnsemble1classifierAttention import DistilaEnsemble
    model = DistilaEnsemble()
elif modelName.lower() == 'distilaensembledattentionlstm':
    print('DistilBERTa Ensembled model with one final classifier and attention & LSTM is used')
    from models.DistilBERTaEnsemble1classifierAttentionLSTM import DistilaEnsemble
    model = DistilaEnsemble()
elif modelName.lower() == 'distilaensembledlstm':
    print('DistilBERTa Ensembled model with one final classifier and LSTM is used')
    from models.DistilBERTaEnsemble1classifierLSTM import DistilaEnsemble
    model = DistilaEnsemble()
    
elif modelName.lower() == 'triangleensembled':
    print('Triangle Ensembled model with one final classifier is used')
    from models.TriangleEnsemble1classifier import TriangleEnsemble
    model = TriangleEnsemble()
elif modelName.lower() == 'triangleensembledattention':
    print('Triangle Ensembled model with one final classifier and attention is used')
    from models.TriangleEnsemble1classifierAttention import TriangleEnsemble
    model = TriangleEnsemble()
elif modelName.lower() == 'triangleensembledattentionlstm':
    print('Triangle Ensembled model with one final classifier and attention & LSTM is used')
    from models.TriangleEnsemble1classifierAttentionLSTM import TriangleEnsemble
    model = TriangleEnsemble()
elif modelName.lower() == 'triangleensembledlstm':
    print('Triangle Ensembled model with one final classifier and LSTM is used')
    from models.TriangleEnsemble1classifierLSTM import TriangleEnsemble
    model = TriangleEnsemble()
    
elif modelName.lower() == 'allinensembled':
    print('Allin Ensembled model with one final classifier is used')
    from models.AllinEnsemble1classifier import AllInEnsemble
    model = AllInEnsemble()
elif modelName.lower() == 'allinensembledattention':
    print('Allin Ensembled model with one final classifier and attention is used')
    from models.AllinEnsemble1classifierAttention import AllInEnsemble
    model = AllInEnsemble()
elif modelName.lower() == 'allinensembledattentionlstm':
    print('Allin Ensembled model with one final classifier and attention & LSTM is used')
    from models.AllinEnsemble1classifierAttentionLSTM import AllInEnsemble
    model = AllInEnsemble()
elif modelName.lower() == 'allinmajorensembled':
    print('Allin Ensembled model with majority classification.')
    from models.AllinEnsembleMajor import AllInEnsemble
    model = AllInEnsemble()
    
elif modelName.lower() == 'allinensembledlstm':
    print('Allin Ensembled model with majority classification.')
    from models.AllinEnsemble1classifierLSTM import AllInEnsemble
    model = AllInEnsemble()
elif modelName.lower() == 'allinensembledattentionlstm01':
    print('Allin Ensembled model with one final classifier and attention & LSTM is used')
    from models.AllinEnsemble1classifierAttentionLSTM01 import AllInEnsemble
elif modelName.lower() == 'allinensembledattentionlstm02':
    print('Allin Ensembled model with one final classifier and attention & LSTM is used')
    from models.AllinEnsemble1classifierAttentionLSTM02 import AllInEnsemble
elif modelName.lower() == 'allinensembledattentionlstm03':
    print('Allin Ensembled model with one final classifier and attention & LSTM is used')
    from models.AllinEnsemble1classifierAttentionLSTM03 import AllInEnsemble
    
elif modelName.lower() == 'specialensembled':
    print('Special Ensembled model with one final classifier is used')
    from models.SpecialEnsemble1classifier import SpecialEnsemble
    model = SpecialEnsemble()
elif modelName.lower() == 'specialensembledattention':
    print('Special Ensembled model with one final classifier and attention is used')
    from models.SpecialEnsemble1classifierAttention import SpecialEnsemble
    model = SpecialEnsemble()
elif modelName.lower() == 'specialensembledattentionlstm':
    print('Special Ensembled model with one final classifier and attention and LSTM is used')
    from models.SpecialEnsemble1classifierAttentionLSTM import SpecialEnsemble
    model = SpecialEnsemble()

# DistilBERT BERTweet Ensemble  
elif modelName.lower() == 'distiltweetensembled':
    print('DistilTweet Ensembled model with one final classifier is used')
    from models.DistilTweetEnsemble1Classifier import DistilTweetEnsemble
    model = DistilTweetEnsemble()
elif modelName.lower() == 'distiltweetensembledattention':
    print('DistilTweet Ensembled model with one final classifier and attention is used')
    from models.DistilTweetEnsemble1ClassifierAttention import DistilTweetEnsemble
    model = DistilTweetEnsemble()
elif modelName.lower() == 'distiltweetensembledattentionlstm':
    print('DistilTweet Ensembled model with one final classifier and attention and LSTM is used')
    from models.DistilTweetEnsemble1ClassifierAttentionLSTM import DistilTweetEnsemble
    model = DistilTweetEnsemble()

# RoBERTa BERTweet Ensemble   
elif modelName.lower() == 'robtweetensembled':
    print('RobTweet Ensembled model with one final classifier is used')
    from models.RobTweettEnsemble1Classifier import RobTweetEnsemble
    model = RobTweetEnsemble()
elif modelName.lower() == 'robtweetensembledattention':
    print('RobTweet Ensembled model with one final classifier and attention is used')
    from models.RobTweetEnsemble1ClassifierAttention import RobTweetEnsemble
    model = RobTweetEnsemble()
elif modelName.lower() == 'robtweetensembledattentionlstm':
    print('RobTweet Ensembled model with one final classifier and attention and LSTM is used')
    from models.RobTweetEnsemble1ClassifierAttentionLSTM import RobTweetEnsemble
    model = RobTweetEnsemble()

    
model.cuda()  


# In[ ]:


# This block is for information purposes only. As well as to understand and modify the model. 
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())
modelDescription = str( model)
configuration['modelDescription'] = modelDescription


# In[ ]:


from transformers import AdamW
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 2e-8# args.adam_epsilon  - default is 1e-8.
                )


# In[ ]:


from transformers import get_linear_schedule_with_warmup

# Number of training epochs (authors recommend between 2 and 4)

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


# In[ ]:


import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

from sklearn.metrics import matthews_corrcoef, f1_score, auc, recall_score, precision_score, accuracy_score, roc_auc_score


# In[ ]:


import random

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []
mcc_values = []
fone_values = []
acc_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        if 'allin' in modelName.lower():  
            b_input_ids1 = batch[0].to(device)
            b_input_ids2 = batch[1].to(device)
            b_input_ids3 = batch[2].to(device)
            b_input_ids4 = batch[3].to(device)
            b_input_mask1 = batch[4].to(device)
            b_input_mask2 = batch[5].to(device)
            b_input_mask3 = batch[6].to(device)
            b_input_mask4 = batch[7].to(device)
            b_labels = batch[8].to(device)
        elif 'triangle' in modelName.lower():  
            b_input_ids1 = batch[0].to(device)
            b_input_ids2 = batch[1].to(device)
            b_input_ids3 = batch[2].to(device)
            b_input_mask1 = batch[3].to(device)
            b_input_mask2 = batch[4].to(device)
            b_input_mask3 = batch[5].to(device)
            b_labels = batch[6].to(device)
        elif 'special' in modelName.lower():  
            b_input_ids1 = batch[0].to(device)
            b_input_ids2 = batch[1].to(device)
            b_input_ids3 = batch[2].to(device)
            b_input_mask1 = batch[3].to(device)
            b_input_mask2 = batch[4].to(device)
            b_input_mask3 = batch[5].to(device)
            b_labels = batch[6].to(device)
        elif 'ensembled' in modelName.lower():           
            b_input_ids1 = batch[0].to(device)
            b_input_ids2 = batch[1].to(device)
            b_input_mask1 = batch[2].to(device)
            b_input_mask2 = batch[3].to(device)
            b_labels = batch[4].to(device)

        elif 'bertweet' in modelName.lower():
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
        elif 'distilbert' in modelName.lower():
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
        elif 'BERT' in modelName.upper():           
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_gender =torch.tensor([1]).to(device)
            wav_file = 'test1.wav'
        else:
            b_labels = batch[0].to(device)
            b_gender =torch.tensor([1]).to(device)
            wav_file = 'test1.wav'

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        if 'allin' in modelName.lower():
            outputs = model(b_input_ids1, b_input_ids2, b_input_ids3, b_input_ids4,
                        attention_mask1=b_input_mask1.float(), attention_mask2=b_input_mask2.float(),
                            attention_mask3=b_input_mask3.float(), attention_mask4=b_input_mask4.float(), labels=b_labels)
        elif 'triangle' in modelName.lower():
            outputs = model(b_input_ids1, b_input_ids2, b_input_ids3,
                        attention_mask1=b_input_mask1.float(), attention_mask2=b_input_mask2.float(),
                            attention_mask3=b_input_mask3.float(), labels=b_labels)
        elif 'special' in modelName.lower():
            outputs = model(b_input_ids1, b_input_ids2, b_input_ids3,
                        attention_mask1=b_input_mask1.float(), attention_mask2=b_input_mask2.float(),
                            attention_mask3=b_input_mask3.float(), labels=b_labels)
        elif 'ensembled' in modelName.lower():
            outputs = model(b_input_ids1, b_input_ids2,
                        attention_mask1=b_input_mask1.float(), attention_mask2=b_input_mask2.float(), labels=b_labels)
        elif 'roberta' in modelName.lower(): 
            outputs = model(b_input_ids, 
                        attention_mask=b_input_mask.float(), 
                        labels=b_labels)
        elif 'bertweet'in modelName.lower():
            outputs = model(b_input_ids, 
                        attention_mask=b_input_mask.float(), 
                        labels=b_labels)
        elif 'distilbert' in modelName.lower():
            outputs = model(b_input_ids, 
                        attention_mask=b_input_mask.float(), 
                        labels=b_labels)
        elif 'BERT' in modelName.upper():
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask.float(), 
                        labels=b_labels)
        else:
            outputs = model(b_gender,wav_file,labels=b_labels)
            

        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple.
        
        if 'ensembled' in modelName.lower():     
            loss = outputs
        elif 'bertweet' in modelName.lower():
            loss = outputs
        elif 'distilbert' in modelName.lower():
            loss = outputs
        else:
            loss = outputs[0]
        

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    
    print("Testing:")
    # Prediction on test set


    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        if 'allin' in modelName.lower():                
            b_input_ids1, b_input_ids2, b_input_ids3, b_input_ids4, b_input_mask1, b_input_mask2, b_input_mask3, b_input_mask4, b_labels, b_identifier = batch
        elif 'triangle' in modelName.lower():                
            b_input_ids1, b_input_ids2, b_input_ids3, b_input_mask1, b_input_mask2, b_input_mask3, b_labels, b_identifier = batch
        elif 'special' in modelName.lower():                
            b_input_ids1, b_input_ids2, b_input_ids3, b_input_mask1, b_input_mask2, b_input_mask3, b_labels, b_identifier = batch
        elif 'ensembled' in modelName.lower():                
            b_input_ids1, b_input_ids2, b_input_mask1, b_input_mask2, b_labels, b_identifier = batch
        elif modelName.lower()=='bertweet':                
            b_input_ids, b_input_mask, b_labels, b_identifier = batch
        elif modelName.lower()=='distilbert':                
            b_input_ids, b_input_mask, b_labels, b_identifier = batch
        elif 'multidistilbert' in modelName.lower():                
            b_input_ids, b_input_mask, b_labels, b_identifier = batch
        elif 'BERT' in modelName.upper():           
            b_input_ids, b_input_mask, b_labels, b_identifier = batch
            b_gender = torch.tensor([1]).to(device)
            wav_file = 'test1.wav'
        else:
            b_labels, b_identifier = batch
            b_gender = torch.tensor([1]).to(device)
            wav_file = 'test1.wav'

        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
            if 'allin' in modelName.lower():     
                outputs = model(b_input_ids1, b_input_ids2, b_input_ids3, b_input_ids4,
                                attention_mask1=b_input_mask1,attention_mask2=b_input_mask2,
                               attention_mask3=b_input_mask3, attention_mask4=b_input_mask4)
            elif 'triangle' in modelName.lower():     
                outputs = model(b_input_ids1, b_input_ids2, b_input_ids3,
                                attention_mask1=b_input_mask1,attention_mask2=b_input_mask2,
                               attention_mask3=b_input_mask3)    
            elif 'special' in modelName.lower():     
                outputs = model(b_input_ids1, b_input_ids2, b_input_ids3,
                                attention_mask1=b_input_mask1,attention_mask2=b_input_mask2,
                               attention_mask3=b_input_mask3)
            elif 'ensembled' in modelName.lower():     
                outputs = model(b_input_ids1, b_input_ids2, attention_mask1=b_input_mask1,attention_mask2=b_input_mask2)
            elif 'roberta' in modelName.lower():
                outputs = model(b_input_ids, attention_mask=b_input_mask)
            elif 'bertweet' in modelName.lower():
                outputs = model(b_input_ids, attention_mask=b_input_mask)
            elif 'distilbert' in modelName.lower():
                outputs = model(b_input_ids, attention_mask=b_input_mask)
            elif 'BERT' in modelName.upper():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            else:
                outputs = model(b_gender,wav_file)
                

        if 'ensembled' in modelName.lower(): #if modelName.lower() == 'ensembled': # not sure
            if 'major' in modelName.lower():
                flat_prediction = outputs
            else:                
                logits = outputs
        elif 'bertweet' in modelName.lower():
            logits = outputs
        elif 'distilbert' in modelName.lower():
            logits = outputs  
        else: 
            logits = outputs[0]

        # Move logits and labels to CPU
        if 'major' in modelName.lower():
            flat_prediction = flat_prediction.detach().cpu().numpy()
        else:
            logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        if 'major' in modelName.lower():
            flat_predictions.append(flat_prediction) ##############################################################################
        else:
            predictions.append(logits)
        true_labels.append(label_ids)
        
    print('DONE.')    
    # Combine the predictions for each batch into a single list of 0s and 1s.
    if 'major' in modelName.lower():
        flat_predictions=flat_predictions###########################################################################
    else:
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    trueProbability = []
    falseProbability = []

    for prediction in predictions:
        falseProbability.append(float(prediction[0][0]))
        trueProbability.append(float(prediction[0][1]))
    
    f1 = f1_score(flat_true_labels, flat_predictions)
    recall = recall_score(flat_true_labels, flat_predictions)
    precision = precision_score(flat_true_labels, flat_predictions)
    acc = accuracy_score(flat_true_labels, flat_predictions)
    #auc = roc_auc_score(flat_true_labels,trueProbability)
    results['true_labels'] = list(map(int, flat_true_labels))
    results['predictions'] = (list(map(int, flat_predictions)))
    results['falseProbability'] = (list(map(float, falseProbability)))
    results['trueProbability'] = (list(map(float, trueProbability)))

    
    results['mcc'] = mcc
    results['f1'] = f1
    results['precision'] = precision
    results['recall'] = recall
    results['acc'] = acc
    #results['auc'] = auc
    results['epoch'] = epoch_i
    results['loss'] = avg_train_loss
    
    if logtodb:
        recordRun(configuration,short_configuration,results,tags=str(question)+","+str(typeOfTask),db=db)
    
    mcc_values.append(mcc)
    fone_values.append(f1)
    acc_values.append(acc)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, [flat_true_labels, flat_predictions], 1)
    print("Confusion Matrix")
    print(cm)      
results['loss_values'] = loss_values
results['mcc_values'] = mcc_values
