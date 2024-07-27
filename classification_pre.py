import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import set_seed
import torch
import torch.nn as nn
from collections import OrderedDict
import warnings
import random
warnings.filterwarnings('ignore')
set_seed(4)  
device = "cuda:0"
model_checkpoint = "facebook/esm2_t12_35M_UR50D"
dropout = 0.1

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(4)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=320)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(320,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.output_layer = nn.Linear(64,2)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        with torch.no_grad():
            bert_output = self.bert(input_ids=x['input_ids'].to(device),attention_mask=x['attention_mask'].to(device)) 
        output_feature = self.dropout(bert_output["logits"])
        output_feature = self.dropout(self.relu(self.bn1(self.fc1(output_feature))))
        output_feature = self.dropout(self.relu(self.bn2(self.fc2(output_feature))))
        output_feature = self.dropout(self.relu(self.bn3(self.fc3(output_feature))))
        output_feature = self.dropout(self.output_layer(output_feature))
        return torch.softmax(output_feature,dim=1)

model = MyModel()
model.load_state_dict(torch.load("best_model_classification.pth"))
model = model.to(device)
model.eval()

df = pd.read_csv('test.csv')

test_seq = df["Seq"].tolist()
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def ACE(file):
    test_sequences = file
    max_len = 30
    test_data = tokenizer(test_sequences, max_length=max_len, padding="max_length",truncation=True, return_tensors='pt')
    out_probability = []
    with torch.no_grad():
        predict = model(test_data)
        out_probability.extend(np.max(np.array(predict.cpu()),axis=1).tolist())
        test_argmax = np.argmax(predict.cpu(), axis=1).tolist()
    id2str = {0:"non-nAChRs", 1:"nAChRs"}
    return id2str[test_argmax[0]], out_probability[0]

seq_all = []
output_all = []
probability_all = []
for seq in test_seq:
    output, probability = ACE(str(seq))
    print(output)
    
    seq_all.append(seq)
    output_all.append(output)
    probability_all.append(probability)

summary = OrderedDict()
summary['Seq'] = seq_all
summary['Class'] = output_all
summary['Probability'] = probability_all
summary_df = pd.DataFrame(summary)
summary_df.to_csv('output.csv', index=False)