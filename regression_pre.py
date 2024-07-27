import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification,AutoConfig
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from collections import OrderedDict	
from transformers import set_seed
import random
def setup_seed(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(4)

device = "cuda"
model_checkpoint = "facebook/esm2_t6_8M_UR50D"

config = AutoConfig.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

file_path= "test.txt"
seqs = []
f = open(file_path, 'r', encoding='utf-8')
for line in f:
    line = line.replace('\n','')
    line = line.replace(' ','')
    if line.islower():
        seqs.append(str((Seq(line).translate())))
    else:
        seqs.append(line)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1)
model.load_state_dict(torch.load("best_model_regression.pth"))
model = model.to(device)

if __name__ == '__main__':
    num = 0
    value_all = []
    for i in seqs:
        tokenizer_test = tokenizer(i, return_tensors='pt').to(device)
        with torch.no_grad():
            value = model(**tokenizer_test)
            value_all.append(np.exp(value["logits"][0].item()))

    summary = OrderedDict()
    summary['Seq'] = seqs
    summary['Value'] = value_all
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('output.csv', index=False)