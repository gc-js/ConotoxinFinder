from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import set_seed
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error
import torch
from Bio.Seq import Seq
import numpy as np
from transformers import get_scheduler
import matplotlib.pyplot as plt
device = torch.device("cuda")
import random
from sklearn.model_selection import KFold

n_splits = 10
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=4)
all_fold_metrics = []
path= "./Data/regression_train.csv"
df = pd.read_csv(path)
sequences = df["Sequence"].tolist()
labels = df["Ln_value"].tolist()

for fold, (train_index, test_index) in enumerate(kfold.split(sequences)):
    batch_size = 128
    num_epochs = 1000
    num_labels = 1
    learning_rate = 0.001
    model_checkpoint = "./model/esm_finetuned_model_mlm_best"

    def setup_seed(seed):
        set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(4)
    def spearmanr(y_true, y_pred):
        diff_pred, diff_true = y_pred - np.mean(y_pred), y_true - np.mean(y_true)
        if np.sum(diff_pred **2) == 0 or np.sum(diff_true **2) == 0:
            return 0
        else:
            return np.sum(diff_pred * diff_true) / np.sqrt(np.sum(diff_pred **2) * np.sum(diff_true **2))

    class MyDataset(Dataset):
        def __init__(self,dict_data) -> None:
            super(MyDataset,self).__init__()
            self.data=dict_data
        def __getitem__(self, index):
            return [self.data['text'][index],self.data['labels'][index]]
        def __len__(self):
            return len(self.data['text'])
        
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def collate_fn(batch):
        max_len = max([len(b[0]) for b in batch])+2
        pt_batch=tokenizer([b[0] for b in batch],padding=True,truncation=True,max_length=max_len,
                            return_tensors='pt')
        labels=torch.tensor([b[1] for b in batch])
        return {'labels':labels,'input_ids':pt_batch['input_ids'],
                'attention_mask':pt_batch['attention_mask']}
    train_sequences = [sequences[i] for i in train_index]
    test_sequences = [sequences[i] for i in test_index]
    train_labels = [labels[i] for i in train_index]
    test_labels = [labels[i] for i in test_index]

    train_dict = {"text":train_sequences,'labels':train_labels}
    test_dict = {"text":test_sequences,'labels':test_labels}

    train_data = MyDataset(train_dict)
    test_data = MyDataset(test_dict)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.to(device)

    i = 0
    train_epochs_loss = []
    test_epochs_loss = []
    best_loss = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 2
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs = {'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'labels': batch['labels'].to(device)}
            outputs = model(**inputs)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        

        model.eval()
        eval_loss = 0
        predictions, true_labels = [], []
        for batch in test_dataloader:
            with torch.no_grad():
                inputs = {'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'labels': batch['labels'].to(device)}

                outputs = model(**inputs)
                loss = outputs[0]
                eval_loss += loss.item()

                logits = outputs[1]
                logits = logits.detach().cpu().numpy()
                label_ids = inputs['labels'].cpu().numpy()

                true_labels.append(label_ids)

        avg_test_loss = eval_loss / len(test_dataloader)

        predictions = np.concatenate(logits, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        mse = mean_squared_error(true_labels, predictions)
        rmse = np.sqrt(mean_squared_error(true_labels, predictions))
        score = spearmanr(predictions, true_labels)
        i+=1
        print(f"epoch:{i} Average training loss: {avg_train_loss} Average test loss: {avg_test_loss} score: {score} rmse: {rmse}" )

        if avg_test_loss <= best_loss:
            best_mse = mse
            best_score = score
            best_loss = avg_test_loss
            best_rmse = rmse
            
            torch.save(model.state_dict(),"./model/toxin_regression_best_model.pth")

    fold_metrics = [
    best_score,
    best_rmse
    ]

    all_fold_metrics.append(fold_metrics)
    print(all_fold_metrics)


