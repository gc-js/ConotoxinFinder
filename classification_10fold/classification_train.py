from sklearn.model_selection import KFold
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import set_seed
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import OrderedDict
import random
from sklearn.metrics import RocCurveDisplay, auc
import warnings
warnings.filterwarnings('ignore')

device = "cuda:0"
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, matthews_corrcoef, roc_curve, auc
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

train_epochs_acc_all = []
valid_epochs_acc_all = []
train_epochs_loss_all = []
valid_epochs_loss_all = []


df = pd.read_csv('./Data/classification_train.csv')

sequences = df["Seq"].tolist()
labels = df["Label"].tolist()
n_splits = 10
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=4)
all_fold_metrics = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(8, 8))
for fold, (train_index, val_index) in enumerate(kfold.split(sequences)):
    setup_seed(4)
    num_epochs = 200
    dropout = 0.1
    learning_rate = 0.01
    batch_size = 32
    model_checkpoint = "./model/esm_finetuned_model_mlm_best"

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
        max_len = max([len(b[0]) for b in batch])
        pt_batch=tokenizer([b[0] for b in batch], max_length=max_len, padding="max_length",truncation=True, return_tensors='pt')
        
        labels=[b[1] for b in batch]
        return {'labels':labels,'input_ids':pt_batch['input_ids'],
                'attention_mask':pt_batch['attention_mask']}
    
    train_sequences = [sequences[i] for i in train_index]
    val_sequences = [sequences[i] for i in val_index]
    train_labels = [labels[i] for i in train_index]
    val_labels = [labels[i] for i in val_index]

    train_dict = {"text":train_sequences,'labels':train_labels}
    val_dict = {"text":val_sequences,'labels':val_labels}

    train_data = MyDataset(train_dict)
    val_data = MyDataset(val_dict)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
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
            return torch.softmax(output_feature,dim=1),output_feature

    model = MyModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []
    train_epochs_acc = []
    valid_epochs_acc = []

    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        train_epoch_loss = []
        currect = 0
        for index, batch in enumerate(train_dataloader):
            batchs = {k: v for k, v in batch.items()}
            optimizer.zero_grad()
            outputs, _ = model(batchs)
            label = torch.nn.functional.one_hot(torch.tensor(batchs["labels"]).to(torch.int64), num_classes=2).float()
            loss = criterion(outputs.to(device), label.to(device))
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            train_argmax = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            for j in range(len(train_argmax)):
                if batchs["labels"][j] == train_argmax[j]:
                    currect += 1
        train_acc = currect / len(train_labels)
        train_epochs_acc.append(train_acc)
        train_epochs_loss.append(np.average(train_epoch_loss))
        
        model.eval()
        valid_epoch_loss = []
        valid_epoch_acc = []
        valid_label = []
        val_outputs_scores_all = []
        with torch.no_grad():
            currect = 0
            for index, batch in enumerate(val_dataloader):
                batchs = {k: v for k, v in batch.items()}
                outputs, output_feature = model(batchs)
                label = torch.nn.functional.one_hot(torch.tensor(batchs["labels"]).to(torch.int64), num_classes=2).float()
                loss = criterion(outputs.to(device), label.to(device))
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
                val_argmax = np.argmax(outputs.cpu(), axis=1)
                valid_epoch_acc.extend(val_argmax)
                valid_label.extend(batchs["labels"])
                val_outputs_scores = [i[1] for i in outputs.cpu().numpy().tolist()]
                val_outputs_scores_all.extend(val_outputs_scores)
        metrics_out = confusion_matrix(valid_label, valid_epoch_acc)
        val_acc = accuracy_score(valid_label, valid_epoch_acc)
        val_pre = precision_score(valid_label, valid_epoch_acc)
        val_sen = metrics_out[0][0] / (metrics_out[0][0] + metrics_out[0][1])
        val_spe = metrics_out[1][1] / (metrics_out[1][0] + metrics_out[1][1])
        val_MCC = matthews_corrcoef(valid_label, valid_epoch_acc)
        valid_epochs_loss.append(np.average(valid_epoch_loss))

        if val_acc >= best_acc:
            best_acc = val_acc
            best_pre = val_pre
            best_sen = val_sen
            best_spe = val_spe
            best_MCC = val_MCC
            val_outputs_scores_all_best = val_outputs_scores_all
            valid_label_best = valid_label
            torch.save(model.state_dict(),f"./model/best_model1_fold_{fold}.pth")
            fpr, tpr, thresholds = roc_curve(valid_label, val_outputs_scores_all)
            AUC = auc(fpr, tpr)
            valid_epochs_acc.append(val_acc)
            print(AUC)
            print(f'Fold:{fold}, epoch:{epoch}, train_acc:{round(train_acc,4)}, train_loss:{round(np.average(train_loss),4)}, val_acc:{round(val_acc,4)}, val_loss:{round(np.average(valid_loss),4)}, val_pre:{round(val_pre,4)}, val_sen:{round(val_sen,4)}, val_spe:{round(val_spe,4)}, val_MCC:{round(val_MCC,4)}')
    
    fpr, tpr, thresholds = roc_curve(valid_label_best, val_outputs_scores_all_best)
    AUC = auc(fpr, tpr)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(AUC)
    ax.plot(
    fpr,
    tpr,
    label =f"ROC fold {fold+1} (AUC = %0.3f)"%AUC,
    lw=2,
    alpha=0.8
)

    fold_metrics = [
        fold,
        best_acc,
        best_pre,
        best_sen,
        best_spe,
        best_MCC,
        AUC
    ]

    all_fold_metrics.append(fold_metrics)
    print(all_fold_metrics)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Chance level (AUC = 0.5)', alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve with variability\n(Positive label)",
)
ax.legend(loc="lower right")
plt.savefig(f"./img/ROC_curve.png")
