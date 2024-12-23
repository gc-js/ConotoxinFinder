import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForMaskedLM, DataCollatorForLanguageModeling, AdamW, get_scheduler
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
import math
import os
from transformers import set_seed
import matplotlib.pyplot as plt

device = "cuda:1" if torch.cuda.is_available() else "cpu"
set_seed(4)
checkpoint = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = EsmForMaskedLM.from_pretrained(checkpoint).to(device)

path = "./conoserver_data.csv"
df = pd.read_csv(path)
sequences = df["Seq"].tolist()

class SequenceMLMDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=128):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(sequence, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        inputs = {key: val.squeeze(0).to(device) for key, val in inputs.items()}
        return inputs

train_sequences, eval_sequences = train_test_split(sequences, test_size=0.1, random_state=42)
batch_size = 64
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
train_dataset = SequenceMLMDataset(train_sequences, tokenizer)
eval_dataset = SequenceMLMDataset(eval_sequences, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

optimizer = AdamW(model.parameters(), lr=5e-5)
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 200
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

best_eval_loss = float('inf')
output_dir = "./model/esm_finetuned_model_mlm_best"
os.makedirs(output_dir, exist_ok=True)

train_losses = []
eval_losses = []

for epoch in range(num_train_epochs):
    # Training
    model.train()
    total_train_loss = 0.0
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    # Evaluation
    model.eval()
    total_eval_loss = 0.0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        total_eval_loss += loss.item()
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    eval_losses.append(avg_eval_loss)
    print(f"Epoch {epoch + 1} average training loss: {avg_train_loss:.4f}, average evaluation loss: {avg_eval_loss:.4f}")

    # Save the best model based on evaluation loss
    if avg_eval_loss < best_eval_loss:
        best_eval_loss = avg_eval_loss
        print(f"New best evaluation loss: {best_eval_loss:.4f}. Saving model...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

plt.figure(figsize=(10, 6))
plt.plot(range(0, num_train_epochs), train_losses, label='Training Loss')
plt.plot(range(0, num_train_epochs), eval_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
xticks = np.arange(0, num_train_epochs + 1, 20)
plt.xticks(xticks)
plt.legend()
plt.grid(True)
plt.savefig('./img/loss_curves.png')
plt.show()
