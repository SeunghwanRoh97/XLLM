import os
import torch
import numpy as np
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from tqdm import tqdm
from sklearn.metrics import classification_report

# Ensure device compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WARMUP_STEPS = 0
EPOCHS = 5
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
SEED = 42
MAX_LENGTH = 512

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Load .npz datasets
train_path = "imdb_tokenized_bert_train.npz"
test_path = "imdb_tokenized_bert_test.npz"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Train or test .npz file not found!")

print("Loading pre-tokenized train and test datasets (.npz format)...")

train_data = np.load(train_path, allow_pickle=True)
test_data = np.load(test_path, allow_pickle=True)

train_labels = train_data['labels']
train_tokens = train_data['tokens']

test_labels = test_data['labels']
test_tokens = test_data['tokens']

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# Preallocate
train_input_id_arr = np.zeros((len(train_labels), MAX_LENGTH), dtype=np.int64)
train_attention_mask_arr = np.zeros((len(train_labels), MAX_LENGTH), dtype=np.int64)
train_label_arr = np.zeros(len(train_labels), dtype=np.int64)

test_input_id_arr = np.zeros((len(test_labels), MAX_LENGTH), dtype=np.int64)
test_attention_mask_arr = np.zeros((len(test_labels), MAX_LENGTH), dtype=np.int64)
test_label_arr = np.zeros(len(test_labels), dtype=np.int64)

for i, (label, tokens) in enumerate(zip(train_labels, train_tokens)):
    ids = tokenizer.convert_tokens_to_ids(tokens[:MAX_LENGTH])
    train_input_id_arr[i, :len(ids)] = ids
    train_attention_mask_arr[i, :len(ids)] = 1
    train_label_arr[i] = label - 1
    
for i, (label, tokens) in enumerate(zip(test_labels, test_tokens)):
    ids = tokenizer.convert_tokens_to_ids(tokens[:MAX_LENGTH])
    test_input_id_arr[i, :len(ids)] = ids
    test_attention_mask_arr[i, :len(ids)] = 1
    test_label_arr[i] = label - 1

# Directly extract arrays, and adjust labels to be 0-based
train_input_ids = torch.tensor(train_input_id_arr)
train_attention_masks = torch.tensor(train_attention_mask_arr)
train_labels = torch.tensor(train_label_arr)

test_input_ids = torch.tensor(test_input_id_arr)
test_attention_masks = torch.tensor(test_attention_mask_arr)
test_labels = torch.tensor(test_label_arr)

# Define datasets
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

print("Dataset successfully loaded and prepared for training.")

# Optimizer and Learning Rate Scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(train_dataloader) * EPOCHS)

# Training Function
def train():
    model.train()
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            
            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss = total_loss / len(train_dataloader))
        
        print(f"Epoch {epoch+1} completed. Loss: {total_loss / len(train_dataloader):.4f}")
    
    print("Training complete.")
    model.save_pretrained("bert_imdb_model")
    print("Model and tokenizer saved.")

# Evaluation Function
def evaluate():
    model.eval()
    print("Starting evaluation...")
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(all_labels, all_preds)
    print("Evaluation Report:\n", json.dumps(report, indent=2))
    
    with open("bert_imdb_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Results saved to bert_imdb_results.json")

# Run training and evaluation
if __name__ == "__main__":
    train()
    evaluate()
