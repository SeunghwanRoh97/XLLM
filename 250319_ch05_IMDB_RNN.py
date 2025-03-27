import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import re
import gensim.downloader as api
from transformers import BertTokenizer
import numpy as np
from sklearn.metrics import precision_score
from torch.nn.utils.rnn import pad_sequence
import os

# Ensure device compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load IMDB dataset
train_iter, test_iter = torchtext.datasets.IMDB(split=("train", "test"))

# Define IMDBataset
class IMDBDataset(Dataset):
    def __init__(self, data, vocab=None):
        self.samples = data
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        label, tokens = self.samples[idx]
        text = [self.vocab[token] for token in tokens]
        return torch.tensor(label - 1, dtype=torch.long), torch.tensor(text, dtype=torch.long)

# ============================
# Regex Tokenizer + Word2Vec + RNN
# ============================

# 'Fast tokenizer' to used constructing Tokenized dataset
def fast_tokenizer(text):
    return re.findall(r"\b\w+\b", text.lower())  # Extract words efficiently

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield fast_tokenizer(text)

# 'BERT' Tokenizer to prepare later
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

word2vec_model = api.load("word2vec-google-news-300")
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

# Convert words to pretrained Word2Vec vectors
embedding_dim = 300
word_vectors = np.random.rand(len(vocab), embedding_dim)
for word, idx in vocab.get_stoi().items():
    if word in word2vec_model:
        word_vectors[idx] = word2vec_model[word]

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_vectors), freeze=False)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)
        print("RNNClassifier model initialized.")
    
    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden[-1])

# Function s.t. makes one batch size
def collate_batch(batch):
    labels, texts = zip(*batch)
    labels = torch.stack(labels)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    return labels, texts_padded
    
# Set RNN model
rnn_model = RNNClassifier(len(vocab), 300, 128, 2).to(device)

# ============================
# Train & Evaluate Function (for all models)
# ============================

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_results(file_path, precision):
    with open(file_path, "w") as f:
        f.write(f"Precision: {precision:.4f}\n")

def train_model(model, dataloader, criterion, optimizer, epochs=5, model_path="model.pth"):
    model.train()
    print("Training started.")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            label, text = batch  # ✅ RNN still receives two values
            text, label = text.to(device), label.to(device)
            output = model(text)  # RNN model just takes text
            
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} completed. Loss: {total_loss/len(dataloader)}")
    print("Training finished.")
    save_model(model, model_path)

# ============================
# Evaluation Function (Precision Score)
# ============================
def evaluate_model(model, dataloader, result_path="results.txt"):
    model.eval()
    print("Evaluation started.")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            label, text = batch
            text, label = text.to(device), label.to(device)
            output = model(text)  # RNN model takes only text input
            
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())
    precision = precision_score(all_labels, all_preds, average='binary')
    print(f"Model Precision: {precision:.4f}")
    print("Evaluation completed.")
    save_results(result_path, precision)

# ============================
# Usage
# ============================
# Choose Optimizer and criterion
optimizer = optim.AdamW(rnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("optimizer initialized.")

if __name__ == "__main__":
    print("Preparing dataset...")
    
    # Define file paths
    CACHE = {
        "rnn_train": "imdb_tokenized_rnn_train.npz",
        "rnn_test": "imdb_tokenized_rnn_test.npz",
        "bert_train": "imdb_tokenized_bert_train.npz",
        "bert_test": "imdb_tokenized_bert_test.npz"
    }

    def save_tokenized_data(path, data):
        np.savez(path,
                 labels=[x[0] for x in data],
                 tokens=[x[1] for x in data],
                 allow_pickle=True)

    def load_tokenized_data(path):
        data = np.load(path, allow_pickle=True)
        return list(zip(data['labels'], data['tokens']))
    
    if all(os.path.exists(path) for path in CACHE.values()):
        print("Loading all tokenized datasets from cache...")
        data_rnn_train = load_tokenized_data(CACHE["rnn_train"])
        data_rnn_test = load_tokenized_data(CACHE["rnn_test"])
        data_bert_train = load_tokenized_data(CACHE["bert_train"])
        data_bert_test = load_tokenized_data(CACHE["bert_test"])

    else:
        print("Tokenizing datasets...")
        train_iter, test_iter = torchtext.datasets.IMDB(split=("train", "test"))
        data_rnn_train = [(label, fast_tokenizer(text)) for label, text in train_iter]
        train_iter, test_iter = torchtext.datasets.IMDB(split=("train", "test"))  # reload
        data_bert_train = [(label, bert_tokenizer.tokenize(text)) for label, text in train_iter]

        train_iter, test_iter = torchtext.datasets.IMDB(split=("train", "test"))  # reload
        data_rnn_test = [(label, fast_tokenizer(text)) for label, text in test_iter]
        train_iter, test_iter = torchtext.datasets.IMDB(split=("train", "test"))  # reload
        data_bert_test = [(label, bert_tokenizer.tokenize(text)) for label, text in test_iter]

        print("Saving tokenized datasets...")
        save_tokenized_data(CACHE["rnn_train"], data_rnn_train)
        save_tokenized_data(CACHE["rnn_test"], data_rnn_test)
        save_tokenized_data(CACHE["bert_train"], data_bert_train)
        save_tokenized_data(CACHE["bert_test"], data_bert_test)

    print("Tokenized datasets are ready.")

    # Define DataLoaders
    batch_size = 32

    train_dataset_rnn = IMDBDataset(data_rnn_train, vocab)
    test_dataset_rnn = IMDBDataset(data_rnn_test, vocab)
    train_dataloader_rnn = DataLoader(train_dataset_rnn, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_batch)
    test_dataloader_rnn = DataLoader(test_dataset_rnn, batch_size=batch_size, num_workers=4, collate_fn=collate_batch)

    print("DataLoaders initialized successfully.")

    # Train and Evaluate
    train_model(rnn_model, train_dataloader_rnn, criterion, optimizer, epochs=5, model_path="rnn_model.pth")
    evaluate_model(rnn_model, test_dataloader_rnn, result_path="rnn_results.txt")
