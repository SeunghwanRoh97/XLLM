import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, Value
from evaluate import load

# Load the datasets
train_path = "./archive/train_prepared.csv"
test_path = "./archive/test_prepared.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Ensure labels are within range and of correct type (convert 1-4 to 0-3 for model compatibility)
train_df["class_index"] = train_df["class_index"].astype(int) - 1
test_df["class_index"] = test_df["class_index"].astype(int) - 1

# Split the training dataset into half-data and full-data
half_train_df = train_df.sample(frac=0.5, random_state=42)  # 50% of training data
full_train_df = train_df  # Full training data

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Function to tokenize the dataset
def tokenize_function(examples):
    combined_texts = [t + " " + d for t, d in zip(examples["title"], examples["description"])]
    return tokenizer(combined_texts, padding="max_length", truncation=True)

# Convert pandas DataFrames to Hugging Face Dataset
def convert_to_hf_dataset(df):
    dataset = Dataset.from_pandas(df[['title', 'description', 'class_index']])
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.rename_column("class_index", "labels")  # Rename class_index to labels
    dataset = dataset.cast_column("labels", Value("int64"))  # Ensure labels are cast correctly
    return dataset.remove_columns(["title", "description"])  # Remove unused columns

# Prepare training datasets
half_train_dataset = convert_to_hf_dataset(half_train_df)
full_train_dataset = convert_to_hf_dataset(full_train_df)
test_dataset = convert_to_hf_dataset(test_df)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",     # Output directory
    evaluation_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch",      # Save the model at each epoch
    num_train_epochs=3,          # Number of epochs
    per_device_train_batch_size=16,  # Training batch size
    per_device_eval_batch_size=16,   # Evaluation batch size
    logging_dir="./logs",        # Logging directory
    logging_steps=200,
    fp16=True
)

# Load metric functions
metric = load("accuracy")
precision_metric = load("precision")
recall_metric = load("recall")
f1_metric = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    accuracy = metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")  # "macro" for multi-class
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }

# Function to train and evaluate a model
def train_model(train_dataset, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4).to(device)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    model.save_pretrained(f"./{model_name}")
    return model

# check
print("CUDA available:", torch.cuda.is_available())
print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Train model on half dataset
print("Training with Half Dataset...")
half_trained_model = train_model(half_train_dataset, "bert_half_trained")

# Train model on full dataset
print("Training with Full Dataset...")
full_trained_model = train_model(full_train_dataset, "bert_full_trained")

# Evaluate the models
def evaluate_model(model, dataset):
    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)
    metrics = trainer.evaluate(dataset)
    return metrics

print("Evaluating Half-Trained Model...")
half_results = evaluate_model(half_trained_model, test_dataset)

print("Evaluating Full-Trained Model...")
full_results = evaluate_model(full_trained_model, test_dataset)

print("Half-Trained Model Results:", half_results)
print("Full-Trained Model Results:", full_results)
