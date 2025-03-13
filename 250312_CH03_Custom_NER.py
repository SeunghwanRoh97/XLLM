from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import classification_report
import numpy as np
from transformers import pipeline

# Load dataset from Hugging Face
dataset = load_dataset("conll2003", trust_remote_code=True)

# Print a sample
print(dataset["train"][0])

# Load a BERT tokenizer
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Label mapping (CoNLL-2003 has 9 labels)
label_list = dataset["train"].features["ner_tags"].feature.names
print(label_list)  # ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        padding="max_length",  # Ensures all sequences have the same length
        is_split_into_words=True,
        max_length=128  # You can adjust this based on your GPU memory
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word IDs
        new_labels = []
        previous_word = None
        for word_idx in word_ids:
            if word_idx is None:
                new_labels.append(-100)  # Special tokens get -100 (ignored in training)
            elif word_idx != previous_word:
                new_labels.append(label[word_idx])  # Assign the first sub-token the label
            else:
                new_labels.append(-100)  # Assign -100 to subwords to ignore them
            previous_word = word_idx
        labels.append(new_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization again
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Load pretrained BERT model for token classification
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

# Training arguments
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

# Compute accuracy function for evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [[label_list[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]

    return {"ner_f1": classification_report(true_labels, true_predictions, output_dict=True)["weighted avg"]["f1-score"]}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

################### INFORM #####################
### It took 3 hour in rtx-1080ti environment ###
################################################

# Run evaluation
results = trainer.evaluate(tokenized_datasets["test"])
print(results)

# Create a mapping from label index to actual label name
id2label = {i: label for i, label in enumerate(label_list)}

# Load fine-tuned model
ner_pipeline = pipeline("ner", model="./ner_model/checkpoint-1317", tokenizer=tokenizer)

# Test sentence
text = "Elon Musk is the CEO of Tesla and was born in Pretoria."

# Run NER
ner_results = ner_pipeline(text)

# Print results
for entity in ner_results:
    word = entity['word'].replace("##", "")  # Remove subword artifacts
    label = id2label[int(entity['entity'].split("_")[-1])]  # Convert LABEL_X to real label
    score = entity['score']
    print(f"{word} -> {label} (Score: {score:.2f})")