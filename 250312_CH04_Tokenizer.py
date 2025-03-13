# The practice in Ch 4
text = open('example.txt', 'r').read()
words = text.split(" ")
tokens = {v: k for k, v in enumerate(words)}

print(tokens)

token_map = map(lambda t: tokens[t], words)
print(list(token_map))

#############################################################################
### Now we will train BPE algorithm based TOKENIZER from wikitext dataset ###
#############################################################################

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset

# Load the Wikitext-103 dataset from Hugging Face
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Extract text data from train, test, and validation splits
files = []
for split in ["train", "validation", "test"]:
    text_data = dataset[split]["text"]
    
    # Save as a temporary text file (needed for tokenizers)
    temp_file = f"wikitext_{split}.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for line in text_data:
            if line.strip():  # Avoid empty lines
                f.write(line + "\n")
    
    files.append(temp_file)

# Initialize the tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Define the BPE trainer
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# Use whitespace pre-tokenizer to preserve word boundaries
tokenizer.pre_tokenizer = Whitespace()

# Train tokenizer on dataset
tokenizer.train(files, trainer)

# Save trained tokenizer
tokenizer.save("bpe_tokenizer_wikitext.json")

print("Training complete! Tokenizer saved as 'bpe_tokenizer_wikitext.json'")

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer_wikitext.json")

# Test sentence
text = "Elon Musk is the CEO of Tesla and was born in Pretoria."

# Encode the test sentence
encoded = tokenizer.encode(text)

# Print the tokenized output
print("Tokenized IDs:", encoded.ids)
print("Tokens:", encoded.tokens)