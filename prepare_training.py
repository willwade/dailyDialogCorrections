import nltk
from nltk.tokenize import sent_tokenize
import random
import csv
import re

nltk.download('punkt')

def introduce_typos(sentence, typo_rate=0.1):
    """Introduce simple typos into a sentence."""
    new_sentence = ""
    for word in sentence.split():
        if random.random() < typo_rate:  # Decide whether to introduce a typo
            typo_type = random.choice(['swap', 'miss', 'add', 'wrong'])
            if typo_type == 'swap' and len(word) > 1:  # Swap letters
                idx = random.randint(0, len(word) - 2)
                word = word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
            elif typo_type == 'miss' and len(word) > 1:  # Miss letters
                idx = random.randint(0, len(word) - 1)
                word = word[:idx] + word[idx+1:]
            elif typo_type == 'add':  # Add letters
                idx = random.randint(0, len(word) - 1)
                word = word[:idx] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[idx:]
            elif typo_type == 'wrong' and len(word) > 1:  # Wrong letters
                idx = random.randint(0, len(word) - 1)
                word = word[:idx] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[idx+1:]
        new_sentence += word + " "
    return new_sentence.strip()

def clean_punctuation_spacing(sentence):
    # Correct spacing around contractions
    sentence = re.sub(r"\s+’\s+", "’", sentence)
    sentence = re.sub(r"\s+'\s+", "'", sentence)

    # Correct spacing before punctuation marks
    sentence = re.sub(r"\s+\?", "?", sentence)
    sentence = re.sub(r"\s+\.", ".", sentence)
    sentence = re.sub(r"\s+!", "!", sentence)
    sentence = re.sub(r"\s+,", ",", sentence)
    sentence = re.sub(r"\s+;", ";", sentence)
    sentence = re.sub(r"\s+:", ":", sentence)

    return sentence

def process_dialogues(text):
    """Process dialogues to create a dataset for text correction."""
    sentences = sent_tokenize(text.replace(" __eou__", ""))
    data_pairs = []
    for sentence in sentences:
        cleaned_sentence = clean_punctuation_spacing(sentence)
        typo_sentence = introduce_typos(cleaned_sentence.replace(" ", ""), typo_rate=0.1)
        data_pairs.append({"input": typo_sentence, "target": cleaned_sentence})
    return data_pairs

# New function to read dialogues from a file and process them
def process_dialogues_file(file_path):
    data_pairs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        dialogues = text.split('__eou__')
        for dialogue in dialogues:
            if dialogue.strip():  # Check if dialogue is not just whitespace
                data_pairs.extend(process_dialogues(dialogue.strip()))
    return data_pairs

# New function to save the processed data to a CSV file
def save_to_csv(data_pairs, output_file='processed_dialogues.csv'):
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["input", "target"])
        writer.writeheader()
        for pair in data_pairs:
            writer.writerow(pair)


# Example usage
file_path = 'dialogues_text.txt'
data_pairs = process_dialogues_file(file_path)
save_to_csv(data_pairs)

print(f"Processed data saved to CSV.")
