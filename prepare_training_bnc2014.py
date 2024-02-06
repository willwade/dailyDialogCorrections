import csv
import random
import re

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

def process_csv_file(file_path):
    data_pairs = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            sentence = row[0] if row else ""
            # Lowercase everything
            sentence = sentence.lower()
            # Replace 'cos' with 'because'
            sentence = re.sub(r"\bcos\b", "because", sentence)
            # Remove ' quotes, 'mm', and double words
            sentence = re.sub(r"'", "", sentence)
            sentence = re.sub(r"\bmm\b", "", sentence, flags=re.IGNORECASE)
            sentence = re.sub(r"\b(\w+)\s+\1\b", r"\1", sentence)
            sentence = sentence.strip()
            
            words = sentence.split()
            if 1 < len(words) <= 5:  # Only take lines with 2 to 5 words
                typo_sentence = introduce_typos(''.join(words), typo_rate=0.1)  # Join without spaces to simulate missing spaces
                data_pairs.append({"input": typo_sentence, "target": sentence})
    return data_pairs


def save_to_csv(data_pairs, output_file='processed_bnc2014_typos.csv'):
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["input", "target"])
        writer.writeheader()
        for pair in data_pairs:
            writer.writerow(pair)

# Process the BNC2014 CSV and save the results
file_path = 'processed_bnc2014.csv'  # Update this path to your CSV file's location
data_pairs = process_csv_file(file_path)
save_to_csv(data_pairs, 'processed_bnc2014_typos.csv')

print(f"Processed data with typos saved to CSV.")
