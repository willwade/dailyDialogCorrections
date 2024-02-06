// http://cass.lancs.ac.uk/cass-projects/spoken-bnc2014/

import xml.etree.ElementTree as ET
import re
import csv
import glob

def clean_and_chunk_utterance(text):
    # Remove XML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Split based on pauses and ignore non-word elements
    chunks = [chunk for chunk in re.split(r"\s*<pause dur=\"[^\"]+\"/>\s*", text) if chunk.strip() and len(chunk.split()) > 1]
    return chunks

def parse_xml_to_phrases(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    phrases = []
    
    for utterance in root.findall(".//u"):
        text = ''.join(utterance.itertext())
        chunks = clean_and_chunk_utterance(text)
        phrases.extend(chunks)
    
    return phrases

# Specify the directory containing the XML files
xml_directory = "spoken/untagged/"

# Use glob to find all XML files in the directory
xml_files = glob.glob(xml_directory + "*.xml")

all_phrases = []
for xml_file in xml_files:
    all_phrases.extend(parse_xml_to_phrases(xml_file))

# Write phrases to a CSV file
with open('processed_bnc2014.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    for phrase in all_phrases:
        csvwriter.writerow([phrase])
