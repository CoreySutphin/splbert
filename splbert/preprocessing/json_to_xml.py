"""
Script to convert predicted labels in JSON format to the XML format required for the TAC 2019 DDI Challenge.

Usage: python json_to_xml.py predictions_directory output_directory
"""
import sys
import os
import spacy
import json
import warnings
import xml.etree.ElementTree as ET
from spacy.gold import offsets_from_biluo_tags, docs_to_json

assert len(sys.argv) > 1

nlp = spacy.load("en_core_web_sm")

# Get the prediction directory path from command line arguments
prediction_directory = sys.argv[1]

output_directory = sys.argv[2]

# Iterate over the prediction files
for filename in os.listdir(prediction_directory):
    print(f"Processing file {filename}")
    with open(os.path.join(prediction_directory, filename), "r") as f:
        document = json.loads(f.read())

        root = ET.Element("Label")
        root.set("drug", document["drug_name"])

        # Create an empty child for the Text section
        ET.SubElement(root, "Text")

        sentences_elem = ET.SubElement(root, "Sentences")

        mention_counter = 0

        # Iterate over every sentence and write any predictions into a "Mention" element
        for i in range(len(document["labels"])):
            sentence = document["sentences"][i]
            labels = document["labels"][i]

            sentence_elem = ET.SubElement(sentences_elem, "Sentence")
            sentence_elem.set("id", sentence["id"])

            sentence_text = sentence["text"]
            sentence_text_elem = ET.SubElement(sentence_elem, "SentenceText")
            sentence_text_elem.text = sentence_text

            # print(sentence_text)

            # Create a SpaCy doc to convert the BILUO labels to the original labels
            doc = nlp(sentence_text)

            try:
                for (start, end, label) in offsets_from_biluo_tags(doc, labels):
                    if label not in ["O", "-"]:
                        mention_elem = ET.SubElement(sentence_elem, "Mention")
                        mention_elem.set("type", "Precipitant")
                        mention_elem.set("span", f"{start} {end - start}")
                        mention_elem.set("code", "N/A")
                        mention_elem.set("str", sentence_text[start:end])
                        mention_elem.set("id", f"M{mention_counter}")
                        mention_counter += 1
            except ValueError:
                print(sentence_text, labels)

        # Create an empty child for the "LabelInteractions" section
        ET.SubElement(root, "LabelInteractions")

        # ET.dump(root)
        tree = ET.ElementTree(root)
        tree.write(os.path.join(output_directory, document["file_name"]))
