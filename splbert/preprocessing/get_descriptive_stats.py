"""
Given a directory of XML files containing drug labels, calculate descriptive statistics regarding the # and type of both
Mentions and Interactions. Prints the results to stdout.

Usage: python get_descriptive_stats.py xml_directory_path
"""
import sys
import os
import xml.etree.ElementTree as ET


def parse_file(file):
    """Given a path to an XML file, read it in and parse out descriptive stats.

    Returns:
        (dict) - Dict of results for this file.
    """
    num_mentions = {}
    num_interactions = {}
    non_contiguous_mentions = {}

    tree = ET.parse(file)
    root = tree.getroot()
    drug_name = root.attrib.get("drug")
    sentences = root.find("Sentences")

    for sentence in sentences.iter():
        mentions = sentence.findall("Mention")
        interactions = sentence.findall("Interaction")

        for mention in mentions:
            type = mention.attrib["type"]
            span = mention.attrib["span"]
            non_contiguous = ";" in span

            if type not in num_mentions:
                num_mentions[type] = 1
            else:
                num_mentions[type] += 1

            if non_contiguous:
                if type not in non_contiguous_mentions:
                    non_contiguous_mentions[type] = 1
                else:
                    non_contiguous_mentions[type] += 1

        for interaction in interactions:
            type = interaction.attrib["type"]
            trigger = interaction.attrib.get("trigger")
            precipitant = interaction.attrib.get("precipitant")
            effect = interaction.attrib.get("effect")

            if type not in num_interactions:
                num_interactions[type] = 1
            else:
                num_interactions[type] += 1

    return {
        "num_mentions": num_mentions,
        "num_interactions": num_interactions,
        "non_contiguous_mentions": non_contiguous_mentions,
    }


def main():
    assert (
        len(sys.argv) == 2
    ), "Usage: python get_descriptive_stats.py xml_directory_path"

    xml_dir = sys.argv[1]

    total_stats = {
        "num_mentions": {},
        "num_interactions": {},
        "non_contiguous_mentions": {},
    }

    for filename in os.listdir(xml_dir):
        document_stats = parse_file(os.path.join(xml_dir, filename))

        for key in document_stats["num_mentions"]:
            if key not in total_stats["num_mentions"]:
                total_stats["num_mentions"][key] = document_stats["num_mentions"][key]
            else:
                total_stats["num_mentions"][key] += document_stats["num_mentions"][key]

        for key in document_stats["num_interactions"]:
            if key not in total_stats["num_interactions"]:
                total_stats["num_interactions"][key] = document_stats[
                    "num_interactions"
                ][key]
            else:
                total_stats["num_interactions"][key] += document_stats[
                    "num_interactions"
                ][key]

        for key in document_stats["non_contiguous_mentions"]:
            if key not in total_stats["non_contiguous_mentions"]:
                total_stats["non_contiguous_mentions"][key] = document_stats[
                    "non_contiguous_mentions"
                ][key]
            else:
                total_stats["non_contiguous_mentions"][key] += document_stats[
                    "non_contiguous_mentions"
                ][key]

    print(total_stats)


if __name__ == "__main__":
    main()
