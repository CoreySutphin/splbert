#
#   Script to iterate over .ann files in a specified directory and count the number
#   of occurences of different labels.  Prints findings to console.
#

import sys
import os
import re

assert (
    len(sys.argv) > 1
), "Must pass in directory containing annotation files to command line"
directory = sys.argv[1]

# Change labels here to the labels in your data
label_dict = {"Precipitant": 0, "SpecificInteraction": 0, "Trigger": 0}


num_discontinuous_mentions = 0

for filename in os.listdir(directory):
    if filename.endswith(".ann"):
        with open(os.path.join(directory, filename), "r") as ann_file:
            print(filename + "==========================")
            annotations = re.split("T\d+\t", ann_file.read())
            for annotation in annotations:
                if annotation.strip() == "":
                    continue
                print(annotation)
                label = re.search("^(.*?)\s\d", annotation).group(1)
                if label in label_dict:
                    label_dict[label] += 1
                if re.search(r"(\d+);(\d+)", annotation):
                    print(annotation)
                    num_discontinuous_mentions += 1


print(label_dict)
print(num_discontinuous_mentions)
