"""
Split a dataset of files into a training and validation set.
Expects both the training and validation directories to already be created.
"""

import os
import sys
import random
from shutil import copy2


source_directory = sys.argv[1]
split_directory = sys.argv[2]

filenames = os.listdir(source_directory)
threshold = len(filenames) * 0.15
counter = 0

random.shuffle(filenames)
for filename in filenames:
    if filename.endswith(".ann"):
        ann_path = os.path.join(source_directory, filename)
        text_path = os.path.join(source_directory, filename.replace(".ann", ".txt"))

        if counter < threshold:
            copy2(ann_path, os.path.join(split_directory, "validation"))
            copy2(text_path, os.path.join(split_directory, "validation"))
        else:
            copy2(ann_path, os.path.join(split_directory, "training"))
            copy2(text_path, os.path.join(split_directory, "training"))

        counter += 1
    elif filename.endswith(".xml"):
        xml_path = os.path.join(source_directory, filename)

        if counter < threshold:
            copy2(xml_path, os.path.join(split_directory, "validation"))
        else:
            copy2(xml_path, os.path.join(split_directory, "training"))

        counter += 1
