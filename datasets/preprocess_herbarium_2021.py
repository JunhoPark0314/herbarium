"""
Generate meatadata json file for Herbarium dataset
and change original "metadata.json" to "annotaions.json"

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/" as this file exists.

Also we need to split train datasets to train and validation
"""

from collections import defaultdict
import os
import shutil
import json
from threading import Timer
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import itertools
import copy
import sys
from tqdm import tqdm
from fvcore.common.timer import Timer

DATASET_ROOT = "datasets"
from herbarium.data.datasets.builtin import _PREDEFINED_SPLITS_HERB

def create_metadata(annotation_json):
    with open(annotation_json) as annotation_file:
        dataset = json.load(annotation_file)
    
    category_list = dataset["categories"]

    order_map = set()
    family_map = set()
    species_map = dict()
    hierarchy = defaultdict(lambda: defaultdict(set))

    for cat in category_list:
        order_map.add(cat["order"])
        family_map.add(cat["family"])
        species_map[cat["name"]] = cat["id"]

    order_map = sorted(order_map)
    family_map = sorted(family_map)
    order_map = dict(zip(order_map, np.arange(len(order_map)).tolist() ))
    family_map = dict(zip(family_map, np.arange(len(family_map)).tolist() ))

    #order_family_hierarchy = np.zeros((len(order_map), len(family_map)), dtype=np.int32)
    #family_species_hierarchy = np.zeros((len(family_map), len(species_map)), dtype=np.int32)

    for cat in tqdm(category_list):
        order_id = order_map[cat["order"]]
        family_id = family_map[cat["family"]]
        species_id = species_map[cat["name"]]

        hierarchy[order_id][family_id].add(species_id)

        #order_family_hierarchy[order_id, family_id] = 1
        #family_species_hierarchy[family_id, species_id] = 1
    
    for order_id in list(hierarchy.keys()):
        for family_id in list(hierarchy[order_id].keys()):
            hierarchy[order_id][family_id] = list(hierarchy[order_id][family_id])

    ret = {
        "order_map": order_map,
        "family_map": family_map,
        "species_map": species_map,
        "hierarchy": hierarchy,
    }

    return ret

def create_train_val_split(annotation_json, meta, num_split):
    with open(annotation_json) as annotation_file:
        dataset = json.load(annotation_file)
    
    annotation = dataset["annotations"]
    annotation.sort(key=lambda x: x['id'])
    image_list = dataset["images"]
    image_list.sort(key=lambda x: x['id'])
    per_split_image = 25

    ann_split = defaultdict(list)
    image_split = defaultdict(list)

    annotation_per_id = defaultdict(list)
    for ann in annotation:
        annotation_per_id[ann["category_id"]].append(ann)
    
    class generator:
        def __init__(self, data):
            self.data = data
            np.random.shuffle(self.data)
            self.st = 0
            self.en = 0 

        def __call__(self, num):
            self.en = self.st + num
            curr_set = self.data[self.st:self.en]
            if len(curr_set) < num:
                self.st = 0
                num -= len(curr_set)
                self.en = num
                np.random.shuffle(self.data)
                curr_set = np.concatenate((curr_set, self.data[self.st:self.en]))
            return curr_set
                
    len_per_id = []
    for key, val in tqdm(annotation_per_id.items()):
        per_id_len = len(val)
        len_per_id.append(per_id_len)
        split = np.arange(per_id_len)
        split = generator(split) 

        for _ in range(num_split):

            batch = per_split_image
            if per_id_len // per_split_image < 1:
                batch = per_id_len
                if _ > 3:
                    batch = per_id_len // 4 + 1
                if _ > 7:
                    continue
            
            curr_ann = np.array(val)[split(batch)].tolist()

            ann_split[_].extend(curr_ann) 

            for ann in curr_ann:
                image_split[_].append(image_list[ann["image_id"]])


    class JsonProgress(object):
        def __init__(self):
            self.count = 0

        def __call__(self, obj):
            self.count += 1
            sys.stdout.write("\r%8d" % self.count)
            return obj


    for _ in range(num_split):
        annotation_json_path = os.path.join(Path(annotation_json).parent, "{}_annotations.json".format(_))
        if os.path.exists(annotation_json_path) is False:
            timer = Timer()
            print("Write train dataset")
            with open(annotation_json_path,"w") as train_ann_file:
                dataset["annotations"] = ann_split[_]
                dataset["images"] = image_split[_]
                json.dump(dataset, train_ann_file)
            print("{} seconds to write {} annotation file".format(timer.seconds(), _))

    return 

if __name__ == "__main__":
    num_split = 10

    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_HERB.items():
        for key, (dataset_root, metadata_file, annotation_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            images_root = os.path.join(DATASET_ROOT, dataset_root, "images")
            metadata_path =  os.path.join(DATASET_ROOT, dataset_root, metadata_file)
            annotation_path = os.path.join(DATASET_ROOT, dataset_root, "annotations.json")

            print("Processing {}".format(key))

            if not os.path.isfile(annotation_path):
                # Initially, dataset have annotation info in "metadata.json"
                # Change it to "annotations.json" and create metadata for dataset based on annotation file
                shutil.copy2(metadata_path, annotation_path)

            if "train" in key:
                # If this dataset is train dataset, create metadata file and split into validation

                new_metadata_path = os.path.join(DATASET_ROOT, Path(dataset_root).parent, metadata_file)


                if not os.path.isfile(new_metadata_path):
                    print("Train dataset! Create metadata from annotation file.")
                    metadata = create_metadata(annotation_path)
                    with open(new_metadata_path, "w") as new_metadata_file:
                        json.dump(metadata, new_metadata_file)
                else:
                    with open(new_metadata_path,"r") as new_metadata_file:
                        metadata = json.load(new_metadata_file)

                # TODO: split train to train and validation in here

                print("Split dataset into train and val")
                create_train_val_split(annotation_path, metadata, num_split)