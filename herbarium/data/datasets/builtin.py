# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from herbarium.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import _get_builtin_metadata
from .herb import register_herb_instances
from .cub import register_cub_instances
from pathlib import Path

# ==== Predefined datasets and splits for Herbarium ==========

_PREDEFINED_SPLITS_HERB = {}
_PREDEFINED_SPLITS_HERB["herb"] = {
    "herb_2021_train": ("herb/2021/train", "metadata.json", "train_annotations.json"),
    "herb_2021_val": ("herb/2021/train", "metadata.json", "val_annotations.json"),
    "herb_2021_test": ("herb/2021/test", "metadata.json", "annotations.json"),
}

_PREDEFINED_SPLITS_CUB = {}
_PREDEFINED_SPLITS_CUB["cub"] = {
    "cub_2011_train": ("cub/", "images.txt", "classes.txt", "image_class_labels.txt", "train_test_split.txt"),
    "cub_2011_test": ("cub/", "images.txt", "classes.txt", "image_class_labels.txt", "train_test_split.txt")
}

def register_all_herbarium(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_HERB.items():
        for key, (dataset_root, metadata_file, annotation_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            dataset_root = os.path.join(root, dataset_root)
            images_root = dataset_root
            metadata_file =  os.path.join(Path(dataset_root).parent, metadata_file)
            annotation_file = os.path.join(dataset_root, annotation_file)
            register_herb_instances(
                key,
                _get_builtin_metadata(dataset_name, metadata_file),
                annotation_file if "://" not in annotation_file else annotation_file,
                images_root,
            )

def register_all_cub(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_CUB.items():
        for key, (dataset_root, images_txt, classes_txt, image_class_txt, train_test_txt) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            dataset_root = os.path.join(root, dataset_root)
            images_root = os.path.join(dataset_root, "images")
            images_txt =  os.path.join(dataset_root, images_txt)
            classes_txt =  os.path.join(dataset_root, classes_txt)
            image_class_txt =  os.path.join(dataset_root, image_class_txt)
            train_test_txt =  os.path.join(dataset_root, train_test_txt)

            register_cub_instances(
                key,
                _get_builtin_metadata(dataset_name, classes_txt),
                images_root,
                [images_txt, classes_txt, image_class_txt, train_test_txt],
            )

# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("HERBARIUM_DATASETS", "datasets")
    register_all_herbarium(_root)
    register_all_cub(_root)
