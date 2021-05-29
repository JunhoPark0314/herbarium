# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import itertools
import json
import logging
import pickle
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
import multiprocessing as mp
from itertools import product

from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image
from tqdm import tqdm
import torch

#from herbarium.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from herbarium.utils.file_io import PathManager

from .. import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse Herb-format annotations into dicts in "Herbarium format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_cub_json", "convert_to_cub_json", "register_cub_instances"]

def process_per_record(img_ann, image_root, ann_keys, meta):
    #print("Processor {} start".format(worker_id))
    img_dict, anno_dict = img_ann
    record = {}
    record["file_name"] = os.path.join(image_root, img_dict["file_name"])
    img_file = Image.open(record["file_name"])
    record["width"] = img_file.size[0]
    record["height"] = img_file.size[1]
    image_id = record["image_id"] = img_dict["id"]

    assert anno_dict["image_id"] == image_id

    obj = {key: anno_dict[key] for key in ann_keys if key in anno_dict}
    # TODO: change class_id into hierarchy id here
    if meta is not None:
        obj["species_id"] = anno_dict["category_id"]

    record["annotations"] = [obj]

    return record

class anns_generator:
    def __init__(self, anns, w_id, num_workers):
        self.anns = anns
        self.w_id = w_id
        self.num_workers = num_workers

    def __iter__(self):
        for ann in itertools.islice(self.anns, self.w_id, None, self.num_workers):
            yield ann 

def update_meta(json_file, dataset_name=None):

    from pyherbtools.herb import HERB

    classes_txt = open(json_file[1],'r').readlines()

    if dataset_name is not None:

        meta = MetadataCatalog.get(dataset_name)

        logger.info("Creating hierarchy target from given annotation")

        family_species_hierarchy = torch.rand(meta.num_classes["species"],meta.num_classes["family"])
        
        from torch import nn
        family_species_hierarchy = nn.Softmax(dim=1)(family_species_hierarchy)
        meta.hierarchy_prior = {
            "family|species": family_species_hierarchy
        }
        meta.thing_classes = [cl.split()[1] for cl in classes_txt]


def load_cub_json(ann_files, image_root, dataset_name=None):
    images_txt, classes_txt, image_class_txt, train_test_split_txt = ann_files

    split = 0
    if 'test' in dataset_name:
        split = 1

    images_txt = open(images_txt, 'r').readlines()
    image_class_txt = open(image_class_txt, 'r').readlines()
    classes_txt = open(classes_txt, 'r').readlines()
    train_test_split_txt = open(train_test_split_txt, 'r').readlines()

    imgs = []
    anns = []
    classes = {}

    for i in range(len(train_test_split_txt)):
        image_id, curr_split = train_test_split_txt[i].split()
        if int(curr_split) == split:
            _, image_path = images_txt[i].split()
            _, class_id = image_class_txt[i].split()
            _, class_name = classes_txt[int(class_id) - 1].split()

            curr_image = {"id": int(image_id), "file_name": image_path}
            curr_ann = {"id": i, "category_id":int(class_id) - 1, "image_id":int(image_id)}

            imgs.append(curr_image)
            anns.append(curr_ann)
            classes[int(class_id) - 1] = class_name

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in CUB format from CUB-200".format(len(imgs_anns)))

    dataset_dicts = []

    ann_keys = ["category_id"]

    logger.info("Convert CUB format into herbarium format")

    timer = Timer()

    meta = MetadataCatalog.get(dataset_name)
    dataset_dicts = [process_per_record(anns, image_root, ann_keys, meta) for anns in imgs_anns]

    logger.info("Processing Record takes {:.2f} seconds.".format(timer.seconds()))

    return dataset_dicts


# TODO: Change here to fit on herbarium dataset
# Need at evaluation stage

def convert_to_herb_dict(dataset_name):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in herbarium's standard format into COCO json format.

    Generic dataset description can be found here:
    https://herbarium.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in herbarium's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.dataset_id_to_hierarchy_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                if isinstance(segmentation, list):
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                elif isinstance(segmentation, dict):  # RLE
                    area = mask_util.area(segmentation).item()
                else:
                    raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
            else:
                # Computing areas using bounding boxes
                if to_bbox_mode == BoxMode.XYWH_ABS:
                    bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()
                else:
                    area = RotatedBoxes([bbox]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/herbarium/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
            coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                seg = coco_annotation["segmentation"] = annotation["segmentation"]
                if isinstance(seg, dict):  # RLE
                    counts = seg["counts"]
                    if not isinstance(counts, str):
                        # make it json-serializable
                        seg["counts"] = counts.decode("ascii")

            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def convert_to_herb_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in herbarium's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in herbarium's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to HERB format ...)")
            coco_dict = convert_to_herb_dict(dataset_name)

            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)


def register_cub_instances(name, metadata, image_root, ann_files):
    """
    Register a dataset in Herbarium's json annotation format for classification.

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "herb_2021_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_cub_json(ann_files, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging


    if metadata is not None:
        MetadataCatalog.get(name).set(
            image_root=image_root, evaluator_type="herb", **metadata
        )

        update_meta(ann_files, name)


if __name__ == "__main__":
    """
    Test the Herbarium json dataset loader.

    Usage:
        python -m herbarium.data.datasets.herb \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "herb_2021_val", or other
        pre-registered ones
    """
    from herbarium.utils.logger import setup_logger
    import herbarium.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_herb_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "herb-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        # TODO: do something for visualize in this "herb-data-vis" and implement in util.visualizer