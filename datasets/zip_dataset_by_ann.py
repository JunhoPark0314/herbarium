from zipfile import ZipFile as ZF
import os
from tqdm import tqdm
import json

DATASET_ROOT = "datasets"
from herbarium.data.datasets.builtin import _PREDEFINED_SPLITS_HERB

if __name__ == "__main__":
    num_split = 10

    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_HERB.items():
        for key, (dataset_root, metadata_path, annotation_path) in splits_per_dataset.items():
            
            contd = ("train" in key) or ("val" in key) or ("test" in key)

            if contd:
                continue

            # Assume pre-defined datasets live in `./datasets`.
            images_root = os.path.join(DATASET_ROOT, dataset_root)
            metadata_path =  os.path.join(DATASET_ROOT, dataset_root, metadata_path)
            annotation_path = os.path.join(DATASET_ROOT, dataset_root, annotation_path)
            zip_path = os.path.join("output", "zips", "{}.zip".format(key))

            print("{} dataset zip to {}".format(key,zip_path))

            curr_dataset_zip = ZF(zip_path,"w")

            with open(annotation_path,"r") as annotation_file:
                dataset_dict = json.load(annotation_file)
            image_list = dataset_dict["images"]
            
            for image in tqdm(image_list):
                file_name = os.path.join(images_root, image["file_name"])
                curr_dataset_zip.write(file_name)

            curr_dataset_zip.write(annotation_path)
            curr_dataset_zip.close()
