"""
version 20240601
"""

from datasets import load_dataset
import os
import json
from tqdm import tqdm

### settings of this run
# Settings
start_idx = 0
original_max_idx = 200000
partition = 10000
skip_equal_pair = False

### data dir
cache_dir = "/home/rliuay/haoyu/dataset/picapic_v2/"
dataset_name = "yuvalkirstain/pickapic_v2"
dataset = load_dataset(path=dataset_name, cache_dir=cache_dir)
# print("picapic v2 has ",dataset)

dataset_splits = [
    "train",
    "validation",
    "test",
    "test_unique",
    "validation_unique",
]  # 959040  20596 20716 500 500 sample repectively

# subset = dataset[dataset_splits[2]]
# print("subset has ",len(subset))

# the dataset directly generated to the target folder to save copying data time
root_path = "/home/rliuay/haoyu/dataset/VPD-full"  # VPD-{num_origin_sample}-{With or No equal pair}-{piece number}
os.makedirs(root_path, exist_ok=True)


# pwd = os.getcwd()
# json_path = os.path.join(pwd, root_path, "metadata.json")
# text_path = os.path.join(pwd, root_path, "caption.txt")
# WINFOLDER = os.path.join(pwd, root_path, "win")
# LOSFOLDER = os.path.join(pwd, root_path, "lose")

# os.makedirs(os.path.dirname(json_path), exist_ok=True)
# os.makedirs(WINFOLDER, exist_ok=True)
# os.makedirs(LOSFOLDER, exist_ok=True)

# if os.path.exists(text_path):
# os.remove(text_path)


# def decode_image(im_bytes):
# image = Image.open(io.BytesIO(im_bytes)).convert("RGB")
# return image


def dump_metadata(metadata_list, json_path, to_dict=True):
    if to_dict:
        metadata = {"dataset": metadata_list}
    else:
        metadata = metadata_list
    # Write metadata to JSON file
    with open(json_path, "w") as f:
        json.dump(metadata, f)


for dataset_split in dataset_splits:
    subset = dataset[dataset_split]
    print(
        f"subset {dataset_split} has {len(subset)} examples,\n    metadata generated at the partition of {partition}"
    )
    start_p = 0
    datadir = os.path.join(root_path, dataset_split, f"dataset_{start_p}")
    os.makedirs(datadir, exist_ok=True)
    json_path = os.path.join(datadir, "metadata.json")
    # continue
    metadata_list = []
    for idx, example in enumerate(tqdm(subset[start_idx:original_max_idx], desc=f"Processing {dataset_split} examples")):
        if idx % partition == 0 and idx > 0:
            dump_metadata(metadata_list, json_path)
            start_p += 1
            datadir = os.path.join(root_path, dataset_split, f"dataset_{start_p}")
            os.makedirs(datadir, exist_ok=True)
            json_path = os.path.join(datadir, "metadata.json")
            metadata_list = []
        caption = example["caption"]
        if example["label_0"] == 0.5 and skip_equal_pair:
            continue
        metadata_instance = {
            "index": idx,
            "lable_0": example["label_0"],
            "caption": caption,
            "wvideopath": "",
            "lvideopath": "",
        }
        metadata_list.append(metadata_instance)
    dump_metadata(metadata_list, json_path)

        ## Determine win/lose images based on label_0
        # if example['label_0'] != 0:
        #     win_image_path = os.path.join(WINFOLDER, f"{idx:06}.jpg")
        #     lose_image_path = os.path.join(LOSFOLDER, f"{idx:06}.jpg")
        # else:
        #     win_image_path = os.path.join(WINFOLDER, f"{idx:06}.jpg")
        #     lose_image_path = os.path.join(LOSFOLDER, f"{idx:06}.jpg")

        ## Save images
        # if os.path.exists(win_image_path)==False:
        #     jpg_0 = decode_image(example['jpg_0'])
        #     jpg_0.save(win_image_path)
        # if os.path.exists(lose_image_path)==False:
        #     jpg_1 = decode_image(example['jpg_1'])
        #     jpg_1.save(lose_image_path)

        # Append instance to metadata list
        # metadata_instance = {
        #     "index": idx,
        #     "lable_0": example["label_0"],
        #     "caption": caption,
        #     "wvideopath": "",
        #     "lvideopath": "",
        # }
        # metadata_list.append(metadata_instance)

    # print(len(os.listdir(WINFOLDER)),len(os.listdir(LOSFOLDER)))
    # dump_metadata(metadata_list, json_path)
