"""module functions.utils.py
"""
import os
import shutil

from typing import Dict


def validate_save_dir(save_dir: str, image_to_mask_map: Dict[str, str], split: int) -> bool:
    """Validate save dir

    Validate save dir, and clear/ignore the content of the save directory based on user input

    Args:
        save_dir {str}: the save dir to be validated
        image_to_mask_map {dict}: a python dictionary specifying an image to mask key mapping (corresponding to the keys in draw_info_dict), 
                            as following
                            {
                                <image_key>: <mask_key>,
                            }
        split {int}: the split of the dataset

    Returns:
        True/False indicating whether the process calling validate_save_dir function should proceed or not
    """
    # * double check for the validity of arguments
    assert type(
        split) == int, f"Expect split argument to be an int, got {type(split)} instead"
    assert type(
        save_dir) == str, f"Expect save_dir argument to be a str, got {type(save_dir)} instead"
    assert type(
        image_to_mask_map) == dict, f"Expect image_to_mask_map argument to be a dict, got {type(image_to_mask_map)} instead"
    assert os.path.isdir(
        save_dir), f"Expect save_dir to be a direcory, {save_dir} is invalid"

    split = str(split)
    target_dirs = []
    for image_key in image_to_mask_map.keys():
        target_dir = os.path.join(save_dir, split, image_key)
        target_dir_label = os.path.join(save_dir, split, f"{image_key}-label")
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(target_dir_label, exist_ok=True)
        target_dirs.append(target_dir)
        target_dirs.append(target_dir_label)

    for target_dir in target_dirs:
        if len(os.listdir(target_dir)) == 0:
            continue
        print(
            f"Warning: the target dir '{target_dir}' is NOT empty\nIt contains (we will at maximum show 5 items)")

        dir_content = os.listdir(target_dir)
        dir_content.sort()
        print('\n'.join(dir_content[:5]))

        should_proceed = ''
        while should_proceed.lower() != 'yes' and should_proceed.lower() != 'no' and should_proceed.lower() != 'ignore':
            should_proceed = input(
                "Would you like us to empty the directory for you and proceed? (yes/no/ignore)\n> ").lower()
            if should_proceed.lower() == 'no':
                return False
            if should_proceed.lower() == 'yes':
                shutil.rmtree(target_dir, ignore_errors=True)
                os.makedirs(target_dir, exist_ok=True)

    print(f">>> Save dir validtity check passed")
    return True
