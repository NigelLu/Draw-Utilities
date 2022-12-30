""" module functions.convert.py
   
    Summary:

        This module contains utility conversion functions often used in drawing

    Function Provided:

        - detach(draw_info_dict: Dict[str, torch.Tensor], keys_to_process: List[str], should_log: bool = False) -> None

        - denormalize(draw_info_dict: Dict[str, torch.Tensor], keys_to_process: List[str], MEAN: List[float], STD: List[float], in_place: bool = True, should_log: bool = False) -> dict

        - reshape(draw_info_dict: Dict[str, torch.Tensor], keys_to_process: List[str], rearrange_str: str, in_place: bool = True, should_log: bool = False) -> dict

        - logit_to_mask(draw_info_dict: Dict[str, torch.Tensor], masks_to_translate: List[str], fg_dim: int = 1, in_place: bool = True, should_log: bool = False) -> Dict[str, torch.Tensor]

        - combine(draw_info_dict: Dict[str, torch.Tensor], images_and_masks_to_combine: Dict[str, List[str]], img_save_key: str = "img_combined", mask_save_key: str = "mask_combined", should_log: bool = False) -> None

        - to_float32(draw_info_dict: Dict[str, torch.Tensor], keys_to_process: List[str], in_place: bool = True, should_log: bool = False) -> dict

        - tocpu_and_asNumpy(draw_info_dict: Dict[str, torch.Tensor], keys_to_process: List[str], in_place: bool = True, should_log: bool = False) -> dict
"""
import torch

from einops import rearrange
from typing import Dict, List


def detach(draw_info_dict: Dict[str, torch.Tensor], keys_to_process: List[str], should_log: bool = False) -> None:
    """ Detach tensors in draw_info_dict

    Detach the tensors (in place) in draw_info_dict based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed
        should_log {bool}: whether to print logs

    Returns:
        None
    """
    # * double check for argument type
    assert type(
        should_log) == bool, f"Expect should_log argument to be a boolean, got {type(should_log)} instead"
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start detaching
    if should_log:
        print(f">>> Start detaching for {keys_to_process}...")
    try:
        for key in keys_to_process:
            draw_info_dict[key] = draw_info_dict[key].detach()
        if should_log:
            print(">>> Detaching finished")
    except:
        print(
            "Something went wrong and we failed to detach the corresponding draw info dict")
        print(">>> detach function exited with status code 1")
        return


def denormalize(draw_info_dict: Dict[str, torch.Tensor], keys_to_process: List[str], MEAN: List[float], STD: List[float], in_place: bool = True, should_log: bool = False) -> dict:
    """ Denormalize the image tensors in draw_info_dict

    Denormalize the image tensors in draw_info_dict based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed
        MEAN {list}: a python list with three float values, specifying the means used in denormalizing
        STD {list}: a python list with three float values, specifying the standard deviations used in denormalizing
        in_place {bool}: True/False argument, specifying whether to denormalize the tensors in an in-place manner
        should_log {bool}: whether to print logs


    Returns:
        A dict with the keys in keys_to_process, with values being
            None (if in_place is True)
                OR
            torch.Tensor (if in_place is False), these tensors are copies of the tensors in draw_info_dict, which are then denormalized
    """
    # * double check for argument type
    assert type(
        STD) == list, f"Expect STD argument to be a list, got {type(STD)} instead"
    assert type(
        MEAN) == list, f"Expect MEAN argument to be a list, got {type(MEAN)} instead"
    assert type(
        in_place) == bool, f"Expect keys_to_process argument to be a bool, got {type(in_place)} instead"
    assert type(
        should_log) == bool, f"Expect should_log argument to be a boolean, got {type(should_log)} instead"
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start denormalizing image tensors
    if should_log:
        print(
            f">>> Start denormalizing {keys_to_process} using MEAN: {MEAN}, STD: {STD}...")
    try:
        # * create a new info dict for not in_place conversion
        new_draw_info_dict = dict([(key, None)
                                  for key in keys_to_process])

        for key in keys_to_process:
            if not in_place:
                img = draw_info_dict[key].clone()
                for t, m, s in zip(img, MEAN, STD):
                    t.mul_(s).add_(m).mul(255)
                new_draw_info_dict[key] = img
            if in_place:
                for t, m, s in zip(draw_info_dict[key], MEAN, STD):
                    t.mul_(s).add_(m).mul_(255)

        if should_log:
            print(">>> Denormalizing finished")
        return new_draw_info_dict
    except:
        print(
            "Something went wrong and we failed to denormalize the corresponding draw info dict")
        print(">>> denormalize function exited with status code 1")
        return


def reshape(draw_info_dict: Dict[str, torch.Tensor], keys_to_process: List[str], rearrange_str: str, in_place: bool = True, should_log: bool = False) -> dict:
    """ Reshape the image tensors in draw_info_dict

    Restore RGB Channel sequence and reshape the image tensors in draw_info_dict based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed
        rearrange_str {str}: a string in the form of "c h w -> h w c", for rearranging the image tensor using einops.rearrange function
        in_place {bool}: True/False argument, specifying whether to convert the tensors in an in-place manner
        should_log {bool}: whether to print logs

    Returns:
        A dict with the keys in keys_to_process, with values being
            None (if in_place is True)
                OR
            torch.Tensor (if in_place is False), these tensors of type torch.float32 and are copies of tensors in draw_info_dict
    """
    # * double check for argument type
    assert type(
        in_place) == bool, f"Expect keys_to_process argument to be a bool, got {type(in_place)} instead"
    assert type(
        should_log) == bool, f"Expect should_log argument to be a boolean, got {type(should_log)} instead"
    assert type(
        rearrange_str) == str, f"Expect rearrange_str argument to be a str, got {type(rearrange_str)} instead"
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start converting to float 32
    if should_log:
        print(
            f">>> Start reshaping {keys_to_process} using '{rearrange_str}'...")
    try:
        # * create a new info dict for not in_place conversion
        new_draw_info_dict = dict([(key, None)
                                  for key in keys_to_process])

        for key in keys_to_process:
            if not in_place:
                img = draw_info_dict[key].clone()
                if len(img.shape) == 4:
                    img = img.squeeze()
                if img.shape[0] == 3:
                    img = torch.cat(
                        [img[2].unsqueeze(0), img[1].unsqueeze(0), img[0].unsqueeze(0)], dim=0)
                new_draw_info_dict[key] = rearrange(
                    img, rearrange_str)

            if in_place:
                if len(draw_info_dict[key].shape) == 4:
                    draw_info_dict[key] = draw_info_dict[key].squeeze()
                if draw_info_dict[key].shape[0] == 3:
                    draw_info_dict[key] = torch.cat([draw_info_dict[key][2].unsqueeze(
                        0), draw_info_dict[key][1].unsqueeze(0), draw_info_dict[key][0].unsqueeze(0)], dim=0)
                draw_info_dict[key] = rearrange(
                    draw_info_dict[key], rearrange_str)

        if should_log:
            print(">>> Reshaping finished")
        return new_draw_info_dict
    except:
        print(
            "Something went wrong and we failed to reshape the corresponding draw info dict")
        print(">>> reshape function exited with status code 1")
        return


def logit_to_mask(draw_info_dict: Dict[str, torch.Tensor], masks_to_translate: List[str], fg_dim: int = 1, in_place: bool = True, should_log: bool = False) -> Dict[str, torch.Tensor]:
    """ Translate logits into binary masks

    Translate logits into binary masks by first applying softmax then getting indices with torch.max

    NOTE: this function assumes that the masks are of shape (h, w, num_classes)

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        masks_to_translate {List[str]}: a list of strings specifying the keys to the masks that we want to translate into binary mask
        fg_dim {int}: an integer specifying which channel corresponds to the foreground, default to 1 (i.e., the 2nd channel)
        in_place {bool}: True/False argument, specifying whether to convert the tensors in an in-place manner
        should_log {bool}: whether to print logs

    Returns:
        A dict with keys being the keys specified in masks_to_translate, and values being the result binary masks of type torch.Tensor
    """
    # * double check for argument validity
    assert type(
        fg_dim) == int, f"Expect should_log argument to be an int, got {type(fg_dim)} instead"
    assert type(
        should_log) == bool, f"Expect should_log argument to be a boolean, got {type(should_log)} instead"
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        in_place) == bool, f"Expect images_and_masks_to_combine argument to be a bool, got {type(in_place)} instead"
    assert type(
        masks_to_translate) == list, f"Expect img_save_key argument to be a list, got {type(masks_to_translate)} instead"

    try:
        if should_log:
            print(f">>> Start translating logits for {masks_to_translate}...")

        # * init a dict to store the translated logits
        translated_binary_mask_dict = dict(
            [(key, None) for key in masks_to_translate])

        for mask_key in masks_to_translate:
            # * mask -> probability (h, w, num_classes)
            probability = torch.softmax(draw_info_dict[mask_key], dim=2)

            # * probability -> target_class_prob (h, w, 1)
            fg_probability = probability[:, :, fg_dim].unsqueeze(2)

            # * compute the background probability 
            # * by summing up all probability channels but the fg_dim
            bg_probability = torch.zeros_like(fg_probability)
            for channel in range(probability.shape[2]):
                if channel == fg_dim:
                    continue
                bg_probability += probability[:, :, channel].unsqueeze(2)
            
            # * get the binary foreground mask by using torch.max 
            _, translated_mask = torch.max(
                torch.cat([bg_probability, fg_probability], dim=2), dim=2)
            translated_mask = translated_mask.unsqueeze(2)
            if in_place:
                draw_info_dict[mask_key] = translated_mask
                continue

            translated_binary_mask_dict[mask_key] = translated_mask
        if should_log:
            print(">>> Translating logits completed")
        return translated_binary_mask_dict

    except:
        print(
            "Something went wrong and we failed to translate the the logits into binary masks")
        print(">>> logit_to_mask function exited with status code 1")
        return


def combine(draw_info_dict: Dict[str, torch.Tensor], images_and_masks_to_combine: Dict[str, List[str]], img_save_key: str = "img_combined", mask_save_key: str = "mask_combined", should_log: bool = False) -> None:
    """ Combine images along with their masks horizontally 

    Combine the images and masks specified by images_and_masks_to_combine horizontally 
        and save them back to the original draw_info_dict

    NOTE: 
        expects images and masks to be in shapes like (h, w, c)

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        images_and_masks_to_combine {Dict[str, List[str]]}: a python dictionary specifying which images & masks to combine
            key "images" corresponds to a list of keys to the images (order-sensitive)
            key "masks" corresponds to a list of keys to the masks (order-sensitive)
        img_save_key {str}: the key used to save the combined image back to draw_info_dict
        mask_save_key {str}: the key used to save the combined mask back to draw_info_dict
        should_log {bool}: whether to print logs

    Returns:
        None
    """
    # * double check for argument validity
    assert type(
        should_log) == bool, f"Expect should_log argument to be a boolean, got {type(should_log)} instead"
    assert type(
        img_save_key) == str, f"Expect img_save_key argument to be a string, got {type(img_save_key)} instead"
    assert type(
        mask_save_key) == str, f"Expect mask_save_key argument to be a string, got {type(mask_save_key)} instead"
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        images_and_masks_to_combine) == dict, f"Expect images_and_masks_to_combine argument to be a dict, got {type(images_and_masks_to_combine)} instead"
    assert len(images_and_masks_to_combine) == 2 and \
        "images" in images_and_masks_to_combine.keys() and \
        "masks" in images_and_masks_to_combine.keys() and \
        type(images_and_masks_to_combine["images"]) == list and \
        type(images_and_masks_to_combine["masks"]) == list, \
        f"Expect images_and_masks_to_combine to be a dict with keys 'images', 'masks' and values being two List[str], got {type(images_and_masks_to_combine)}\n{images_and_masks_to_combine}"

    try:
        # * start converting to float 32
        if should_log:
            print(f">>> Start combining {images_and_masks_to_combine}...")
        num_to_combine = len(images_and_masks_to_combine['images'])

        img_list = []
        mask_list = []
        for idx in range(num_to_combine):
            img_list.append(
                draw_info_dict[images_and_masks_to_combine['images'][idx]])
            mask_list.append(
                draw_info_dict[images_and_masks_to_combine['masks'][idx]])
        draw_info_dict[img_save_key] = torch.cat(img_list, dim=1)
        draw_info_dict[mask_save_key] = torch.cat(mask_list, dim=1)

        if should_log:
            print(f">>> Combining {images_and_masks_to_combine} completed")
        return

    except:
        print(
            "Something went wrong and we failed to combine the the images and masks")
        print(">>> combine function exited with status code 1")
        return


def to_float32(draw_info_dict: Dict[str, torch.Tensor], keys_to_process: List[str], in_place: bool = True, should_log: bool = False) -> dict:
    """ Convert tensors to torch.float32 in draw_info_dict

    Convert the tensors to torch.float32 in draw_info_dict based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed
        in_place {bool}: True/False argument, specifying whether to convert the tensors in an in-place manner
        should_log {bool}: whether to print logs

    Returns:
        A dict with the keys in keys_to_process, with values being
            None (if in_place is True)
                OR
            torch.Tensor (if in_place is False), these tensors are of type torch.float32 and are copies of tensors in draw_info_dict
    """
    # * double check for argument type
    assert type(
        in_place) == bool, f"Expect keys_to_process argument to be a bool, got {type(in_place)} instead"
    assert type(
        should_log) == bool, f"Expect should_log argument to be a boolean, got {type(should_log)} instead"
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start converting to float 32
    if should_log:
        print(f">>> Start converting {keys_to_process} to float32...")
    try:
        # * create a new info dict for not in_place conversion
        new_draw_info_dict = dict([(key, None)
                                  for key in keys_to_process])

        for key in keys_to_process:
            if not in_place:
                new_draw_info_dict[key] = draw_info_dict[key].clone().type(
                    torch.float32)
            if in_place:
                draw_info_dict[key] = draw_info_dict[key].type(torch.float32)

        if should_log:
            print(">>> Converting to float32 finished")
        return new_draw_info_dict
    except:
        print(
            "Something went wrong and we failed to convert the corresponding draw info dict")
        print(">>> to_float32 function exited with status code 1")
        return


def tocpu_and_asNumpy(draw_info_dict: Dict[str, torch.Tensor], keys_to_process: List[str], in_place: bool = True, should_log: bool = False) -> dict:
    """ Send tensors to CPU and convert them from torch.Tensor to Numpy

    Send the tensors in draw_info_dict to CPU and convert them from torch.Tensor to Numpy based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed
        in_place {bool}: True/False argument, specifying whether to convert the tensors in an in-place manner
        should_log {bool}: whether to print logs

    Returns:
        A dict with the keys in keys_to_process, with values being
            None (if in_place is True)
                OR
            torch.Tensor (if in_place is False), these tensors are copies of tensors in draw_info_dict,
                which got sent to CPU and converted to Numpy
    """
    # * double check for argument type
    assert type(
        in_place) == bool, f"Expect keys_to_process argument to be a bool, got {type(in_place)} instead"
    assert type(
        should_log) == bool, f"Expect should_log argument to be a boolean, got {type(should_log)} instead"
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start to CPU and convert to numpy
    if should_log:
        print(
            f">>> Start sending {keys_to_process} to CPU and transform to Numpy...")
    try:
        # * create a new info dict for not in_place conversion
        new_draw_info_dict = dict([(key, None)
                                  for key in keys_to_process])

        for key in keys_to_process:
            if not in_place:
                new_draw_info_dict[key] = draw_info_dict[key].clone(
                ).cpu().numpy()
            if in_place:
                draw_info_dict[key] = draw_info_dict[key].cpu().numpy()

        if should_log:
            print(">>> Sending to CPU and transform to Numpy finished")
        return new_draw_info_dict
    except:
        print(
            "Something went wrong and we failed to convert the corresponding draw info dict")
        print(">>> tocpu_and_asNumpy function exited with status code 1")
        return
