"""module convert.py
"""
import torch

from einops import rearrange


def detach(draw_info_dict: dict, keys_to_process: list) -> None:
    """Detach tensors in draw_info_dict

    Detach the tensors (in place) in draw_info_dict based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed

    Returns:
        None
    """
    # * double check for argument type
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start detaching
    print(f">>> Start detaching for {keys_to_process}...")
    try:
        for key in keys_to_process:
            draw_info_dict[key] = draw_info_dict[key].detach()
        print(">>> Detaching finished")
    except:
        print(
            "Something went wrong and we failed to detach the corresponding draw info dict")
        print(">>> detach function exited with status code 1")
        return


def reshape(draw_info_dict: dict, keys_to_process: list, rearrange_str: str, in_place: bool = True) -> dict:
    """Reshape the image tensors in draw_info_dict

    Convert the tensors to torch.float32 in draw_info_dict based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed
        rearrange_str {str}: a string in the form of "c h w -> h w c", for rearranging the image tensor using einops.rearrange function
        in_place {bool}: True/False argument, specifying whether to convert the tensors in an in-place manner

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
        rearrange_str) == str, f"Expect rearrange_str argument to be a str, got {type(rearrange_str)} instead"
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start converting to float 32
    print(f">>> Start reshaping {keys_to_process} using '{rearrange_str}'...")
    try:
        # * create a new info dict for not in_place conversion
        new_draw_info_dict = dict([(key, None)
                                  for key in keys_to_process])

        for key in keys_to_process:
            if not in_place:
                new_draw_info_dict[key] = rearrange(
                    draw_info_dict[key].clone(), rearrange_str)
            if in_place:
                draw_info_dict[key] = rearrange_str(
                    draw_info_dict[key], rearrange_str)

        print(">>> Reshaping finished")
        return new_draw_info_dict
    except:
        print(
            "Something went wrong and we failed to reshape the corresponding draw info dict")
        print(">>> reshape function exited with status code 1")
        return


def to_float32(draw_info_dict: dict, keys_to_process: list, in_place: bool = True) -> dict:
    """Convert tensors to torch.float32 in draw_info_dict

    Convert the tensors to torch.float32 in draw_info_dict based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed
        in_place {bool}: True/False argument, specifying whether to convert the tensors in an in-place manner

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
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start converting to float 32
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

        print(">>> Converting to float32 finished")
        return new_draw_info_dict
    except:
        print(
            "Something went wrong and we failed to convert the corresponding draw info dict")
        print(">>> to_float32 function exited with status code 1")
        return


def denormalize(draw_info_dict: dict, keys_to_process: list, MEAN: list, STD: list, in_place: bool = True) -> dict:
    """Denormalize the image tensors in draw_info_dict

    Denormalize the image tensors in draw_info_dict based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed
        MEAN {list}: a python list with three float values, specifying the means used in denormalizing
        STD {list}: a python list with three float values, specifying the standard deviations used in denormalizing
        in_place {bool}: True/False argument, specifying whether to denormalize the tensors in an in-place manner

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
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start denormalizing image tensors
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
                    t.mul_(s).add_(m).mul(255)

        print(">>> Denormalizing finished")
        return new_draw_info_dict
    except:
        print(
            "Something went wrong and we failed to denormalize the corresponding draw info dict")
        print(">>> denormalize function exited with status code 1")
        return


def tocpu_and_asNumpy(draw_info_dict: dict, keys_to_process: list, in_place: bool = True) -> dict:
    """Send tensors to CPU and convert them from torch.Tensor to Numpy

    Send the tensors in draw_info_dict to CPU and convert them from torch.Tensor to Numpy based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed
        in_place {bool}: True/False argument, specifying whether to convert the tensors in an in-place manner

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
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start to CPU and convert to numpy
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

        print(">>> Sending to CPU and transform to Numpy finished")
        return new_draw_info_dict
    except:
        print(
            "Something went wrong and we failed to convert the corresponding draw info dict")
        print(">>> tocpu_and_asNumpy function exited with status code 1")
        return
