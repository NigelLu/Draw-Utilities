"""module convert.py
"""
import torch

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
            torch.Tensor (if in_place is False), these tensors of type torch.float32 and are copies of tensors in draw_info_dict
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
                new_draw_info_dict[key] = new_draw_info_dict[key].clone().type(
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
                new_draw_info_dict[key] = new_draw_info_dict[key].clone(
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
