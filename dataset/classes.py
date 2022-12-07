"""module dataset.classes.py
"""
from typing import Dict, Any
from collections import defaultdict

classId2className = {
    "pascal": {
        1: "airplane",  # 0.14
        2: "bicycle",  # 0.07
        3: "bird",  # 0.13
        4: "boat",  # 0.12
        5: "bottle",  # 0.15
        6: "bus",  # 0.35
        7: "cat",  # 0.20
        8: "car",  # 0.26
        9: "chair",  # 0.10
        10: "cow",  # 0.24
        11: "diningtable",  # 0.22
        12: "dog",  # 0.23
        13: "horse",  # 0.21
        14: "motorcycle",  # 0.22
        15: "person",  # 0.20
        16: "pottedplant",  # 0.11
        17: "sheep",  # 0.19
        18: "sofa",  # 0.23
        19: "train",  # 0.27
        20: "tv",  # 0.14
    }
}

# * construct a mapping from className to classId
# * with root keys being datasetName
className2classId = defaultdict(dict)
for dataset in classId2className:
    for id, className in classId2className[dataset].items():
        className2classId[dataset][className] = id


def get_split_classes_pascal() -> Dict[str, Any]:
    """Returns the split of classes for PASCAL-5i
    Args:
        None

    Returns:
        A Dict specifying classes for different splits and set -> split_classes[<dataset_name>][<split_num>]['train'/'val'] = training classes in fold 0 of PASCAL-5i
    """
    split_classes = {"pascal": defaultdict(dict)}

    # * =============== Pascal ===================
    name = "pascal"
    class_list = list(range(1, 21))
    vals_lists = [
        list(range(1, 6)),
        list(range(6, 11)),
        list(range(11, 16)),
        list(range(16, 21)),
    ]

    # * idx -1 to retrieve all classes
    split_classes[name][-1]["val"] = class_list

    # * train classes is the complementer of val classes
    for i, val_list in enumerate(vals_lists):
        split_classes[name][i]["val"] = val_list
        split_classes[name][i]["train"] = list(set(class_list) - set(val_list))

    return split_classes
