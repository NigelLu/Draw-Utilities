"""module utils.py
"""
import os
import shutil

def validate_save_dir(save_dir: str) -> bool:
    """Validate save dir

    Validate save dir, and clear/ignore the content of the save directory based on user input

    Args:
        save_dir {str}: the save dir to be validated

    Returns:
        True/False indicating whether the process calling validate_save_dir function should proceed or not
    """
    # * double check for the validity of argument save_dir
    assert type(save_dir) == str, f"Expect save_dir argument to be a str, got {type(save_dir)} instead"

    if not os.path.isdir(save_dir): 
        os.makedirs(save_dir, exist_ok=False)
        return True

    if len(os.listdir(save_dir)) == 0:
        print(f">>> Save dir validtity check passed")
        return True
    
    print(f"Warning: the save dir '{save_dir}' you specified is NOT empty\nIt contains")

    dir_content = os.listdir(save_dir)
    dir_content.sort()
    print('\n'.join(dir_content))

    should_proceed = ''
    while should_proceed.lower() != 'yes' and should_proceed.lower() != 'no' and should_proceed.lower() != 'ignore':
        should_proceed = input(
            "Would you like us to empty the directory for you and proceed? (yes/no/ignore)\n> ").lower()
        if should_proceed.lower() == 'no':
            return False
        if should_proceed.lower() == 'yes':
            shutil.rmtree(save_dir, ignore_errors=True)
            os.mkdir(save_dir)
    print(f">>> Save dir validtity check passed")
    return True