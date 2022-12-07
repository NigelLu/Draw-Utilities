# The Drawing Tools for Semantic Segmentation

## Project Structure
- `config`
    - `__init__.py`
    - `config.py`
- `draw`
    - `__init__.py`
    - `convert.py`
    - `mask.py`
    - `utils.py`
- `dataset`
    - `lists`
        - `pascal`
            - `train.txt`
            - `val.txt`
    - `VOC2012`
    - `__init__.py`
    - `classes.py`
    - `dataset.py`
    - `utils.py`
    - `transform.py`
- `output`
    - `...`
- `pt-files`
    - `...`
- `.gitignore`
- `main.py`
- `LICENSE`
- `README.md`

---
## Preparation
- Install the required packages by running `python -m pip install -r requirements.txt` 
    - we recommend using `Python ^3.7.6` here
    - if you could not find the exact version of the packages listed in `requirements.txt`, it is acceptable to use a newer/older version

---
## Hands-on
- Download a pre-defined `.pt` file from this [drive link](https://drive.google.com/file/d/1ydF8tyVupIqYYsqoZGGYR3hWYr3_jykM/view?usp=sharing)
- Place the downloaded file under `pt-files/`
- Create folders in the following structure `<project_root>/output/task-definition`
- `cd` to the root folder of this project
- Run `python main.py` and see the results!

---
## Usage
1. Save the `.pt` files 
2. Modify the `main.py` based on the task
    - you may go through what utility functions offered in `functions` module to determine how best you can modify `main.py`
3. Customize the path to save your outputs and specifying which `.pt` file to load by modifying `config/config.py`
4. `cd` to the root folder of this project
5. Run the main process by calling `python main.py`

---
## v1.0 Release Notes
- `v1.0` is still in its preliminary stage
- Now we could not support drawing original images alongside with the masked version