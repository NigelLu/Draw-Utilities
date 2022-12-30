# The Drawing Tools for Semantic Segmentation

## *Notion* Tutorial
- You may access the **Notion doc** for this repo via [this link](https://distinct-walleye-0b3.notion.site/Draw-Utilities-f06c41393ba840edaccec2c02b4bcbda);
- The Notion documentation walks you through the modules provided by this repo, and the underlying logic behind its design;
- Hope this doc can help you better utilize this repo. Cheers! 🤗

## File Structure
- `config`
    - `__init__.py`
    - `config.py`
    - `config_pt.py`
    - `config_pascal.py`
- `dataset`
    - `__init__.py`
    - `lists`
        - `pascal`
            - `train.txt`
            - `val.txt`
    - `VOC2012`
        - `.gitkeep`
    - `utils.py`
    - `classes.py`
    - `dataset.py`
    - `transform.py`
- `draw`
    - `__init__.py`
    - `drawmask.py`
    - `drawlabel.py`
- `functions`
    - `__init__.py`
    - `mask.py`
    - `utils.py`
    - `convert.py`
- `output`
    - `.gitkeep`
- `pt-files`
    - `.gitkeep`
- `main-pt.py`
- `main-pascal.py`
- `.gitignore`
- `LICENSE`
- `README.md`
- `requirements.txt`

---
## Preparation
- Install the required packages by running `python -m pip install -r requirements.txt` 
    - we recommend using `Python ^3.7.6` here
    - if you could not find the exact version of the packages listed in `requirements.txt`, it is acceptable to use a slightly newer/older version

---
## Hands-on

### `main-pt.py`
- Download a pre-defined `.pt` file from this [drive link](https://drive.google.com/file/d/1ydF8tyVupIqYYsqoZGGYR3hWYr3_jykM/view?usp=sharing)
- Place the downloaded .pt file and put it under `pt-files/`
- `cd` to the root folder of this project
- Run `python main-pt.py` and see the results in `output` folder!

### `main-pascal.py`
- Download `PASCAL-VOC2012` dataset from this [drive link](https://drive.google.com/file/d/1zuek0869oc6VZO1kf8K-DWx9tgo8PVyc/view?usp=sharing)
- Unzip the file and place the folder named `VOC2012` under `dataset/`
- `cd` to the root folder of this project
- Run `python main-pascal.py` and see the results in `output` folder!

---
## Usage 
1. Save the `.pt` files 
2. Modify the `main-pt.py` based on the task
    - you may go through what utility functions offered in `functions` module to determine how best you can tailor `main-pt.py` to your needs
3. Customize the path to save your outputs and specifying which `.pt` file to load by modifying `config/config.py`
4. `cd` to the root folder of this project
5. Run the main process by calling `python main-pt.py`

---
## v1.0 Release Notes
- `v1.0` is still in its preliminary stage
- Now we could not support drawing original images alongside with the masked version

---
## v1.1 Release Notes
- `v1.1` adds support for directly loading PASCAL-VOC2012 
- This codebase now supports both drawing from pt files or raw pascal data. 
- For this release, we added dataset module to establish dataloders for pascal.

## v1.2 Release Notes
- `v1.2` provides more detailed documentation and updates `README.md` to better guide the users