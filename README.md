# background-remover

A flexible tool for background removal that can be trained on custom datasets.
It is based on the original U-Net architecture for precise segmentation and supports both API and CLI usage, making it easy to integrate into various applications. 

## Features

- U-Net Based
- Data augmentation
- Custom Training
- API & CLI

## Installation
If you don’t have done yet, clone this repository using ```git clone```.


### Without docker

- Requirements:

    - Python 3.11+
    - Git installed on the system
    - pip packages (see below)
- Install dependencies (with venv or another environment): ```pip install -r requirements.txt```

- Run locally: python3 main.py training

### With docker


## Usage

### CLI

The tool can be controlled through command-line arguments. Below is an overview of all available options:

Training:
- ```--epochs```: Number of training epochs (default: 20)
- ```--batch```: Batch size for training (default: 4)
- ```--lr```: Learning rate for the optimizer (default: 1e-4)
- ```--output-dir```: Path to the output directory for saving model checkpoints (default: ./model/)
- ```--data-dir```: Path to the data directory (default: ./data/)
- ```--verbose```: Print training progress

Inference:
- ```--image```: Path to the input image for inference
- ```--model```: Path to the trained UNet model (default: background_remover/output/unet_bg_removal.pth)
- ```--output-dir```: Path to the output directory for saving inference results (default: ./output/)

### API

### Training

Needs this folder structure:
```
data/
├───train
│   ├───images
│   └───masks
└───val
    ├───images
    └───masks
```
Image names should be numbers (1,2,3,4,5,6.png)

## References