# background-remover

A flexible and customizable tool for background removal, based on the original U-Net structure. It supports training on your own datasets, provides multiple data-loading modes, and offers both CLI and API interfaces for integration into any workflow.

## Features

- **U-Net** architecture for segmentation
- **Custom training** support for your own datasets (flat or split directory structures)
- **Data augmentation** built-in for more robust models
- **CLI** tools for training, inference, and dataset handling
- **REST API** for real-time inference, returning results as ZIP files (cropped image + mask)

## Installation
If you have not done so already, clone this repository using ```git clone```.

### Without docker

- Requirements:
    - Python 3.11+

- Install dependencies (using a virtual environment or another environment): ```pip install -r requirements.txt```

- Run locally: ```python3 main.py training```

### With docker

- Build the container:
    - ```docker build -t unet .```

- Run the container:
    - ```docker run -it unet bash```

- You are now inside the container and can run the main script:
    - ```python3 main.py training```

<em>Note: If you need to use external directories inside the container, mount them with ```-v "path/to/data:/data"``` and access them via ```/data```. </em>

## Usage

### Training

Training is performed exclusively via the command line:

**Available arguments:**
- ```--epochs```: Number of training epochs (default: 20)
- ```--batch```: Batch size for training (default: 4)
- ```--lr```: Learning rate for the optimizer (default: 1e-4)
- ```--early-stopping```: Use early stopping during training
- ```--output-dir```: Path to the output directory for saving model checkpoints (default: ./model)
- ```--data-dir```: Path to the data directory (default: ./data)
- ```--resume-from```: Path to a model checkpoint to resume training from (default: None)
- ```--verbose```: Print training progress
- ```--data-type```: Data directory structure (more below)
- ```--val-split```: Fraction of data to use for validation when --data-type=flat (default: 0.2)

<em>Note: After each epoch, a checkpoint is saved in ```models/``` as a backup in case training fails. This temporary checkpoint will be deleted once the final model is successfully saved. If training stops unexpectedly and you want to resume, use this checkpoint with ```--resume-from```.</em>

### Dataset Structure

**When using ```--data-type flat```:**

Use the following simplified structure:

```
data/
├───images
└───masks
```

The validation split will be created automatically using ```--val-split```.

**When using ```--data-type split```:**

The dataset must follow this structure:

```
data/
├───train
│   ├───images
│   └───masks
└───val
    ├───images
    └───masks
```
Image and mask filenames should be numeric (e.g., ```1.png```, ```2.png```, ```3.png```).

### Inference

**Via CLI:**

- ```--image```: Path to the input image used for inference
- ```--model```: Path to the trained UNet model
- ```--output-dir```: Path to the output directory for saving inference results (default: ./output)

**Via API:**

To start the API server, run:
- ```python3 main.py api --run```

Endpoint:
- ```/inference?model=model_name```

The API accepts an uploaded image (e.g., as multipart/form-data). The response is a ZIP file containing:
- image.png – the cropped input image used for inference
- mask.png – the predicted segmentation mask

## References

```bibtex
@misc{ronneberger2015unetconvolutionalnetworksbiomedical,
      title={U-Net: Convolutional Networks for Biomedical Image Segmentation}, 
      author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
      year={2015},
      eprint={1505.04597},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1505.04597}, 
}
```