# Training Binary Neural Networks
<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4716863.svg)](https://doi.org/10.5281/zenodo.4716863)

We reproduce the results for the paper, "*Training Binary Neural Networks using the Bayesian Learning Rule*". We make an end-to-end trainer for training Binary Neural Networks using various methods and Keras like usage. This is our entry for the ML Reproducibility Challenge 2020.

## Getting Started
### Central Methods
- **BayesBiNN** - This was the central method in the above-mentioned paper and gives a mathematically principled way of solving the discrete optimization problem in case BNNs.
- **STE** - This is another method for optimizing BNNs originally mentioned in the paper "*Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*". This method gives a much smoother training as compared to BayesBiNN and also gives good performance on more complex tasks like Semantic Segmentation than Image Classification.

### Requirements
```
tqdm
torch
wandb               #optional
torchvision
torchsummary
Pillow==7.2.2
opencv-python
```

### Installing Prerequisites
```bash
pip install -r requirements.txt
```
### Supported Datasets
Currently, our code explicitly supports MNIST, CIFAR-10 and CIFAR-100 datasets but small changes to the data loader file can extend it to other custom datasets.

### Wandb Support
We have added WandB support to monitor the training of models on a larger scale.

## Usage
### BayesBiNN
#### Running the code
Use *wandb_on* argument to connect it to the WandB server. 
```
python train_bayesbinn.py /path/to/config [wandb_on]

# Example python train_bayesbinn.py configs/mnist_config_bayesbinn.json wandb_on

```

#### Configuration File
```json
{
    "dataset": "cifar10",
    "input_shape": 3,
    "output_shape": 10,
    "data_augmentation": true,
    "epochs": 500,
    "criterion": "crossentropy",
    "validation_split": 0.1,
    "lr_scheduler": "cosine",
    "batch_size": 50,
    "lr_init": 3e-4,
    "lr_final": 1e-16,
    "drop_prob": 0.2,
    "batch_affine": false,
    "model_architecture": "VGGBinaryConnect",
    "mc_steps": 1,
    "temperature": 1e-10,
    "evaluate_steps": 0,
    "momentum": 0.2
}
```
### STE
#### Running the code
Use *wandb_on* argument to connect it to the WandB server. 
```
python train_ste.py /path/to/config [wandb_on]

# Example python train_ste.py configs/mnist_config_ste.json wandb_on

```

#### Configuration File
```json
{
    "dataset": "cifar10",
    "input_shape": 3,
    "output_shape": 10,
    "data_augmentation": true,
    "epochs": 500,
    "criterion": "crossentropy",
    "validation_split": 0.1,
    "lr_scheduler": "cosine",
    "batch_size": 50,
    "lr_init": 0.01,
    "lr_final": 1e-16,
    "batch_affine": false,
    "model_architecture": "VGGBinaryConnect",
    "momentum": 0.2,
    "grad_clip_value": 1,
    "weight_clip_value": 1
}
```

## Main Contributors
* Prateek Garg (IIT Delhi)
* Lakshya Singhal (IIT Delhi)

## References
```
@misc{meng2020training,
      title={Training Binary Neural Networks using the Bayesian Learning Rule}, 
      author={Xiangming Meng and Roman Bachmann and Mohammad Emtiyaz Khan},
      year={2020},
      eprint={2002.10778},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{bengio2013estimating,
      title={Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation}, 
      author={Yoshua Bengio and Nicholas Léonard and Aaron Courville},
      year={2013},
      eprint={1308.3432},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.