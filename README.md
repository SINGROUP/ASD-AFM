# ASD-AFM

The ASD-AFM repository contains the code for generating the datasets, training the machine learning models, and doing the configuration matching analysis for atomic force microscopy (AFM) images described in the article [*B. Alldritt et. al, Automated structure discovery in atomic force microscopy, Sci. Adv., 2020.*](https://advances.sciencemag.org/content/6/9/eaay6913.full)

The dataset generation is done using the [ProbeParticleModel](https://github.com/ProkopHapala/ProbeParticleModel) AFM simulation code. The machine learning models are implemented in Tensorflow through the Keras API. The code is currently written in Python 2. At least the following Python packages are required:
* numpy
* matplotlib
* tensorflow-gpu
* keras
* pyopencl
* jupyter

Additionally, you need to have Cuda and cuDNN correctly configured on your system in order to train the models on an Nvidia GPU.

If you are using Anaconda, you can create the required Python environment with
```sh
conda env create -f environment.yml
```
This will create a conda enviroment named tf-gpu with the all the required packages. It also has a suitable version of the Cuda toolkit and cuDNN already installed. Activate the environment with
```sh
conda activate tf-gpu
```

To create the datasets and train the models, run `jupyter notebook` in the repository folder, open the `train_model.ipynb` notebook, and follow the instructions therein.

Alternatively, run the script `generate_data.py` to generate the datasets and the script `train_models.py` to train the models.

The folder `pretrained_weights` holds the weights for pretrained models on the two datasets.
