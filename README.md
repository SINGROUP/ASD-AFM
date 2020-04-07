# ASD-AFM
Automated Structure Discovery in Atomic Force Microscopy

If you are using Anaconda, you can create the required Python environment with
```sh
conda env create -f environment.yml
```
This will create a conda enviroment named tf-gpu with the all the required packages. Activate the environment with
```sh
conda activate tf-gpu
```

To create the datasets and train the models, run `jupyter notebook` in the repository folder, open the `train_model.ipynb` notebook, and follow the instructions therein.

Alternatively, run the script `generate_data.py` to generate the datasets and the script `train_models.py` to train the models.

The folder `pretrained_weights` holds weights for pretrained models on the two datasets.
