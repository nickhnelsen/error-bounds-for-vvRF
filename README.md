# Code for the paper ``Error Bounds for Learning with Vector-Valued Random Features''

## Installation
The command
```
conda env create -f Project.yml
```
creates an environment called ``operator``. [PyTorch](https://pytorch.org/) will be installed in this step.

Activate the environment with
```
conda activate operator
```
and deactivate with
```
conda deactivate
```


## Data
The 1D viscous Burgers' equation dataset is a standard operator learning benchmark first introduced in [Nelsen and Stuart 2021](https://arxiv.org/abs/2005.10224).

The particular setup used in this example comes from [zongyi-li/fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator) and is found below:

* [Burgers' dataset](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)

Please download ``Burgers_R10.zip`` which contains the dataset file ``burgers_data_R10.mat``. There are $2048$ input-outpairs at spatial resolution $8192$.

## Running the example
In the script ``run_sweep_script.py``, assign in the variable ``data_path`` the global path to the data file ``burgers_data_R10.mat``.

The example may then be run as
```
python -u run_sweep_script.py M N J
```
where
* ``M`` is the number of random features,
* ``N`` is the number of training data pairs,
* ``J`` is the desired spatial resolution for training and testing.

The code defaults to running on GPU, if one is available.

## References
- [Error Bounds for Learning with Vector-Valued Random Features](https://arxiv.org/abs/2305.17170)
- [The Random Feature Model for Input-Output Maps between Banach Spaces](https://arxiv.org/abs/2005.10224)
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)

