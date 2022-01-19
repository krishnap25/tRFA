# RFA: Robust Aggregation for Federated Learning

This code provides a PyTorch implementation of 
robust aggregation algorithms for federated learning.
This codebase is based on a fork of the [Leaf](leaf.cmu.edu) benchmark suite
and provides scripts to reproduce the experimental results in the 
[paper](https://arxiv.org/abs/1912.13445):

K. Pillutla, S. M. Kakade and Z. Harchaoui. 
Robust Aggregation for Federated Learning. arXiv preprint, 2019 

If you use this code, please cite the paper using the bibtex reference below

```
@article{pillutla2019robust ,
  title={{R}obust {A}ggregation for {F}ederated {L}earning},
  author={Pillutla, Krishna and  Kakade, Sham M. and Harchaoui, Zaid},
  journal={arXiv preprint},
  year={2019}
}
```

### Please see [here](https://github.com/krishnap25/RFA) for a TensorFlow implementation of RFA.

Introduction
-----------------
Federated Learning is a paradigm to train centralized machine learning models 
on data distributed over a large number of devices such as mobile phones.
A typical federated learning algorithm consists in local computation on some 
of the devices followed by secure aggregation of individual device updates 
to update the central model. 

The [accompanying paper](https://arxiv.org/abs/1912.13445) describes a 
robust aggregation approach to make federated learning robust 
to settings when a fraction of the devices may be sending outlier updates to the server. 

This code compares the RobustFedAgg algorithm proposed in the accompanying paper
to the FedAvg algorithm ([McMahan et. al. 2017](https://arxiv.org/abs/1602.05629)).
The code has been developed from a fork of [Leaf](leaf.cmu.edu), commit 
```51ab702af932090b3bd122af1a812ea4da6d8740```.


Installation                                                                                                                   
-----------------
This code is written in Python 3.8
and has been tested on PyTorch 1.4+.
A conda environment file is provided in 
`rfa.yml` with all dependencies except PyTorch. 
It can be installed by using 
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
as follows

```
conda env create -f rfa.yml 
```


**Installing PyTorch:** Instructions to install 
a PyTorch compatible with the CUDA on your GPUs
can be found [here](https://pytorch.org/get-started/locally/).
Note that PyTorch can be installed without GPU support, and can be used to reproduce experiments
from the paper. 
However, for speed of execution, it is highly recommended to enable GPU support for the neural network 
experiments on datasets EMNIST and Shakespeare.

The primary dependencies are PyTorch, Numpy, Scipy, Pillow and Pandas.
The code has been tested on Ubuntu 18.04.


Data Setup
-----------


1. Sent140

  * **Overview:** Classify sentiment of tweets as positive or negative
  * **Details:** 877 users used for experiments
  * **Task:** Sentiment Analysis
  * **Setup:** Go to ```data/sent140``` and run the command (~1.2G of disk space)
 
```
time ./preprocess.sh -s niid --sf 1.0 -k 100 -t sample -tf 0.8
```


2. EMNIST (Called FEMNIST here)

  * **Overview:** Character Recognition Dataset
  * **Details:** 62 classes (10 digits, 26 lowercase, 26 uppercase), 3500 total users, 1000 users used for experiments
  * **Task:** Image Classification
  * **Setup:** Go to ```data/femnist``` and run the command (takes ~1 hour and ~25G of disk space) 
  
```
time ./preprocess.sh -s niid --sf 1.0 -k 100 -t sample
```
NOTE: The EMNIST experiments in the paper were produced using the [TensorFlow implementation of RFA](https://github.com/krishnap25/RFA), so 
please use that repository to exactly reproduce the results.

3. Shakespeare

  * **Overview:** Text Dataset of Shakespeare Dialogues
  * **Details:** 2288 total users, 628 users used for experiments
  * **Task:** Next-Character Prediction
  * **Setup:** Go to ```data/shakespeare``` and run the command (takes ~17 sec and ~50M of disk space)
 
```
time ./preprocess.sh -s niid --sf 1.0 -k 100 -t sample -tf 0.8
```
NOTE: The Shakespeare experiments in the paper were produced using the [TensorFlow implementation of RFA](https://github.com/krishnap25/RFA), so 
please use that repository to exactly reproduce the results.



Reproducting Experiments in the Paper
-------------------------------------

Once the data has been set up, the scripts provided in the folder ```experiments/``` can be used 
to reproduce the experiments in the paper.
Note that GPU computations are non-deterministic. Consequently, the ConvNet and LSTM 
experiments reported in the paper, which were run using GPUs, are not perfectly reproducible. 


From the base folder of this repository, first create the folder 
`outputs` as 
```
mkdir outputs
```
and run the scripts, for example, `shakespeare_lstm.sh` as 
```
./experiments/main/shakespeare_lstm.sh
``` 
