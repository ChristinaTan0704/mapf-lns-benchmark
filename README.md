<div align="center">
<h1 align="center">
  <a href="http://arxiv.org/abs/2407.09451">Benchmarking Large Neighborhood Search for Multi-Agent Path Finding</a>
</h1>

### [Jiaqi Tan*<sup>1</sup>](https://www.linkedin.com/in/jiaqi-christina-tan-800697158/) , [Yudong Luo*<sup>2</sup>](https://miyunluo.com/), [Jiaoyang Li<sup>3</sup>](https://jiaoyangli.me/) , [Hang Ma<sup>1</sup>](https://www.cs.sfu.ca/~hangma/)

### <sup>1</sup> Simon Fraser University <sup>2</sup> University of Waterloo <sup>3</sup> Carnegie Mellon University


</div>



<p align="center">
    <img src="https://github.com/ChristinaTan0704/mapf-lns-benchmark/blob/main/docs/runtime_delay.jpg">
</p>



This repository provides the official implementation of the paper [Benchmarking Large Neighborhood Search for Multi-Agent Path Finding](http://arxiv.org/abs/2407.09451). 


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Methods](#methods)
- [Acknowledgements](#acknowledgements)


## Introduction
Multi-Agent Path Finding (MAPF) aims to arrange collision-free goal-reaching paths for a group of agents. Anytime MAPF solvers based on large neighborhood search (LNS) have gained prominence recently due to their flexibility and scalability. Neighborhood selection strategy is crucial to the success of MAPF-LNS and a flurry of methods have been proposed. 

We conduct a fair comparison across prominent methods on the same benchmark and hyperparameter search settings and promote new challenges for existing learning based methods and present opportunities for future research when machine learning is integrated with MAPF-LNS.

## Installation


**Step 1** Clone the repository with all submodules.
```shell
# Clone the repo and submodules
git clone --recurse-submodules https://github.com/ChristinaTan0704/mapf-lns-benchmark.git
# For each submodue, checkout to the right branch
git submodule foreach 'git checkout $(git config -f $toplevel/.gitmodules submodule.$name.branch || echo master)'
```

**Step 2** Install BOOST (https://www.boost.org/) and Eigen (https://eigen.tuxfamily.org/). 

```shell script
sudo apt update
```
- Install the Eigen library (used for linear algebra computing)
 ```shell script
sudo apt install libeigen3-dev
 ```
- Install the boost library 
 ```shell script
sudo apt install libboost-all-dev
 ```

**Step 3** Create conda environment and install Python dependencies.

```
# Create conda environment and activate
conda create --name mapf python=3.9.1
conda activate mapf
# Install pytorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# Install other requirements
pip install -r requirements.txt
```


## Data preparation

Please check [data preparation](docs/data.md) for how to download the data.
## Methods



### Rule-based 

The executable for generating rule-based results can be found in the submodule under `exe/lns-rule-based`. Please check the [instruction](https://github.com/ChristinaTan0704/mapf-lns-exe/blob/rule-based/README.md) of submodule `exe/lns-rule-based` for how to implement the rule-based results.

### Learning-based

- ### SVM

For instreuctions on how to train and infer SVM-LNS, please follow [SVM-LNS implementation](docs/svm.md).

- ### Neural

For instreuctions on how to train and infer NNS-LNS, please follow [NNS-LNS implementation](docs/nns.md).

- ### Bandit

The executable for generating bandit results can be found in the submodule under `exe/bandit`. For instructions on how to infer bandit, please follow [bandit implementation](https://github.com/ChristinaTan0704/anytime-mapf/blob/main/README.md).


## Acknowledgements
Thanks to the open-source projects [MAPF-LNS](https://github.com/Jiaoyang-Li/MAPF-LNS), [MAPF-LNS2](https://github.com/Jiaoyang-Li/MAPF-LNS2), [lacam2](https://github.com/Kei18/lacam2), [bandit](https://github.com/thomyphan/anytime-mapf), and the model reference from [NNS](https://github.com/mit-wu-lab/mapf_neural_neighborhood_search/tree/main) that we were able to align many details for comparison between different methods, making our project possible.
