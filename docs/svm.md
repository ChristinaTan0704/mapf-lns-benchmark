

# Introduction

This is the implementation of [Anytime Multi-Agent Path Finding via Machine Learning-Guided Large Neighborhood Search](https://taoanhuang.github.io/files/aaai22.pdf).

# Usage 

## Training 

**Step 1** Download the training data by following [Data Preparation](data.md).

**Step 2**: Download the SVM models from the [SVM Rank Official Website](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html).

According to the authors of SVM-LNS, they used the SVM Rank model from the [official SVM Rank website](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html). Please follow the instructions on the official website to download the required SVM Rank models. The unzipped package from the website should contain the files `svm_rank_classify` and `svm_rank_learn`. Place these files under the `exe/svm_rank` folder. The folder structure should look like this:

```
exe
└── svm_rank
    ├── svm_rank_classify
    └── svm_rank_learn
```

Quick example for downloading the SVM models on a Linux 64-bit system:

```shell
# Create a directory for the SVM Rank models 
mkdir -p exe/svm_rank
cd exe/svm_rank
# Download and unzip the model package from the official website
wget https://osmot.cs.cornell.edu/svm_rank/current/svm_rank_linux64.tar.gz
tar -xvzf svm_rank_linux64.tar.gz
```


**Step 3** Train the model by:

```python
python svm/svm_run.py --option train \
--initial_state data/svm/train/map-random-32-32-20-scene-1-agent-150.json data/svm/train/map-random-32-32-20-scene-2-agent-150.json \
--val_data_path data/svm/val/orig_svm/random-32-32-20.dat \
--shortest_path_folder data/svm/train_shortest_path \
--gen_subset_exe exe/lns-removal-replan/lns-removal-replan \
--max_iter 100  --replan_time_limit 0.6 --pprun 6 \
--num_subset 20  --uniformNB 1 --destroyStrategy Adaptive

```

- `--option` : train or infer
- `--initial_state` : the initial state of the training data
- `--val_data_path` : the validation data path
- `--shortest_path_folder` : the folder to store the shortest path
- `--gen_subset_exe` : the path to the executable file to generate the subset
- `--max_iter` : the maximum number of iterations
- `--replan_time_limit` : the time limit for each replan
- `--pprun` : the number of ppruns
- `--num_subset` : the number of subsets
- `--uniformNB` : (0) fixed nb_size specified by --neighborSize (1) nb_size sample from {2,4,8,16,32} (2) nb_size sample from 5~16
- `--destroyStrategy` : LNS destroy strategy
- `--neighborSize` : neighborhood size of the specified removal strategy, used when `uniformNB` is 0

## Inference 

After training the model, select the one with the best validation score and perform inference using the following command:

```python
python svm/svm_run.py --option infer \
--initial_state data/svm/train/map-random-32-32-20-scene-1-agent-150.json \
--shortest_path_folder data/svm/train_shortest_path \
--log_path output/svm_inference/map-random-32-32-20-scene-1-agent-150.log \
--svm_model_path output/svm_val_data/2024-06-22_10-18-29/svm_random-32-32-20_agent150_iter0_score12_69.model \
--gen_subset_exe exe/lns-removal-replan/lns-removal-replan \
--max_iter 100  --replan_time_limit 0.6 --pprun 6 \
--num_subset 20  --uniformNB 1 --destroyStrategy Adaptive

```

- `--log_path` : the path to store the log file
- `--svm_model_path` : the path to the svm model


# Parameter Setup for Benchmark Result
To obtain the benchmark results for Orig-SVM and Our-SVM, configure each parameter as specified below:

## Training Configuration


|  Method  |           Map           |        \--destroyStrategy | \--fix_adaptive_weight |        \--uniformNB |        \--neighborSize |
|:--------:|:-----------------------:|:-------------------------:|:----------------------:|:-------------------:|:----------------------:|
| Orig-SVM |           ALL           |          Adaptive         |          1 1 0         |          1          |            /           |
|  Our-SVM |       empty-32-32       |            RandomWalk     |            /           |          0          |               16       |
|  Our-SVM |     random-32-32-20     |             Adaptive      |          1 1 1         |          0          |           16           |
|  Our-SVM | warehouse-10-20-10-2-1  |             Adaptive      |          1 1 1         |          0          |           16           |
|  Our-SVM |         ost003d         |          RandomWalkProb   |            /           |          0          |           16           |
|  Our-SVM |          den520d        |          RandomWalkProb   |            /           |          0          |           16           |
|  Our-SVM |       Paris_1_256       |             Adaptive      |          1 1 1         |          0          |           32           |

## Inference Configuration

|  Method  |     Map     |    Agent   |        \--destroyStrategy | \--fix_adaptive_weight |        \--uniformNB |        \--neighborSize |
|:--------:|:-----------:|:----------:|:-------------------------:|:----------------------:|:-------------------:|:----------------------:|
| Orig-SVM |     ALL     |     ALL    |          Adaptive         |          1 1 0         |          1          |            /           |
|  Our-SVM | empty-32-32 |        300 |            RandomWalk     |            /           |          0          |               16       |
|  Our-SVM | empty-32-32 |        350 |            RandomWalk     |            /           |          0          |               16       |
|  Our-SVM | empty-32-32 |        400 |            Adaptive       |          1 1 1         |          0          |               8        |
|  Our-SVM | empty-32-32 |        450 |            RandomWalk     |            /           |          0          |               8        |
|  Our-SVM | empty-32-32 |        500 |            RandomWalk     |            /           |          0          |               8        |
|  Our-SVM | random-32-32-20 | 150 |            Adaptive       |          1 1 1         |          0          |               16       |
|  Our-SVM | random-32-32-20 | 200 |            RandomWalk     |            /           |          0          |               16       |
|  Our-SVM | random-32-32-20 | 250 |            RandomWalk     |            /           |          0          |               8        |
|  Our-SVM | random-32-32-20 | 300 |            RandomWalk     |            /           |          0          |               8        |
|  Our-SVM | random-32-32-20 | 350 |            RandomWalkProb |            /           |          0          |               8        |
|  Our-SVM | warehouse-10-20-10-2-1 | 150 | Adaptive | 1 1 1 | 0 | 16 |
|  Our-SVM | warehouse-10-20-10-2-1 | 200 | Adaptive | 1 1 1 | 0 | 16 |
|  Our-SVM | warehouse-10-20-10-2-1 | 250 | Adaptive | 1 1 1 | 0 | 32 |
|  Our-SVM | warehouse-10-20-10-2-1 | 300 | RandomWalk | / | 0 | 16 |
|  Our-SVM | warehouse-10-20-10-2-1 | 350 | RandomWalk | / | 0 | 16 |
|  Our-SVM | ost003d | 200 | RandomWalkProb | / | 0 | 16 |
|  Our-SVM | ost003d | 300 | RandomWalkProb | / | 0 | 16 |
|  Our-SVM | ost003d | 400 | RandomWalkProb | / | 0 | 16 |
|  Our-SVM | ost003d | 500 | RandomWalkProb | / | 0 | 8 |
|  Our-SVM | ost003d | 600 | RandomWalkProb | / | 0 | 8 |
|  Our-SVM | den520d | 500 | RandomWalkProb | / | 0 | 16 |
|  Our-SVM | den520d | 600 | RandomWalkProb | / | 0 | 16 |
|  Our-SVM | den520d | 700 | RandomWalkProb | / | 0 | 16 |
|  Our-SVM | den520d | 800 | RandomWalkProb | / | 0 | 16 |
|  Our-SVM | den520d | 900 | RandomWalkProb | / | 0 | 16 |
|  Our-SVM | Paris_1_256 | 350 | RandomWalkProb | / | 0 | 32 |
|  Our-SVM | Paris_1_256 | 450 | RandomWalkProb | / | 0 | 32 |
|  Our-SVM | Paris_1_256 | 550 | RandomWalkProb | / | 0 | 32 |
|  Our-SVM | Paris_1_256 | 650 | RandomWalkProb | / | 0 | 16 |
|  Our-SVM | Paris_1_256 | 750 | RandomWalkProb | / | 0 | 16 |


## Checkpoint

The trained checkpoints for Orig-SVM is available at [Orig-SVM-download](https://www.dropbox.com/scl/fo/hswapqekyohlmnkfwu4lo/AINqa_wl59DWPLPZzJVWefs?rlkey=zksu2k1z3o5eq7b7jx91i5t0a&st=mjqreh7g&dl=0).
The trained checkpoints for Our-SVM is available at [Our-SVM-download](https://www.dropbox.com/scl/fo/qkbv1ktgdtw3i3dae7v1g/AIk1GQtmL8aHaQX2D1Dik4E?rlkey=2alzg26h3by0hz4gvbmcbhkxu&st=d68elp01&dl=0).
