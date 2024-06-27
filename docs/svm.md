

# Introduction

This is the implementation of [Anytime Multi-Agent Path Finding via Machine Learning-Guided Large Neighborhood Search](https://taoanhuang.github.io/files/aaai22.pdf).

# Usage 

## Training 

**Step 1:** Download the training data by following [Data Preparation](../data.md).

<!-- Generate more initial states for training. For our benchmark and the [orignal paper](https://taoanhuang.github.io/files/aaai22.pdf), it generate 16 scenes for training.


```shell
python data/generate_init_state.py \
--map_path data/example/random-32-32-20.map \
--scene_folder data/scene \
--scene_start_idx 0 \
--scene_end_idx 1 \
--agent_num 250 
```
- map_path (required): the .map file downloaded from the MAPF benchmark
- scene_folder (required): the folder that stores the .scen files downloaded from the MAPF benchmark
- agent_num (required): number of agents in the scene
- scene_start_idx (optional): the start index of the scenes to be generated
- scene_end_idx (optional): the end index of the scenes to be generated


You can find more details and explanations for all parameters with:
```shell
python data/generate_init_state.py --help
``` -->
**Step 2** Train the model by:

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
- `--max_iter` : the maximum number of iteration
- `--replan_time_limit` : the time limit for each replan
- `--pprun` : the number of pprun
- `--num_subset` : the number of subset
- `--uniformNB` : (0) fixed nb_size specified by --neighborSize (1) nb_size sample from {2,4,8,16,32} (2) nb_size sample from 5~16
- `--destroyStrategy` : LNS destory strategy
- `--neighborSize` : neighbourhood size of the specified removal strategy, used when `uniformNB` is 0

## Inference 

After training the model, pick the one with best validation score and inference by:

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
To get the benchmark results for Orig-SVM and Our-SVM, please set the value of each parameter as follow:



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

Trained check point for Orig-SVM can be found on [Orig-SVM-download]().
Trained check point for Our-SVM can be found on [Our-SVM-download]().