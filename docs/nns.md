

# Introduction

This is the implementation of [Neural Neighborhood Search for Multi-agent Path Finding](https://openreview.net/pdf?id=2NpAw2QJBY). The model implementation refers to the [official repo](https://github.com/mit-wu-lab/mapf_neural_neighborhood_search) of the paper.

# Usage 

## Data Collection

**Step 1:** Generate more initial states for data collection.
```shell
python nns/run/generate_init_state.py \
--map_path data/example/random-32-32-20.map \
--scene_folder data/scene \
--scene_start_idx 0 \
--scene_end_idx 1
```
- map_path (required): the .map file downloaded from the MAPF benchmark
- scene_folder (required): the folder that stores the .scen files downloaded from the MAPF benchmark
- scene_start_idx (optional): the start index of the scenes to be generated
- scene_end_idx (optional): the end index of the scenes to be generated


You can find more details and explanations for all parameters with:
```shell
python nns/run/generate_init_state.py --help
```

**Step 2** Collect NNS training data. The training data will be save in npz files.
<!-- for the paper suggested setting (NNS-Orig) with PBS as replan solver. -->
```shell
# Collect NNS training data with PBS as replan solver
python nns/run/data_collection.py \
--replan_solver pbs \
--initial_state data/example/map-random-32-32-20-scene-1-agent-150.json \
--output_folder output/nns_pbs_gen_data \
--gen_subset_exe exe/lns-removal-replan/lns-removal-replan \
--pbs_replan_exe exe/pbs-replan/pbs-replan \
--destroyStrategy  RandomWalkLarge \
--neighborSize 25 --num_subset 100 --max_iter 1 

# Collect NNS training data with PP as replan solver
python nns/run/data_collection.py \
--replan_solver pp \
--initial_state data/example/map-random-32-32-20-scene-1-agent-150.json \
--output_folder output/nns_pp_gen_data \
--gen_subset_exe exe/lns-removal-replan/lns-removal-replan \
--destroyStrategy  RandomWalk \
--neighborSize 16 --num_subset 100 --max_iter 1 

```

- `replan_solver` (required): the option for data collection, which can be either pbs or pp
- `initial_state` (required): the path to the initial state file
- `output_folder` (optional): the folder to stored the generated data
- `gen_subset_exe` (required): the path to the executable file for generating subsets
- `pbs_replan_exe` (required): the path to the executable file for PBS replan
- `destroyStrategy` (optional): the strategy for destroying the current state
- `--num_subset` (optional): the number of subset
- `--destroyStrategy` (optional): LNS destory strategy
- `--neighborSize` (optional): neighbourhood size of the specified removal 
- `--max_iter` (optional): the maximum number of LNS iterations for the data collection

The data collection configuration for getting NNS-ours and NNS-orig is shown in the table below:

|  Method  |         Map |    Agent   |  --destroyStrategy  | --neighborSize | --replan_solver |  --max_iter |
|:--------:|:-----------:|:----------:|:-----------------:|:------------:|:-------------:|:---------:|
| Orig-NNS | empty-32-32 |     350    |       Random      |      50      |       pbs      |     50    |
| Orig-NNS | random-32-32-20 |     250    |       RandomWalkLarge      |      25      |       pbs      |     50    |
| Orig-NNS | warehouse-10-20-10-2-1 |     250    |       RandomWalkLarge      |      25      |       pbs      |     25    |
| Orig-NNS | ost003d |     400    |       RandomWalkLarge      |      10      |       pbs      |     25    |
| Orig-NNS | den520d |     700    |       RandomWalkLarge      |      25      |       pbs      |     50    |
| Orig-NNS | Paris_1_256 |     450    |       RandomWalkLarge      |      25      |       pbs      |     50    |
| Our-NNS | empty-32-32 |     350    |       RandomWalk      |      16      |       pp      |     1400    |
| Our-NNS | random-32-32-20 |     250    |       RandomWalk      |      8      |       pp      |     1000    |
| Our-NNS | warehouse-10-20-10-2-1 |     250    |       Adaptive      |      32      |       pp      |     200    |
| Our-NNS | ost003d |     400    |       RandomWalkProb      |      16      |       pp      |     400    |
| Our-NNS | den520d |     700    |       RandomWalkProb      |      16      |       pp      |     500    |
| Our-NNS | Paris_1_256 |     450    |       RandomWalkProb      |      32      |       pp      |     200    |


## Training 



## Inference 



