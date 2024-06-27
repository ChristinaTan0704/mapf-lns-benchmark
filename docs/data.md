# Download MAPF Benchmark Data

Download the maps (.map) and scene (.scen) files from [MAPF Benchmark](https://movingai.com/benchmarks/mapf/index.html) to `./data/map` and `./data/scene` folder.

```
# make a folder named data under current directly
mkdir -p data/map && mkdir -p data/scene && cd data
wget https://movingai.com/benchmarks/mapf/mapf-map.zip
wget https://movingai.com/benchmarks/mapf/mapf-scen-random.zip
# unzip and put it under data/map
unzip mapf-map.zip -d map
unzip mapf-scen-random.zip && mv scen-random scene
```

# Download Initial States

For MAPF-LNS2 initialied state, please downlowd from [lns2 init](https://www.dropbox.com/scl/fi/n2ymplfot8sam3ilw03bo/lns2_init_states.zip?rlkey=jkxqaai78yy5ny9hefz0fe4ly&st=3oghuigi&dl=0).
For LACAM2 initialied state, please downlowd from [lacam2 init](https://www.dropbox.com/scl/fi/3gkz38e859np4iox1bvau/lacam2_init_states.zip?rlkey=me4sfxfb8k99q8hp92kj4tuvk&st=8ziwj0hi&dl=0).


# Download SVM Data


To train SVM, we need to have 16 randomly generated initial states, and validation data extracted from another 4 states. Please download the data from [svm data download](https://www.dropbox.com/scl/fo/hkngdqqltjsgvntus3g1j/AK14d2MgAJPSsH9bO1ynSIE?rlkey=brv71d4sddjmwzl4jw69dyhql&st=4b07c873&dl=0).




The data structure after the above steps will looks like:
```
mapf-lns-benchmark
└── data
    ├── map
    ├── scene
    └── svm
        ├── train
        ├── train_shortest_path
        └── val

```