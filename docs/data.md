# Download MAPF Benchmark Data

Download the maps (.map) and scene (.scen) files from [MAPF Benchmark](https://movingai.com/benchmarks/mapf/index.html) to `./data/map` and `./data/scene` folder.

```
# make a folder named data under current directory
mkdir -p data/map && mkdir -p data/scene && cd data
wget https://movingai.com/benchmarks/mapf/mapf-map.zip
wget https://movingai.com/benchmarks/mapf/mapf-scen-random.zip
# unzip and put it under data/map
unzip mapf-map.zip -d map
unzip mapf-scen-random.zip && mv scen-random scene
```

# Download Initial States

For MAPF-LNS2 initialized states, please download from [lns2 init states JSON](https://www.dropbox.com/scl/fi/n2ymplfot8sam3ilw03bo/lns2_init_states.zip?rlkey=jkxqaai78yy5ny9hefz0fe4ly&st=3oghuigi&dl=0) or using `wget`:
```
wget "https://www.dropbox.com/scl/fi/n2ymplfot8sam3ilw03bo/lns2_init_states.zip?rlkey=jkxqaai78yy5ny9hefz0fe4ly&st=xoohobnb&dl=1" -O lns2_init_states.zip
```
For LACAM2 initialized states, please download from [lacam2 init states JSON](https://www.dropbox.com/scl/fi/3gkz38e859np4iox1bvau/lacam2_init_states.zip?rlkey=me4sfxfb8k99q8hp92kj4tuvk&st=8ziwj0hi&dl=0)
or using `wget`:
```
wget "https://www.dropbox.com/scl/fi/3gkz38e859np4iox1bvau/lacam2_init_states.zip?rlkey=me4sfxfb8k99q8hp92kj4tuvk&st=ifn3e9wa&dl=1" -O lacam2_init_states.zip
```



# Download SVM Data


For the SVM training process, 16 randomly generated initial states and validation data from 4 additional states are required. You can download this data from [svm data download](https://www.dropbox.com/scl/fo/hkngdqqltjsgvntus3g1j/AK14d2MgAJPSsH9bO1ynSIE?rlkey=brv71d4sddjmwzl4jw69dyhql&st=4b07c873&dl=0) or using `wget`:
```
wget "https://www.dropbox.com/scl/fo/hkngdqqltjsgvntus3g1j/AK14d2MgAJPSsH9bO1ynSIE?rlkey=brv71d4sddjmwzl4jw69dyhql&st=6o7xo3pq&dl=1" -O AK14d2MgAJPSsH9bO1ynSIE.zip
```



After completing the above steps, your data structure should appear as follows:
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
