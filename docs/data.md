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

After this, the data structure should looks like:
```
maptracker
├── mmdetection3d
    ├── example
    ├── generate_init_state.py
    ├── map
    ├── scene
    └── svm

```


# Download SVM Validation Data