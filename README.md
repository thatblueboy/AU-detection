# AU-detection

## 📌 Table of Contents
- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Make Your Own Graph Dataset](#make-your-own-graph-dataset)

## Overview
This repository contains tools and datasets for Action Unit (AU) detection using spatio-temporal graph convolutional networks. The framework processes facial video data into graph formats suitable for graph-based learning approaches.


## Folder Structure
```configs``` Configuration files for graph datasets, contains args such as graph nodes and edges.

```data``` (Local) Raw facial video datasets.

```datasets``` PyTorch datasets to process raw video data and get spatio-temporal graph data.

```logs``` (Local) Tensorboard logs

```models``` Spatio-Temporal Graph Convolutional Network used in this work

```notebooks``` (Local) Notebooks for experimenting, data cleaning, and data analysis.

```pickles``` Graph datasets saved as pickle files.

```train``` Main training files and train utils.

## Usage
To use SAMM, SAMMLong, or CASMEII spatio temporal graph datasets,

1. In a new training notebook, import SAMMAUGraphDataset from datasets

```python
from datasets.SAMM.in_mem_graph_dataset import SAMMAUGraphDataset
```

2. Instantiate the dataset with appropriate arguments. Arguments used in this work are provided in configs. Alternatively graph datasets saved in pickle can be used directly.

2.1. Using arguments from config (requires raw data to enable graph generation)

```python

with open("/configs/samm.json", "r") as file:
    config = json.load(file)
train_config = config["lips"]

dataset = SAMMAUGraphDataset(frames_path = train_config["frames_path"],
                    labels_csv_path = train_config["labels_path"],
                    aus=aus,
                    central_landmark = train_config["central_landmark"],
                    landmarks = train_config["ROI_index"],
                    num_timesteps = train_config["num_timesteps"],
                    edges = train_config["edges"],
                    noise=True,
                    include_flipped = True, 
                    self_loop_in_edge_index = True,
                    normalizing_factor = 200)
```

2.2 Using saved graph dataset. Pl refer to ```pickle/notes.md``` for details of saved graphs.

```python

dataset = SAMMAUGraphDataset(
                    noise=True,
                    processed_data_path="/pickle/samm_au12_with_flips.pkl")

```

3. Some helper functions for training can be found in ```train/utils```. The function are designed to work with the specific outputs of the graph datasets(any derivatives of ```BaseInMemoryGraphDataset``` present in ```datasets/utils/base_in_mem_graph_dataset.py```).


## Make your Own Graph Dataset

1. In ```data/yourdataset``` you will need organized facial video data, and a ```.csv``` file with appropriate indicators(like subject names, or file names) to locate video frames for each datapoint along with binary 0/1 labels for each au.

2. In ```datasets/yourdataset/graph_dataset.py``` define a concrete class derived from the ABC in ```datasets/utils/base_in_mem_graph_dataset.py``` by implemting the generate_graphs(). Note that the generated graph dataset is loaded directly to memory since we are expecting small datasets.

3. Instantiating and calling the dataset once using appropriate arguments and path to a pickle file 
```python
from datasets.yourdataset import  GraphDataset

dataset = GraphDataset(args**, 
                processed_data_path = "pickle/your_graph_dataset.pkl")
```
 will save the generated graph dataset as a pkl, to save graph generation time in future use.
