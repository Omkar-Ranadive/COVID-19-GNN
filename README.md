# Analyzing spread of COVID-19 using Graph Neural Networks 
The aim of this project is to provide users with an end-to-end pipeline for processing data related to COVID-19, create graph structure from that data and allow users to easily manage node features and edge weights. 

## Installation 
* Install pytorch from the [official site](https://pytorch.org/get-started/locally/)
* Install pytorch-geometric by executing the following: 
```
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
```
where ${CUDA} should be replaced by either cpu, cu92, cu100 or cu101 depending on your PyTorch installation.
* Run `pip install -r requirements.txt`
* Download basemap wheel package from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#basemap)
* Go the the directory where wheel package is downloaded and pip install the package 

## Dataset 
The dataset folder can be downloaded [here.](https://drive.google.com/file/d/1ybNYM9-q524GCmXcr-PvlTVR_ye6B5-f/view?usp=sharing) Extract and replace the existing dataset folder with this one.

## File information 
* dataloader.py - This file can be used to load population data, flight data and form a dictionary of graph nodes. 
* data_preparation.py - This file is similar to dataloader.py but also handles timeseries data 
* graph_loader.py  - This file is used to create the actual graph object (pytorch geometric object). 
* model_architecture.py - Three GNN architectures are present here - Graph Convolution Network, SageConv, Message Passing. Users can additionally implement their own architectures in this file 
* corona.py - Can be used to create an inmemory dataset.
* visualizer.py/heatmap.py - These files can be used to visualize the graph and identify nodes of interest 

## Usage 
Make modifications to the files mentioned in the previous section and then run train.py 




