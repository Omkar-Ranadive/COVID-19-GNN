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

