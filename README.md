# Graph Transport Network (GTN)

Reference implementation of the graph transport network (GTN), as proposed in our paper 

**[Scalable Optimal Transport in High Dimensions for Graph Distances, Embedding Alignment, and More](https://www.cs.cit.tum.de/daml/lcn/)**   
by Johannes Gasteiger, Marten Lienen, Stephan Günnemann  
Published at ICML 2021.

Note that the author's name has changed from Johannes Klicpera to Johannes Gasteiger.

The paper furthermore proposed the locally corrected Nyström (LCN) approximation, sparse Sinkhorn, and LCN-Sinkhorn, whose implementations you can find in [this accompanying repository](https://github.com/gasteigerjo/lcn). GTN uses these approximations and relies on the implementations provided in the LCN repository.

## Installation
You can install the repository using `pip install -e .`.

## Training GTN
This repository contains a notebook for training and evaluating GTN (`experiment.ipynb`) and a script for running this on a cluster with [SEML](https://github.com/TUM-DAML/seml) (`experiment_seml.py`).

The config files specify all hyperparameters and allow reproducing the results in the paper.

## Contact
Please contact j.gasteiger@in.tum.de if you have any questions.

## Cite
Please cite our paper if you use our method or code in your own work:

```
@inproceedings{gasteiger_2021_lcn,
  title={Scalable Optimal Transport in High Dimensions for Graph Distances, Embedding Alignment, and More},
  author={Gasteiger, Johannes and Lienen, Marten and G{\"u}nnemann, Stephan},
  booktitle = {Thirty-eighth International Conference on Machine Learning (ICML)},
  year={2021},
}
```
