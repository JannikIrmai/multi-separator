# Multi-separator

This repository contains implementations of local-search algorithms for the multi-separator problem that were presented in [1].
It also contains the source code to reproduce the experiments that are conducted in that paper.

## Installation

The multi-separator algorithms can be installed as a python package by executing
```
pip install multi-separator-algorithms/
```
from the root directory of this repository.


## Experiments

To reproduce the experiments from Section 5 of the article, proceed as follows:

0. Navigate to the `experiments` directory.
1. run `pip install -r requirements.txt` to install all required packages.
This will, in particular, install the custom packages `multi-separator-algorithms` and `partition-comparision` that are included in this supplementary material.
2. run `mkdir results` to create a directory where the results can be saved. 
3. for a given type of image (either `img_type=filament` or `img_type=cell`) run `python experiments.py img_type`.
This will: 
    - Create a hdf5 file to store all the results 
    - Generate synthetic images for different levels of noise
    - Compute segmentations with the multi-separator algorithms and the watershed algorithm
    - Evaluate the quality of the segmentations by computing various metrics
    - Produce figures that summarize the results.
   
Running the experiments takes several hours as many segmentations will be computed 
(10 seeds, 21 levels of noise, 51 bias parameters for the multi-separator algorithm, more than 1000 parameters for the watershed algorithm).
Within the main function of the `experiments.py` file several functions are called in sequence. 
To speed up the evaluation of all experiments some of these function can be called in parallel (for example, it is possible to evaluate the experiments for all 21 different noise levels in parallel).

### Runtime
To evaluate the runtime performance of the GSS and GSG algorithm run
`python performance_analysis.py`.

### Long-range interactions
To evaluate the effect of long-range interactions, run
`python interaction_length_comparison.py img_type`
with either `img_type=filament` or `img_type=cell`.



### Connectomics
The data for reproducing the neuron segmentation experiments can be downloaded from [here](https://datashare.tu-dresden.de/s/c6aYDLaP7bP8Dbp).
This data includes the membrane estimates and affinities that are computed by a CNN.
The implementation of this CNN is publicly available [here](https://github.com/markschoene/MeLeCoLe).

If you use any of this data in your work, please cite [Kasthure et al. (2015)](https://www.sciencedirect.com/science/article/pii/S0092867415008247) and [Lee et al. (2021)](https://ieeexplore.ieee.org/document/9489304).


To run the experiments, create a directeory `results/connectomics` for storing the results and then execute `python connectomics_experiments.py`.

## References

[1] Irmai, J., Zhao, S., Schöne, M. et al. A Graph Multi-separator Problem for Image Segmentation. J Math Imaging Vis (2024). [https://doi.org/10.1007/s10851-024-01201-1](https://link.springer.com/article/10.1007/s10851-024-01201-1)

