# A Tensorflow Implementation of Mask-RCNN

This repository contains an implementation of Mask-RCNN ased on the Matterport implementation. The network is fully developed in tensorflow 1.14 with no keras fuction inside.


### Prerequisites

The required libraries are:
- python = 3.6
- tensorflow-gpu = 1.14

And the following with the latest compatible version:
- numpy
- scipy
- matplotlib
- pillow
- pycocotools
- cython
- IPython[All]
- Imgaug


## Running the tests

Run on Jupyter the Demo.ipynb file to check how to detect and test the time performances.

## Keras Version Compatibility (convert_weights.py)

In order to allow a soft passage from Keras version to Tensorflow version, we develop a script to convert your .h5 weights file in tensorflow checkpoints. The generated checkpoints will perfectly reproduce the performances of your keras version model.

## Available dataset and prepare data scripts

This repository has been developed in the context of Second Hands European project and an associated dataset had been created:
https://github.com/alcor-lab/SecondHandsDataset
This repository also offers a sample script to prepare data for training.

## Authors

* **Edoardo Alati & Malik Bekmurat** 


## Acknowledgments

* Matterport Company for their first Keras implementation at https://github.com/matterport/Mask_RCNN


