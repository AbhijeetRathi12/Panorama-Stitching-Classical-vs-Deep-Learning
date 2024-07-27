# Phase 2: Panoramic Image Stitching using Deep Learning

## Overview

This phase of the project focuses on using deep learning techniques to estimate homography for panoramic image stitching. We utilize a Convolutional Neural Network (CNN) for both supervised and unsupervised learning approaches. The goal is to predict the homography transformation matrix that aligns two images into a seamless panorama. Also, using this Homography, images are stitched.

## Table of Contents

- [Requirements](#Requirements)
- [Usage](#Usage)
- [Results](#results)

## Requirements

- PyTorch
- Torchvision
- Tensorboard
- Random
- Argparse
- Numpy
- Pandas
- OpenCV-Python
- Pytorch Lightning
- Tqdm

You can install these packages using pip:

```bash
pip install torch torchvision tensorboard numpy pandas opencv-python pytorch-lightning tqdm
```
## Usage

* Clone the repository:

```bash
git clone https://github.com/AbhijeetRathi12/Panorama-Stitching-Classical-vs-Deep-Learning.git
cd Panorama-Stitching-Classical-vs-Deep-Learning-main
```

* Initially patches need to be generated for Unsupervised Learning. For that run `Data_Generation.py` file.

```bash
python Phase2/Code/Data_Generation.py
```

* To train the model, run the following command:

For Supervised Model:
```bash
python Phase2/Code/Train_Sup.py --BasePath=<path_to_data> --CheckPointPath=<path_to_checkpoint> --LogsPath=<path_to_logs> --NumEpochs=<number_of_epochs> --MiniBatchSize=<batch_size> --DivTrain=<div_train_factor> --LoadCheckPoint=<load_checkpoint_flag>
```
For Unsupervised Model:
```bash
python Phase2/Code/Train_Unsuper.py --BasePath=<path_to_data> --CheckPointPath=<path_to_checkpoint> --LogsPath=<path_to_logs> --NumEpochs=<number_of_epochs> --MiniBatchSize=<batch_size> --DivTrain=<div_train_factor> --LoadCheckPoint=<load_checkpoint_flag>
```

... __Arguments__
..* --BasePath: Base path of images (default: ../Data)
..* --CheckPointPath: Path to save checkpoints (default: Phase2/Code/Checkpoints/)
..* --LogsPath: Path to save logs for TensorBoard (default: Phase2/Code/Logs/)
..* --NumEpochs: Number of epochs to train (default: 50)
..* --MiniBatchSize: Size of the mini-batch to use (default: 64)
..* --DivTrain: Factor to reduce train data by per epoch (default: 1)
..* --LoadCheckPoint: Load model from latest checkpoint (default: 0)


* Testing: To test the model, run the 'Test.py' script:

For Supervised Model:
```bash
python Phase2/Code/Test_Sup.py --ModelPath 'Phase2/Code/CheckpointsSup/19model.ckpt' --BasePath Phase2/Data/Test_synthetic/ --LabelsPath 'Phase2/Data/Test_synthetic/H4.csv'
```
For Unsupervised Model:
```bash
python Phase2/Code/Test_Unsuper.py --ModelPath 'Phase2/Code/CheckpointsUnsuper/19model.ckpt' --BasePath Phase2/Data/Test_synthetic/ --LabelsPath 'Phase2/Code/TxtFiles/LabelsTest.txt'
```

... __Arguments__
..* --ModelPath: Path to load the latest model from
..* --BasePath: Path to load images from 
..* --LabelsPath: Path of labels file

* To use this trained model in image stitching, run `Wrapper_Sup.py` or `Wrapper_Unsuper.py`
For Supervised Model:
```bash
python Phase2/Code/Wrapper_Sup.py
```
For Unsupervised Model:
```bash
python Phase2/Code/Wrapper_Unsuper.py
```

## Results
The training script saves the model checkpoints and TensorBoard logs in the specified directories.
The testing script outputs the patches on the images. These visualization are saved as an image in the 'Phase2/Results' directory.
The Wrapper script saves the stitched images in 'Phase2/Results'.
