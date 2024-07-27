#  Image Stitching Classical Approach


## Table of Contents

- [Overview](#Overview)
- [Requirements](#Requirements)
- [Usage](#usage)


## Overview
This project focuses on stitching multiple images together to create a panorama. It involves several key steps: corner detection, adaptive non-maximal suppression (ANMS), feature descriptor extraction, feature matching, homography estimation using RANSAC, and final image warping and blending.


## Requirements

To run this script, you need Python 3 and the following Python packages:
- `numpy`
- `opencv-python`
- `SciKit-Image`


You can install these packages using pip:

```bash
pip install numpy opencv-python scikit-image
```

## Usage
* Clone the repository:

```bash
git clone https://github.com/AbhijeetRathi12/Panorama-Stitching-Classical-vs-Deep-Learning.git
cd Panorama-Stitching-Classical-vs-Deep-Learning-main
```

* Replace the path for variable "folder_path" in line 380 with the desired path of the images for the input.

* Run the script:

```bash
python Phase1/Code/Wrapper.py
```

* Intermediate results and the final panorama will be saved in the "Phase1/Results" folder.


