#!/usr/bin/env python

import cv2
import os
import sys
import numpy as np
import argparse
from Network.Network import HomographyModel
from tqdm import tqdm
import torch
from torchvision import transforms
import csv


# Don't generate pyc codes
sys.dont_write_bytecode = True

def StandardizeInputs(Img):

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    Img = test_transform(Img)
    return Img

def ReadLabelsTest(LabelsPathTest):
    if not (os.path.isfile(LabelsPathTest)):
        print("ERROR: Train Labels do not exist in " + LabelsPathTest)
        sys.exit()
    labels_test = {}

    with open(LabelsPathTest, 'r') as labels_file:
        csv_reader = csv.reader(labels_file)
        for row in csv_reader:
            image_name = row[0]
            labels = [float(label) for label in row[1:]]
            labels_test[image_name] = labels

    return labels_test

def TestOperation(H4pt_batch, Patched_batch_torch, ModelPath, LabelsPathPred):

    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """

    model = HomographyModel()

    CheckPoint = torch.load(ModelPath, map_location=torch.device('cpu'))
    model.load_state_dict(CheckPoint["model_state_dict"])
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )
    model.eval()
    OutSaveT = open(LabelsPathPred, "w")
    H4ptPred_list = []
    batch_size = 1
    for j in tqdm(range(0,len(Patched_batch_torch), batch_size)):
        
        Img = Patched_batch_torch[j:j+batch_size]
        PredH = model(Img).detach().numpy()
        OutSaveT.write(str(PredH) + "\n")
        H4ptPred_list.append(PredH)
        del Img
        
    OutSaveT.close()
    
    ESE_list = []
    for i in range(len(H4pt_batch)):
        x = H4ptPred_list[i][0]
        y = H4pt_batch[i]
        ESE_error = np.linalg.norm(x - y, ord=2)
        ESE_list.append(ESE_error)

    Avg_L2_error = np.mean(ESE_list)
    print("ESE Error: ",Avg_L2_error) 
    
    return  H4ptPred_list

def VisuaizePatch(PredH4pt_list, Ca_list, Cb_list, Ia_list):
    
    for i in range(len(PredH4pt_list)):
        PredH4pt = PredH4pt_list[i][0]
        Ca_corner = Ca_list[i]
        Cb_corner = np.array(Cb_list[i], dtype=np.int32).reshape((-1, 4, 2))
        Ia_img = Ia_list[i]
        Cb_Pred = (PredH4pt + Ca_corner).astype(np.int32).reshape((-1, 4, 2))

                
        for center in Cb_corner:
            for k in center:
                cv2.circle(Ia_img, (int(k[0]), int(k[1])), 2, (0, 255, 0), 2) 
                
        for center in Cb_Pred:
            for l in center:
                cv2.circle(Ia_img, (int(l[0]), int(l[1])), 2, (255, 0, 0), 2)
                
        for corners in Cb_corner:
            connections = [(0, 1), (1, 3), (3, 2), (2, 0)]

            for start_idx, end_idx in connections:
                start_point = tuple(corners[start_idx])
                end_point = tuple(corners[end_idx])
                cv2.line(Ia_img, start_point, end_point, (0, 255, 0), 2)
                
        for corners in Cb_Pred:
            connections = [(0, 1), (1, 3), (3, 2), (2, 0)]

            for start_idx, end_idx in connections:
                start_point = tuple(corners[start_idx])
                end_point = tuple(corners[end_idx])
                cv2.line(Ia_img, start_point, end_point, (255, 0, 0), 2)

        cv2.imwrite(f"Phase2/Results/Supervised_test/Image{i}.png", Ia_img)

def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="Phase2/Code/CheckpointsSup/19model.ckpt",
        help="Path to load latest model from",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="Phase2/Data/Test_synthetic/",
        help="Path to load images from",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="Phase2/Data/Test_synthetic/H4.csv",
        help="Path of labels file",
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath
    
    Ia = os.path.join(BasePath, "IA/")
    Patch_a = os.path.join(BasePath, "PA/")
    Patch_b = os.path.join(BasePath, "PB/")
    Corner_a = os.path.join(BasePath, "Ca.csv")
    Corner_b = os.path.join(BasePath, "Cb.csv")
    H4_labels = ReadLabelsTest(LabelsPath)
    Ca_lables = ReadLabelsTest(Corner_a)
    Cb_lables = ReadLabelsTest(Corner_b)
    
    H4pt_list = []
    Patched_batch = []
    Ia_list = []
    Ca_list = []
    Cb_list = []
    
    for filename in os.listdir(Patch_a):
        Image1 = os.path.join(Patch_a, filename)
        Image2 = os.path.join(Patch_b, filename)
        Image_Ia = os.path.join(Ia, filename)
        patch_1 = cv2.imread(Image1, cv2.IMREAD_GRAYSCALE)
        patch_2 = cv2.imread(Image2, cv2.IMREAD_GRAYSCALE)
        Ia_img = cv2.imread(Image_Ia)
        stacked_image = torch.cat([torch.from_numpy(patch_1), torch.from_numpy(patch_2)], axis=0)
        stacked_image = stacked_image.view(2, 128, 128).float()
        stacked_image = StandardizeInputs(stacked_image)
        H4pt = H4_labels[filename]
        Ca = Ca_lables[filename]
        Cb = Cb_lables[filename]
        
        H4pt_list.append(H4pt)
        Ia_list.append(Ia_img)
        Ca_list.append(Ca)
        Cb_list.append(Cb)
        Patched_batch.append(torch.tensor(stacked_image))
        
    Patched_batch_torch = torch.stack(Patched_batch)
    
    LabelsPathPred = "Phase2/Code/TxtFiles/PredOutSup.txt"  # Path to save predicted labels
    H4ptPred_list = TestOperation(H4pt_list, Patched_batch_torch, ModelPath, LabelsPathPred)

    VisuaizePatch(H4ptPred_list, Ca_list, Cb_list, Ia_list)

if __name__ == "__main__":
    main()