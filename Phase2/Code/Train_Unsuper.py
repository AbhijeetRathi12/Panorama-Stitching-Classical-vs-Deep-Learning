#!/usr/bin/env python

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from Network.Network_unsuper import HomographyModel, LossFn
import os
import random
from Misc.MiscUtils import *
from Misc.DataUtils import *
import argparse
import torchvision.io as io


def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, Process):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []
    C_a = []
    P_a = []
    C_b = []
    P_b = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        if Process == "Validation":
            LabelsPath = os.path.join(BasePath, "Val_syntheticNew/H4.csv")
            TrainCoordinates = ReadLabels(LabelsPath)
            DirNamesTrain = os.path.join(BasePath, "Val_syntheticNew/PA/")
            original_warped_image_path = os.path.join(BasePath, "Val_syntheticNew/PB/")
            point_patch_2_path = os.path.join(BasePath, "Val_syntheticNew/Cb.csv")
            point_patch_1_path = os.path.join(BasePath, "Val_syntheticNew/Ca.csv")
            point_patch_2 = ReadLabels(point_patch_2_path)
            point_patch_1 = ReadLabels(point_patch_1_path)
        
        elif Process == "Train":
            LabelsPath = os.path.join(BasePath, "Train_syntheticNew/H4.csv")
            TrainCoordinates = ReadLabels(LabelsPath)
            DirNamesTrain = os.path.join(BasePath, "Train_syntheticNew/PA/")
            original_warped_image_path = os.path.join(BasePath, "Train_syntheticNew/PB/")
            point_patch_2_path = os.path.join(BasePath, "Train_syntheticNew/Cb.csv")
            point_patch_1_path = os.path.join(BasePath, "Train_syntheticNew/Ca.csv")
            point_patch_2 = ReadLabels(point_patch_2_path)
            point_patch_1 = ReadLabels(point_patch_1_path)
            
        else:
            raise ValueError(f"Invalid value for 'Process': {Process}. It should be 'Train' or 'Validation'.")
            
        selected_Image = random.choice(os.listdir(DirNamesTrain))
        
        if selected_Image in TrainCoordinates:
            original_image_path = os.path.join(DirNamesTrain, selected_Image)
            patched_image = io.read_image(original_image_path)
            original_warped_image_path = os.path.join(original_warped_image_path, selected_Image)
            warped_image = io.read_image(original_warped_image_path)
            h4pt = TrainCoordinates[selected_Image]
            C4pt_2 = point_patch_2[selected_Image]
            C4pt_1 = point_patch_1[selected_Image]
        else:
            continue
        stacked_image = torch.cat([patched_image, warped_image], axis=0)
        stacked_image = stacked_image.view(2, 128, 128).float()
        
        data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
        ])

        augmented_image = data_transform(stacked_image)
        ImageNum += 1

        I1Batch.append(torch.tensor(augmented_image))
        CoordinatesBatch.append(torch.tensor(h4pt))
        C_a.append(torch.tensor(C4pt_1))
        C_b.append(torch.tensor(C4pt_2))
        P_a.append(torch.tensor(patched_image, dtype=torch.float64))
        P_b.append(torch.tensor(warped_image, dtype=torch.float64))

    return torch.stack(I1Batch), torch.stack(CoordinatesBatch), torch.stack(C_a), torch.stack(C_b), torch.stack(P_a), torch.stack(P_b)

def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel()

    Optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    
    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")
    
    loss_vs_epoch = []
    loss_vs_iteration = []
    
    for Epochs in range(StartEpoch, NumEpochs):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in range(NumIterationsPerEpoch):
            I1Batch, CoordinatesBatch, Ca, Cb, Pa, Pb = GenerateBatch(
                BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, "Train")

            # Predict output with forward pass
            PbPredicted, H4ptPredicted = model(I1Batch, CoordinatesBatch, Ca, Cb, Pa)

            PbPredicted =  PbPredicted
            LossThisBatch = LossFn(PbPredicted, Pb)

            loss_vs_iteration.append(LossThisBatch.item())
            
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            model.eval()
            with torch.no_grad():
                validation_batch, validation_labels, VCa, VCb, VPa, VPb = GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, "Validation")
         
            result = model.validation_step(validation_batch, validation_labels, VCa, VCb, VPa, VPb)
            
            # Tensorboard
            Writer.add_scalar(
                "LossEveryIter",
                result["val_loss"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        average_epoch_loss = sum(loss_vs_iteration[-NumIterationsPerEpoch:]) / NumIterationsPerEpoch
        loss_vs_epoch.append(average_epoch_loss)
        Writer.add_scalar("LossEveryEpoch", average_epoch_loss, Epochs,)

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")
    
def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Data",
        help="Base path of images, Default:Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="CheckpointsUnsup5000/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=20,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=256,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="LogsUnsup5000/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath)
  
if __name__ == "__main__":
    main()