# README

## summary
  pytorch implement for PointNet(point cloud segmentation)

## dataset
  2018 BDCI point cloud segmentation dataset

## requirement:
 - python3.6
 - pytorch0.3.0

## difference between my implementation and original paper
 - add focal loss in training for dataset classes imbalance
 - add 2 fc layers for better information fusion

## usage
 - train.py for model training
 - evaluate.py for finally test
 - IOU.py for calculating average IOU to estimate model performance
