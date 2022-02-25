# PyTorch BERT Quantization Example

Based on `https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT`

Original README [here](README_orig.md)

Modified the following files:
 * infer_glue_mrpc.py

## Overview
predict glue mrpc validation dataset through bert with faster transformer

## Setup
pip3 install -r requirements.txt

## Run
python3 infer_glue_mrpc.py

## Environment
nvcr.io/nvidia/pytorch:20.12-py3