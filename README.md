# ConotoxinFinder

## ![Huggingface](https://img.shields.io/badge/Hugging%20Face-Spaces-brightgreen)
We also host a trained version of the model on the [HuggingFace Spaces](https://huggingface.co/spaces/oucgc1996/ConotoxinFinder), so you can start your inference using just your browser.

## 1. nAChRs and non-nAChRs classification model
## Training
```shell
python gan.py
```
## Inference
```shell
python gan_generation.py
```
## 2. nAChRs α7 regression model
## Training
```shell
python AMP_Classification.py
```
## Inference
```pshell
python AMP_Classification_Prediction.py 
```
