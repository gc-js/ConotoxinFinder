# ConotoxinFinder
## Description
In this study, we aimed to discover and engineer novel conopeptides targeting the α7 receptor by integrating deep learning, electrophysiology, AlphaFold-multimer/3, MD simulations, cryo-EM, and chemical modification approaches. Initially, a peptide screening model was built using the BERT-based protein pre-trained language model ESM-2, incorporating a classification head and a regression head. This led to the identification of a novel peptide, termed SS1, from a collection of 689 non-disulfide-rich conopeptides. Subsequently, a model of the peptide/receptor complex was constructed, a structure-activity relationship (SAR) study and structure optimization were performed on SS1. Ultimately, we elucidated the unique action mechanism of the [S8R]SS1 mutant, through combined cryo-EM and computational modeling. The workflow that we established for peptide ligands screening and their action mechanism study provide important structural and molecular basis for the development of potential drug leads or neurochemical tools.
![barchart](https://github.com/gc-js/ConotoxinFinder/blob/main/img/model.png)
## ![Huggingface](https://img.shields.io/badge/Hugging%20Face-Spaces-brightgreen)
We also host a trained version of the model on the [HuggingFace Spaces](https://huggingface.co/spaces/oucgc1996/ConotoxinFinder), so you can start your inference using just your browser.

## fine-tuning
We fine-tuning the  encoder layers of ESM-2 with an MLM task specifically with  conopeptides. 
```shell
python fine-tuning.py
```

## 1. nAChRs and non-nAChRs classification model
## Training
```shell
python classification_train.py
```
## Inference

The trained version of this model can be downloaded from [here](https://huggingface.co/spaces/oucgc1996/ConotoxinFinder/resolve/main/best_model.pth?download=true)

```shell
python classification_pre.py
```
## 2. nAChRs α7 regression model
## Training
```shell
python regression_train.py
```
## Inference

The trained version of this model can be downloaded from [here](https://huggingface.co/spaces/oucgc1996/ConotoxinFinder-regression/resolve/main/best_model.pth?download=true)

```pshell
python regression_pre.py 
```
