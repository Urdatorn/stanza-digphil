# Retraining Stanza to optimize dependency parsing on a diachronic Swedish corpus

Experiments (most of which "learning experiences", to put it euphemistically) in training a new Swedish Stanza model that performs better on historical texts. The repository contains forked code from a development branch of the Stanza github, with scripts that help prepare for and train models on different combinations of treebanks relevant to historical Swedish. The main project repo is found [here](https://github.com/alanev52/Diachronic_Treebanks_DigPhil) and the actual trained models are found on [Hugging Face](https://huggingface.co/al1808th/stanza-digphil).

## Guide

The pipeline for preparing a train/dev/test split and training a new model is contained in the script:

```
./make_new_model.sh
```

To change the composition of the training data, modify `prepare-train-val-test-silver.py`. 

Models are saved in `saved_models`.

To run inference with a model, see `digphil_inference_pipeline.py`. 


![digphil](https://www.uu.se/images/18.719e0b6018d874d37f14fa91/1708853093133/DigPhilBild.png)

