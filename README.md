---
language: 
  - sv
thumbnail: "https://www.uu.se/images/18.719e0b6018d874d37f14fa91/1708853093133/DigPhilBild.png"
tags:
- stanza
- historical
- swedish
- bert
- cased
- digphil
base_model: "KBLab/bert-base-swedish-cased"
library_name: stanza
---

# Retraining Stanza to optimize dependency parsing on a diachronic Swedish corpus

This repository contains forked code from the official Stanza github, with scripts that help prepare for and train models on different combinations of treebanks relevant to historical Swedish. 

## Guide

Dev/test for all models is 10/90 of our human validated gold sentences (https://github.com/alanev52/Diachronic_Treebanks_DigPhil/tree/main/parsed_data/validated)

Example work flow, training a model with pretrained vectors from kubhist2 1880 incremental and training data from Swedish and Bokmål treebanks, as well as our own machine-parsed diachronic corpus:

```
python prepare-train-val-test.py sv diachron bm

source scripts/config_alvis.sh

python -m stanza.utils.datasets.prepare_depparse_treebank UD_Swedish-diachronic --wordvec_pretrain_file /cephyr/users/cleland/Alvis/stanza_resources/sv/pretrain/diachronic.pt

python -m stanza.utils.training.run_depparse UD_Swedish-diachronic --wordvec_pretrain_file /cephyr/users/cleland/Alvis/stanza_resources/sv/pretrain/diachronic.pt --batch_size 32 --dropout 0.33
```

All of the above can be done with a single command by using:

```
./make_new_model.sh {vectors} {language codes}
```

which for the example becomes:

```
./make_new_model.sh diachronic.pt sv diachron bm
```

## Results

