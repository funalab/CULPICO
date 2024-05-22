# UnsupDomaAda_for_SemaSeg_of_Cell_Images

## Overview

## Configurations

- OS: Ubuntu 18.04.6
- Language: python 3.8.5

## Environments

### venv
```shell
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip3 install -r requirements.txt
```

## Train and Test for demo

#### Train
```shell
python src/train.py --dataset-dir ./LIVECell_dataset --output-dir result/trial/train --source shsy5y --target mcf7
```
#### Test
```shell
python src/test.py --dataset-dir ./LIVECell_dataset --checkpoint result/trial/train/checkpoint --output-dir result/trial/test --inference_cell mcf7
```