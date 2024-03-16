# Circuit Generation

## Getting Started
### Install requirements
Create conda environment:
```bash
conda create --name circuit-generation
conda activate generation
```
Install pip requirements:
```bash
pip install -r requirements.txt
```

## Run Code
Data preprocessing: `MergeTest/encoder_output_data.ipynb`
Inference (currently only encoder is changed for circuit generation): `sd/demo.ipynb`


## Progress
* Set up training pipeline
* Currently using [SPICE](https://github.com/symbench/spice-datasets) dataset

Future plan:
* Set up evaluation pipeline
* Crawl circuit data from web
