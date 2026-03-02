# Introduction
This file contains two scripts: `run_model.py` and `run_model.sh`. These run the benchmarking experiment and run evaluation (metric collection) on one model, respectively.

## Useage: `run_model.py`
From the root of this repository:
```
python3 src/run_model.py -m <model_name>
```

User can omit the `-m` flag and corresponding argument to pull the model from the `.env` file in the root directory. An example is given as the file `env` there.

## `run_eval.py`
The evaluation script requires the following environment variables:

GT_DIR: path of the directory where ground truths for the comic images.

VLM_DIR: path of the directory where VLM outputs are stored.

From the root of this repository:
```
python3 src/run_eval.py
```

This script pulls arguments entirely from `.env`.
