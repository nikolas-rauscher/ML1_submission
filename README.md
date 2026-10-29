# CS-433 Project 1

Predict MICHD from the official BRFSS survey data. The model uses only NumPy
and the Python standard library.

## Run

Use Python 3.11. Create an environment and install NumPy:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the [official dataset](https://github.com/epfml/ML_course/raw/70049809ec08bf1e0ac2f4c673c43939072cfe90/projects/project1/data/dataset_to_release.zip).
Extract `x_train.csv`, `y_train.csv` and `x_test.csv` into `data/raw/`.
The dataset is not part of this repository.

```bash
python run.py
```

This loads the saved model and writes `predictions.csv`. It checks the model,
test data, labels, row order and CSV checksum. Choose a new filename for another
run: `python run.py --output predictions_check.csv`.

The CSV has 109,379 rows and matches AIcrowd submission **337745**:
F1 **0.456**, accuracy **0.889**. Its SHA-256 is
`b9473759405d43acf5f04a4e10fcdf4119e1b5904f381fa8a5178363031c8032`.

## Model and data preparation

Five histogram boosting models each have a weight of 15%. A neural network
has a weight of 25%. We average their predicted probabilities and use the fixed
threshold `0.238357420897016` to choose label -1 or +1.

We remove `SEQNO`, `_PSU`, `IDATE` and `IYEAR`. Missing values have a separate
code. For `PHYSHLTH`, `MENTHLTH` and `POORHLTH`, code 88 becomes zero; codes
77 and 99 become missing. Features with up to 24 values are categorical.
Other features use up to 128 quantile bins. All bin limits and categories are
learned from training rows only. Trees select features using training labels.

The network uses one-hot categories, scaled numeric bins and missing-value
indicators. It has two ReLU layers with 64 units each, a direct linear path,
10% dropout and L2 regularization. It trains for 82 epochs with Adam.
Exact settings and random seeds are in `models/recipe.json`.

## Training and evaluation

```bash
python train.py --workers 6
python baseline.py --name ridge_check
```

Training builds the six selected models from all labelled rows. Both commands
write their results under `runs/`. Training checks the expected CSV checksum. It
takes much longer than loading the saved model. The ridge command uses
lambda 0.01 and its own stratified 80/20 development split, with seed 433.

For model selection, we used a stratified 60/20/10/10 split with seed 20261023.
The first 60% fitted models; 20% selected settings. We then fitted the fixed
recipe on 80%, chose its threshold on 10%, and evaluated once on the last 10%.
Holdout F1 was **0.433854**, compared with **0.402664** for ridge on the same rows.
These scores evaluate the 80%-trained recipe, not the final full-data weights.
The saved model was then fitted on all 328,135 labelled rows.

The supplied `HAREHAB1` feature describes rehabilitation after a heart attack.
These results classify reported diagnoses; they do not measure future risk.

## Files and checks

- `implementations.py`: the six required course methods.
- `advanced_models.py`, `fast_trees.py`, `neural_model.py`: model code.
- `ensemble.py`, `pipeline.py`, `helpers.py`: prediction and data handling.
- `run.py`, `train.py`, `baseline.py`: the three entry points.

For development checks, install `requirements-dev.txt` and run `make check`.
This runs local tests and the course tests from the official repository.
The course provided `helpers.py`.
