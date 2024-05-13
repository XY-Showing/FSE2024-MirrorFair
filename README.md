# [FSE 2024] MirrorFair: Fixing Fairness Bugs in Machine Learning Software via Counterfactual Predictions

Welcome to the homepage of our FSE'24 paper, "MirrorFair: Fixing Fairness Bugs in Machine Learning Software via Counterfactual Predictions." The homepage contains the source code of MirrorFair, as well as the intermediate results, the installation instructions, and a replication guideline. 

## Experimental environment

We use Python 3.7 for our experiments. We use the IBM AI Fairness 360 (AIF360) toolkit to implement bias mitigation methods and compute fairness metrics. 

Installation instructions for Python 3.8 and AIF360 can be found at https://github.com/Trusted-AI/AIF360. That page provides several ways to install it. We recommend creating a virtual environment for it (as shown below), because AIF360 requires specific versions of many Python packages which may conflict with other projects on your system. If you want to try other installation methods or encounter any errors during the installation process, please refer to the page (https://github.com/Trusted-AI/AIF360) for help.

#### Conda

Conda is recommended for all configurations. [Miniconda](https://conda.io/miniconda.html)
is sufficient (see [the difference between Anaconda and
Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda)
if you are curious) if you do not already have conda installed.

Then, to create a new Python 3.8 environment, run:

```bash
conda create --name aif360 python=3.8
conda activate aif360
```

The shell should now look like `(aif360) $`. To deactivate the environment, run:

```bash
(aif360)$ conda deactivate
```

The prompt will return to `$ `.

Note: Older versions of conda may use `source activate aif360` and `source
deactivate` (`activate aif360` and `deactivate` on Windows).

### Install with `pip`

To install the latest stable version from PyPI, run:

```bash
pip install 'aif360'
```

[comment]: <> (This toolkit can be installed as follows:)

[comment]: <> (```)

[comment]: <> (pip install aif360)

[comment]: <> (```)

[comment]: <> (More information on installing AIF360 can be found on https://github.com/Trusted-AI/AIF360.)

In addition, we require the following Python packages. 
```
pip install sklearn
pip install numpy
pip install shapely
pip install matplotlib
pip install --upgrade protobuf==3.20.0
pip install fairlearn
```

## Dataset

We use the five default datasets supported by the AIF360 toolkit. **When running the scripts that invoke these datasets, you will be prompted how to download these datasets and in which folders they need to be placed.** You can also refer to https://github.com/Trusted-AI/AIF360/tree/master/aif360/data for the raw data files.

## Scripts and results
The repository contains the following folders:

* ```code/``` contains code for implementing our approach.
* ```results/``` contains the raw results of the models after applying MirrorFair. Each file in these folders has 53 columns, with the first column indicating the metric, the next 50 columns the metric values of 50 runs, and the last two columns the mean and std values of the 50 runs.

## Reproduction
You can reproduce the results from scratch. We provide a step-by-step guide on how to reproduce the  results.

We obtain the ML performance and fairness metric values obtained by our approach MAAT (`MAAT/maat.py`). `maat.py` supports three arguments: `-d` configures the dataset; `-c` configures the ML algorithm; `-p` configures the protected attribute.
```
cd code
python MirrorFair.py -d adult -c lr -p sex
python MirrorFair.py -d adult -c lr -p race
python MirrorFair.py -d compas -c lr -p sex
python MirrorFair.py -d compas -c lr -p race
python MirrorFair.py -d german -c lr -p sex
python MirrorFair.py -d german -c lr -p age
python MirrorFair.py -d bank -c lr -p age
python MirrorFair.py -d mep -c lr -p RACE

python MirrorFair.py -d adult -c rf -p sex
python MirrorFair.py -d adult -c rf -p race
python MirrorFair.py -d compas -c rf -p sex
python MirrorFair.py -d compas -c rf -p race
python MirrorFair.py -d german -c rf -p sex
python MirrorFair.py -d german -c rf -p age
python MirrorFair.py -d bank -c rf -p age
python MirrorFair.py -d mep -c rf -p RACE


python MirrorFair.py -d adult -c svm -p sex
python MirrorFair.py -d adult -c svm -p race
python MirrorFair.py -d compas -c svm -p sex
python MirrorFair.py -d compas -c svm -p race
python MirrorFair.py -d german -c svm -p sex
python MirrorFair.py -d german -c svm -p age
python MirrorFair.py -d bank -c svm -p age
python MirrorFair.py -d mep -c svm -p RACE
```

## Declaration
Thanks to the authors of existing bias mitigation methods for open source to facilitate our implementation of this paper. Therefore, when using our code or data for your work, please also consider citing their papers, including [AIF360](https://arxiv.org/abs/1810.01943), [Fairway](https://doi.org/10.1145/3368089.3409697), [Fair-SMOTE](https://doi.org/10.1145/3468264.3468537), [Fairea](https://doi.org/10.1145/3468264.3468565), [MAAT](https://doi.org/10.1145/3540250.3549093), [CARE](https://doi.org/10.1145/3510003.3510080) and [Empirical Study](https://doi.org/10.1145/3583561).



