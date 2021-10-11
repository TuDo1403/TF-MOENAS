# Training-Free Multi-Objective EvolutionaryNeural Architecture Search via Neural TangentKernel and Number of Linear Regions

[ICONIP 2021] "Training-Free Multi-Objective Evolutionary Neural Architecture Search via Neural Tangent Kernel and Number of Linear Regions" by Tu Do, Ngoc Hoang Luong

[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Ngoc Hoang Luong, Tu Do

## Installation

* Clone this repo:

```bash
git clone https://github.com/MinhTuDo/TF-MOENAS.git
cd TF-MOENAS
```

* Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 0. Prepare the NAS Benchmarks

* Follow the instructions [here](https://github.com/google-research/nasbench) to install benchmark files for NAS-Bench-101.
* Follow the instructions [here](https://github.com/D-X-Y/NATS-Bench/blob/main/README.md) to install benchmark files for NAS-Bench-201.
* **Remember to properly set the benchmark paths in config files, default data path is ~/.torch.**

### 1. Search

#### [NAS-Bench-101](https://github.com/google-research/nasbench)

```python
# Baseline MOENAS
python search.py --cfg config/baseline_moenas-101.yml --n_evals 5000 --population_size 50 --loops_if_rand 30

# Training-free MOENAS
python search.py --cfg config/tf_moenas-101.yml --n_evals 5000 --population_size 50 --loops_if_rand 30
```

#### [NAS-Bench-201](https://github.com/D-X-Y/AutoDL-Projects/blob/master/docs/NAS-Bench-201.md)

```python
# Baseline MOENAS
python search.py --cfg config/baseline_moenas-201.yml --n_evals 3000 --population_size 50 --loops_if_rand 30

# Training-free MOENAS
python search.py --cfg config/tf_moenas-201.yml --n_evals 3000 --population_size 50 --loops_if_rand 30
```

**For customized search, additional configurations can be modified through yaml config files in `config` folder.

## Acknowledgement

* Code inspired from:
[NAS-Bench-101](https://github.com/google-research/nasbench),
[NASBench-Pytorch](https://github.com/romulus0914/NASBench-PyTorch),
[NATS-Bench](https://github.com/D-X-Y/NATS-Bench),
[pymoo](https://github.com/anyoptimization/pymoo),
[TENAS](https://github.com/VITA-Group/TENAS),
[AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects)
