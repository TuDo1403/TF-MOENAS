# Training-Free Multi-Objective Evolutionary Neural Architecture Search via Neural Tangent Kernel and Number of Linear Regions

[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Ngoc Hoang Luong, Tu Do

In ICONIP 2021.

## Installation

- Clone this repo:

```bash
git clone https://github.com/MinhTuDo/TF-MOENAS.git
cd TF-MOENAS
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 0. Prepare the NAS Benchmarks

- Follow the instructions [here](https://github.com/google-research/nasbench) to install benchmark files for NAS-Bench-101.
- Follow the instructions [here](https://github.com/D-X-Y/NATS-Bench/blob/main/README.md) to install benchmark files for NAS-Bench-201.
- Optional: To evaluate IGD on the optimal front during a NAS run, for NAS-Bench-101, you need to download the pre-computed benchmark query data [here](https://drive.google.com/file/d/1s3uQkDuHtZQWSKLWMQ3Ikrik4BbJ3ajl/view?usp=sharing) and put it in the `data` folder.
- **Remember to properly set the benchmark paths in config files, default data path is ~/.torch.**

### 1. Search

#### [NAS-Bench-101](https://github.com/google-research/nasbench)

```shell
# Baseline MOENAS
python search.py -cfg config/baseline_moenas-101.yml --n_evals 5000 --pop_size 50 --loops_if_rand 30 -sw --use_archive

# Training-free MOENAS
python search.py -cfg config/tf_moenas-101.yml --n_evals 5000 --pop_size 50 --loops_if_rand 30 -sw --use_archive
```

#### [NAS-Bench-201](https://github.com/D-X-Y/AutoDL-Projects/blob/master/docs/NAS-Bench-201.md)

```shell
# Baseline MOENAS
python search.py -cfg config/baseline_moenas-201.yml --n_evals 3000 --pop_size 50 --loops_if_rand 30 -sw --use_archive

# Training-free MOENAS
python search.py -cfg config/tf_moenas-201.yml --n_evals 3000 --pop_size 50 --loops_if_rand 30 -sw --use_archive
```

To evaluate IGD score on pre-computed optimal front during the search, simply provide `--eval_igd` flag.

For customized search, additional configurations can be modified through yaml config files in `config` folder.

## Acknowledgement

Code inspired from:

- [NASBench: A Neural Architecture Search Dataset and Benchmark](https://github.com/google-research/nasbench),
- [NASBench-Pytorch](https://github.com/romulus0914/NASBench-PyTorch),
- [NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size](https://github.com/D-X-Y/NATS-Bench),
- [Pymoo: Multi-Objective Optimization in Python](https://github.com/anyoptimization/pymoo),
- [Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective](https://github.com/VITA-Group/TENAS),
- [Automated Deep Learning Projects (AutoDL-Projects)](https://github.com/D-X-Y/AutoDL-Projects)
