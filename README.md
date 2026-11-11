# CellTarget

CellTarget is a computational framework for inferring cellular objective functions from experimental flux data using convex optimisation under partial observability. This work has been published in NPJ Systems Biology & Applications: https://www.nature.com/articles/s41540-026-00700-8 
The framework is applicable to both reduced metabolic models and genome-scale models.


## Repository structure

- **`inputs/`**  
  Input data files (experimental fluxes, flux bounds, and model-specific mappings).

- **`bo/`**  
  Bayesian optimisation utilities used for multi-start initialisation and hyperparameter exploration.

- **`full_observability.py/`**  
  CellTarget implementation for a small toy model under full observability. These scripts internally load the required data, execute the optimisation, and write the outputs to disk without additional user input.

- **`partial_observability_cho/`**  
  CellTarget implementation for reduced CHO metabolic models under partial observability. This file expects:

- `--condition` : Experimental condition (e.g. `early`, `stat`, `AWZ11expo`, `AWZ11stat`)
- `--output_dir` : Directory where results will be saved
- `--fluxes_mode` : Objective definition (`standard`, `extracellular`, `measured`, or `all`)

- **`partial_observability_gem/`**  
  CellTarget implementation for genome-scale metabolic models under partial observability.This file expects:

- `--condition` : Experimental condition (e.g. `early`, `stat`, `AWZ11expo`, `AWZ11stat`)
- `--output_dir` : Directory where results will be saved
- `--fluxes_mode` : Objective definition (`standard`, `extracellular`, `measured`, or `all`)

- **`benchmark_cho/`**  
  Benchmark implementation of ObjFind*  used for benchmarking CellTarget. This file expects:

- `--fluxes_mode` : Objective definition (`standard`, `extracellular`, `measured`, or `all`) 

- **`plot_publication/`**  
  Scripts used to generate publication-quality figures.

- **`setup.py`**  
  Further definitions

- **`README.md`**  
  Project overview (this file).

##  Requirements

To reproduce the environment:
```bash
conda env create -f environment.yml
conda activate <env_name>
