# Automatic Registration of Point Clouds with Building Models

## Description

This repository contains the implementation of a multi-stage registration pipeline for aligning TLS point clouds with CityGML building models in real urban scenes.

The method addresses challenges such as partial overlap, surrounding clutter, structural similarity, and uneven point density. The pipeline includes pre-alignment, spatial cropping, wall-based candidate selection, and weighted point-to-plane ICP.

---
## Environment

The code was developed and tested with:

- Python 3.8
- Open3D

---

## Repository Structure


```text
.
├── main_registration.py
├── crop_utils.py
├── getvoxel.py
├── experiments/
│   ├── tuning_tum_entrance.py
│   └── window_prior_test.py
├── reports/
│   └── Jialin_Chen_finalreport.pdf
└── README.md
```
---
## Workflow

The typical workflow includes:

1. Pre-alignment of TLS point cloud and CityGML model  
2. Spatial cropping to reduce irrelevant surroundings  
3. Candidate facade generation  
4. Weighted point-to-plane ICP refinement  

---

## Usage

The main registration pipeline can be executed using:


python main_registration.py


Additional experiments can be found in the `experiments/` folder.

Before running, update MODEL_PATH and MLS_PATH in main_registration.py to point to your local input datasets.

---

## Data

The dataset used in this project consists of TLS point clouds and corresponding CityGML LoD3 building models of an urban scene (TUM main campus entrance area).

Due to their large size, the data are not included in this repository.

**Data available upon request.**

---

## Notes

- The implementation is designed for experimental validation rather than a fully packaged software.
- Some scripts share similar logic and may contain redundant parts for clarity during experimentation.
- Parameter settings (e.g., voxel size, thresholds) follow those described in the report.


---

## Project Report

The detailed methodology, experimental results, and limitations are documented in the [Final Project Report](reports/Jialin_Chen_finalreport.pdf).

*Note: The report references the original university-hosted GitLab repository. This GitHub repository provides the publicly accessible version of the project code.*


---

## Author

Jialin Chen  
Technical University of Munich
