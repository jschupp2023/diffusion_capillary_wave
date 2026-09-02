# CapillaryWaveTurbulence

This repository contains the codebase for the work:

> **Stochastic Operator Inference for reduced-order modeling of capillary wave turbulence using experimental measurements.**

The code is based on and extends the methodology introduced in:

**[1]** M. A. Freitag, J. M. Nicolaus, and M. Redmann (2025).  
[*Learning Stochastic Reduced Models from Data: A Nonintrusive Approach.*](https://epubs.siam.org/doi/full/10.1137/24M1679756)  
*SIAM Journal on Scientific Computing*, 47(5), A2851–A2880.

<details><summary>BibTex</summary><pre>
@article{freitag2025learning,
  title={Learning stochastic reduced models from data: A nonintrusive approach},
  author={Freitag, MA and Nicolaus, JM and Redmann, M},
  journal={SIAM Journal on Scientific Computing},
  volume={47},
  number={5},
  pages={A2851--A2880},
  year={2025},
  publisher={SIAM}
}</pre></details>


The experimental data used in this work are publicly available through:

**[2]** L. Zhang, H. Kim, B. Kramer, and J. Friend (2026).  
[*Water Surface Data. In Experimental Datasets of Capillary Wave Turbulence in a Micro-Scale System.*](https://doi.org/10.6075/J0K07581)  
UC San Diego Library Digital Collections.

<details><summary>BibTex</summary><pre>
@misc{capillary_wave_datasets,
  author = {Zhang, Lei and Kim, Hyeonghun and Kramer, Boris and Friend, James},
  title = {Water Surface Data. {I}n Experimental Datasets of Capillary Wave Turbulence in a Micro-Scale System},
  year = {2026},
  note = {UC San Diego Library Digital Collections},
  doi = {10.6075/J0K07581}
}</pre></details>

The Python codes are implemented in Python 3.10.

## Saving reduced dynamics

Use `save_reduced_data.py` to compute the uncentered POD used in `main.ipynb`
without constructing the full concatenated snapshot matrix in memory. The
default rank is 23:

```bash
export PATH=~/miniforge3/envs/capillarywave/bin:$PATH
python save_reduced_data.py 0p10
```

To select another rank or output file:

```bash
python save_reduced_data.py 0p10 --rank 30 --output reduced_0p10_r30.h5
```

The default file is
`reduced_data/<power>/reduced_dynamics_<power>_r<rank>.h5`. Each repeated
experiment is preserved as one uninterrupted trajectory—there is no
segmentation. The file contains the reduced state in `(mode, experiment,
time)` order, the POD basis and full singular-value spectrum, the spatial and
time grids, and the source label for every experiment. Reduced coordinates use
compressed `float32` storage by default; pass `--dtype float64` when storage
size matters less than retaining double precision.
