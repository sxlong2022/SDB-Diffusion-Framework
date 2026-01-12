# SDB-Diffusion-Framework

A Robust Computational Framework for Satellite-Derived Bathymetry Integrating Machine Learning with Spatio-Temporal Generative Diffusion Models.

## Highlights

- Modular framework integrates Random Forest with diffusion models for bathymetry
- Achieves R²=0.941, RMSE=3.09 m with only 11% ground-truth coverage
- Subtle conditioning weight (α=0.1) essential for stable diffusion sampling
- Consumer GPU processes annual dataset in approximately 12 minutes
- Framework transferable to other geospatial inverse problems

## Project Structure

```
├── run_bathymetry.py              # Main entry point
├── data_acquisition_preprocessing.py  # Data acquisition (Sentinel-2, GEBCO)
├── miwc.py                        # Multi-temporal Image Weighted Composition
├── environment.yml                # Conda environment
├── bathymetry/                    # Core bathymetry modules
│   ├── classic_models.py          # Random Forest model
│   ├── preprocessor.py            # Data preprocessing
│   ├── s3gm_wrapper.py            # S³GM wrapper
│   ├── s3gm_config.py             # S³GM configuration
│   ├── main.py                    # Hybrid system main class
│   ├── gpu_memory.py              # GPU memory management
│   └── utils.py                   # Utility functions
├── configs/                       # Configuration files
│   ├── classic_models.yaml
│   └── s3gm_default.yaml
└── S3GM/Code/                     # S³GM model (adapted from Li et al., 2024)
    ├── models/                    # U-Net video architecture
    ├── sampler/                   # VP-SDE sampling
    └── trainer/                   # Training utilities
```

## Requirements

- Python 3.9
- PyTorch ≥2.0.1
- CUDA 12.4
- NVIDIA GPU with ≥6GB VRAM

## Installation

```bash
conda env create -f environment.yml
conda activate bathymetry
```

## Usage

```bash
# Stage 1: Data preprocessing
python run_bathymetry.py --stage 1

# Stage 1.5: RF model training
python run_bathymetry.py --stage 1.5

# Stage 2: S³GM processing
python run_bathymetry.py --stage 2

# Stage 3: Post-processing and visualization
python run_bathymetry.py --stage 3
```

## Configuration

Before running, update `data_acquisition_preprocessing.py` with your GEE project ID:
```python
ee.Initialize(project='YOUR_GEE_PROJECT_ID')
```

## Data Requirements

- **Sentinel-2**: Accessed via Google Earth Engine
- **GEBCO**: Download from https://www.gebco.net/
- **Nautical charts**: Contact local maritime authority

## Citation

If you use this code, please cite:

```bibtex
@article{song2025sdb,
  title={A Robust Computational Framework for Satellite-Derived Bathymetry Integrating Machine Learning with Spatio-Temporal Generative Diffusion Models},
  author={Song, Xiaolong and Liu, Boliang and Xiao, Zhong and Xu, Haijue and Bai, Yuchuan},
  journal={Computers \& Geosciences},
  year={2025},
  note={Under Review}
}
```

## Acknowledgements

This implementation builds upon the S³GM framework by [Li et al. (2024)](https://github.com/lzy12301/S3GM).

## License

MIT License
