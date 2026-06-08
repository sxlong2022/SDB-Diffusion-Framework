# SDB-Diffusion-Framework

A Robust Computational Framework for Satellite-Derived Bathymetry Integrating Machine Learning with Spatio-Temporal Generative Diffusion Models.

## Overview

This framework implements a modular two-stage pipeline that combines Random Forest regression with Spatio-Temporal Generative Diffusion Models (S³GM) for satellite-derived bathymetry mapping. It achieves R²=0.941, RMSE=3.09 m with only 11% ground-truth coverage in complex coastal waters.

## Highlights

- Modular framework integrates Random Forest with diffusion models for bathymetry
- Achieves R²=0.941, RMSE=3.09 m with only 11% ground-truth coverage
- Subtle conditioning weight (α=0.1) essential for stable diffusion sampling
- Consumer GPU processes annual dataset in approximately 12 minutes
- Framework transferable to other geospatial inverse problems

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with ≥6GB VRAM (tested on RTX 2060)
- **RAM**: ≥8GB system memory
- **Storage**: ~2GB for code and dependencies (excluding datasets)

### Software Requirements
- **Operating System**: Linux, Windows, or macOS
- **Python**: 3.9
- **CUDA**: 12.4 (for GPU acceleration)
- **Conda**: Recommended for environment management

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/sxlong2022/SDB-Diffusion-Framework.git
cd SDB-Diffusion-Framework
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate sdb-diffusion
```

The `environment.yml` file includes all required dependencies:
- PyTorch ≥2.0.1
- scikit-learn ≥1.6
- NumPy, SciPy
- Google Earth Engine API
- Other supporting libraries

### Step 3: Verify Installation

Run the quick test to verify your installation:

```bash
python tests/quick_test.py
```

If successful, you should see output confirming that all modules load correctly and basic functionality works.

## Project Structure

```
├── run_bathymetry.py              # Main entry point for the pipeline
├── data_acquisition_preprocessing.py  # Data acquisition (Sentinel-2, GEBCO)
├── miwc.py                        # Multi-temporal Image Weighted Composition
├── environment.yml                # Conda environment specification
├── tests/
│   └── quick_test.py              # Quick verification test
├── bathymetry/                    # Core bathymetry modules
│   ├── __init__.py
│   ├── main.py                    # Hybrid system main class
│   ├── classic_models.py          # Random Forest model
│   ├── preprocessor.py            # Data preprocessing utilities
│   ├── s3gm_wrapper.py            # S³GM wrapper
│   ├── s3gm_config.py             # S³GM configuration
│   ├── gpu_memory.py              # GPU memory management
│   └── utils.py                   # Utility functions
├── configs/                       # Configuration files
│   ├── classic_models.yaml        # RF hyperparameters
│   └── s3gm_default.yaml          # S³GM hyperparameters
└── S3GM/Code/                     # S³GM model (adapted from Li et al., 2024)
    ├── models/                    # U-Net video architecture
    ├── sampler/                   # VP-SDE sampling
    └── trainer/                   # Training utilities
```

## Quick Start

### Running the Quick Test

To verify your installation and understand basic usage:

```bash
python tests/quick_test.py
```

This test will:
1. Verify all required modules can be imported
2. Check GPU availability (optional but recommended)
3. Test basic Random Forest functionality with synthetic data
4. Validate configuration file loading
5. Confirm S³GM wrapper initialization

**Expected output**: All tests should pass with "✓" marks. If GPU is not available, a warning will be shown but the test will continue.

### Full Pipeline Execution

The framework operates in six stages:
#### Stage 1: Data Preprocessing

Acquires and preprocesses Sentinel-2 imagery using MIWC method:

```bash
python run_bathymetry.py --stage 1
```

**Before running**: Update the Google Earth Engine project ID placeholder in `run_bathymetry.py`, `data_acquisition_preprocessing.py`, and `bathymetry/preprocessor.py`:

```python
ee.Initialize(project='YOUR_GEE_PROJECT_ID')
```

**Inputs**: 
- Sentinel-2 imagery (via Google Earth Engine)
- GEBCO bathymetry grids
- Nautical chart data (XYZ format)

**Outputs**: 
- Preprocessed Sentinel-2 composites
- Aligned GEBCO grids
- Chart data mapped to computational grid

#### Stage 1.5: Random Forest Training

Trains the RF model on preprocessed optical data:

```bash
python run_bathymetry.py --stage 1.5
```

**Inputs**: Preprocessed Sentinel-2 features, nautical chart ground truth

**Outputs**: 
- Trained RF model (`rf_model.pkl`)
- Initial bathymetry estimates (`rf_bathymetry_*.npy`)

#### Stage 2: S³GM Processing

Refines bathymetry using the diffusion model:

```bash
python run_bathymetry.py --stage 2
```

**Inputs**: RF bathymetry, GEBCO data, sparse chart measurements

**Outputs**: 
- Pre-trained S³GM model (one-time, ~8 hours)
- Refined bathymetry time series (`s3gm_bathymetry_*.npy`)

**Note**: Pre-training is a one-time cost. Subsequent runs only require conditional sampling (~12 min/year).

#### Stage 3: Post-processing and Visualization

Denormalizes results and generates visualizations:

```bash
python run_bathymetry.py --stage 3
```

**Outputs**: 
- Final bathymetry maps (GeoTIFF format)
- Validation metrics (R², RMSE, MAE)
- Visualization plots

#### Stage 1.8: Classic RF Model Validation

Validates the trained RF model against held-out nautical chart measurements:

```bash
python run_bathymetry.py --stage 1.8
```

**Outputs**:
- Validation metrics (R², RMSE, MAE) for the RF model
- Scatter plots comparing RF predictions vs. ground truth

#### Stage 4: Statistical Significance Analysis

Performs Wilcoxon signed-rank tests to evaluate statistical significance of model improvements:

```bash
python run_bathymetry.py --stage 4
```

**Outputs**:
- p-values and significance results for pairwise model comparisons
- Statistical summary tables

#### Stage 5: Zoning Performance Analysis

Conducts detailed spatial analysis of model performance across depth zones and geographic regions:

```bash
python run_bathymetry.py --stage 5
```

**Outputs**:
- Per-zone performance metrics (R², RMSE, MAE)
- Spatial performance maps and zone-specific visualizations

### Running All Stages

To execute the complete pipeline:

```bash
python run_bathymetry.py --stage all
```

## Configuration

The framework uses YAML configuration files for reproducibility:

### Random Forest Configuration (`configs/classic_models.yaml`)

```yaml
rf_params:
  model_params:
    n_estimators: 500        # Number of trees
    max_depth: 25           # Maximum depth
    max_features: 'sqrt'    # Feature selection method
    random_state: 42        # Random seed
    n_jobs: -1             # Parallel processing

  feature_engineering:
    normalization:
      use_standard_scaling: true
    band_ratio:
      use_log_transform: true
```

### S³GM Configuration (`configs/s3gm_default.yaml`)

Key hyperparameters:

```yaml
# SDE parameters
beta_min: 0.1
beta_max: 1000  # Critical for stability

# Sampling parameters (data fidelity and spatial smoothness)
sampling:
  alpha: 0.1       # Data fidelity weight (gentle guidance)
  gamma_spatial: 5.0e-7  # Spatial smoothness weight
  num_steps: 3
  snr: 0.01
```

**Important**: The subtle conditioning weight (α=0.1) is essential for stable convergence. Higher values may cause gradient explosions.

## Data Requirements

### Required Datasets

1. **Sentinel-2 Imagery**
   - Source: Google Earth Engine (`COPERNICUS/S2_HARMONIZED`)
   - Access: Free, requires GEE account
   - Coverage: 2018-2023 (or your study period)

2. **GEBCO Bathymetry**
   - Source: https://www.gebco.net/
   - Format: NetCDF
   - Resolution: 15 arc-second

3. **Nautical Chart Data**
   - Format: XYZ (longitude, latitude, depth)
   - Datum: LAT (Lowest Astronomical Tide)
   - Source: Local maritime authority

4. **JRC Global Surface Water** (optional, for visualization)
   - Source: Google Earth Engine (`JRC/GSW1_4/GlobalSurfaceWater`)
   - Purpose: Land/water mask overlay

### Data Preparation

Place your data in the following structure:

```
data/
├── sentinel2/          # GEE will download here
├── gebco/
│   ├── GEBCO_2019.nc
│   ├── GEBCO_2020.nc
│   └── ...
└── nautical_charts/
    └── chart_data.xyz
```

## Computational Performance

Benchmarks on NVIDIA RTX 2060 (6GB VRAM):

| Stage | Time | GPU Memory | Notes |
|-------|------|------------|-------|
| Data Preprocessing | ~5 min | N/A | CPU-bound |
| RF Training | ~12 sec | N/A | CPU-bound |
| RF Prediction (6 years) | ~19 sec | N/A | CPU-bound |
| S³GM Pre-training | ~8 hours | 4.2 GB | One-time cost |
| S³GM Sampling | ~12 min/year | 4.2 GB | Per annual dataset |
| **Total Inference** | **~12.5 min/year** | **4.2 GB** | Excluding pre-training |

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce batch size in `configs/s3gm_default.yaml`
- Use smaller grid resolution (e.g., 32×32 instead of 64×64)
- Enable gradient checkpointing

**2. GEE Authentication Error**
```bash
earthengine authenticate
```

**3. Import Errors**
- Verify conda environment is activated: `conda activate sdb-diffusion`
- Reinstall dependencies: `conda env update -f environment.yml`

**4. Sampling Instability**
- Ensure α=0.1 (gentle guidance is critical)
- Verify β_max=1000 in configuration
- Check input data normalization range [-1, 1]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{song2026sdb,
  title={A Robust Computational Framework for Satellite-Derived Bathymetry Integrating Machine Learning with Spatio-Temporal Generative Diffusion Models},
  author={Song, Xiaolong and Liu, Boliang and Xiao, Zhong and Xu, Haijue and Bai, Yuchuan},
  journal={Computers \& Geosciences},
  year={2026},
  note={Under Review}
}
```

## Acknowledgements

This implementation builds upon the S³GM framework by Li et al. (2024):

```bibtex
@article{li2024learning,
  title={Learning spatiotemporal dynamics with a pretrained generative model},
  author={Li, Zeyu and Han, Weiwei and Zhang, Yicheng and Lin, Tao},
  journal={Nature Machine Intelligence},
  volume={6},
  number={12},
  pages={1566--1579},
  year={2024}
}
```

Original S³GM repository: https://github.com/lzy12301/S3GM

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues:
- **GitHub Issues**: https://github.com/sxlong2022/SDB-Diffusion-Framework/issues
- **Email**: xlsong@tju.edu.cn

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Version History

- **v1.0.0** (2026): Initial release
  - Random Forest + S³GM hybrid framework
  - MIWC preprocessing
  - Conditional sampling with guided diffusion
  - Configuration-driven execution

## References

See the paper for complete references. Key dependencies:
- PyTorch: https://pytorch.org/
- scikit-learn: https://scikit-learn.org/
- Google Earth Engine: https://earthengine.google.com/
- GEBCO: https://www.gebco.net/
