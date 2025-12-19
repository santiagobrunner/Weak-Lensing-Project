# Weak Lensing Convergence Estimation using Maximum Mean Discrepancy

A statistical framework for inferring convergence fields from simulated galaxy size data using kernel-based distribution testing and regression methods.

## Overview

This project develops a non-parametric statistical approach to detect and quantify distributional shifts in large-scale datasets. Using Maximum Mean Discrepancy (MMD) as a distance metric in reproducing kernel Hilbert spaces (RKHS), we construct robust estimators that achieve strong predictive performance on 31 million observations across 1,248 spatial regions.

**Key Achievement**: Built linear estimators from kernel-based distribution distances with R² ≈ 0.91 and 75% correlation with ground truth, demonstrating the effectiveness of kernel methods for signal extraction in noisy, large-scale data.

## Statistical Methodology

### Maximum Mean Discrepancy (MMD)

The MMD quantifies the distance between two probability distributions P and Q by comparing their embeddings in a reproducing kernel Hilbert space:

```
MMD²(P,Q) = ||μ_P - μ_Q||²_H
```

where μ_P and μ_Q are kernel mean embeddings. For finite samples, the empirical MMD is computed as:

```
MMD²(P,Q) = (1/n²)Σ k(x_i,x_j) + (1/m²)Σ k(y_i,y_j) - (2/nm)Σ k(x_i,y_j)
```

This provides a characteristic kernel distance that is zero if and only if the distributions are identical.

### Kernel Design and Selection

Three kernel functions were implemented and rigorously compared:

1. **Linear Kernel**: `k(x,y) = x·y`
   - Reduces MMD² to squared difference of means
   - Baseline for detecting first-moment shifts
   - Validated against previous work

2. **RBF Kernel**: `k(x,y) = exp(-||x-y||²/2σ²)`
   - Captures non-linear distributional changes
   - Sensitive to higher-order moments
   - Achieved R² = 0.91 for the constructed estimator

3. **Directed RBF Kernel**: `k(x,y) = sgn(x-y) · exp(-||x-y||²/2σ²)`
   - Custom asymmetric kernel encoding directional information
   - Intrinsically captures sign of distributional shift
   - Not positive semi-definite but empirically effective (R² = 0.92)

### Two-Sample Distribution Testing

For each spatial region (pixel), we perform hypothesis testing:
- **Null hypothesis H₀**: Local distribution = Global distribution
- **Test statistic**: MMD² between local sample (n=20,000) and global sample (m=20,000)
- **Noise threshold**: Established via bootstrap (15,000 iterations): μ = 2.93×10⁻⁶, σ = 4.16×10⁻⁶ (for linear kernel)

Regions with MMD² exceeding the noise threshold indicate significant distributional shifts.

### Regression and Estimator Construction

**Weighted Least Squares Regression:**

Due to heteroscedastic residuals (variance increases with signal strength), we binned the κ² values and applied weighted regression:

- **Weights**: w_i = 1/σ²_i where σ_i is the standard error within each bin
- **Model**: MMD² = m·κ² + b
- **Optimization**: Minimize weighted residual sum of squares

**Results:**

| Kernel | Slope (m) | Intercept (b) | R² |
|--------|-----------|---------------|-----|
| RBF | 0.288 ± 0.013 | (1.61 ± 0.25)×10⁻⁶ | 0.91 |
| Directed RBF | 104.0 ± 11.0 | (-2.72 ± 0.31)×10⁻³ | 0.92 |

### Sign Assignment Strategy

Since MMD² is non-negative, we developed two approaches for sign recovery:

1. **Mean-Shift Method**: Sign determined by comparing distribution means (74.4% accuracy)
2. **Intrinsic Method**: Directed RBF kernel captures sign naturally (71.4% accuracy)

Both methods successfully identify the direction of distributional shifts in >70% of cases, with errors concentrated in low-signal regions where |κ| ≈ 0.

## Results and Performance

### Pixel-Level Statistics (1,145 regions)

| Metric | RBF Kernel | Directed RBF |
|--------|-----------|--------------|
| Pearson Correlation (r) | 0.745 | 0.747 |
| R² (regression fit) | 0.91 | 0.92 |
| RMSE | 6.0×10⁻³ | 6.2×10⁻³ |
| Mean Residual | -5.4×10⁻⁴ | 1.3×10⁻⁴ |
| Std. Residual | 6.0×10⁻³ | 6.2×10⁻³ |
| Sign Accuracy | 74.4% | 71.4% |

### Statistical Validation

**Distribution Moments:**

| Statistic | True | RBF | Directed RBF |
|-----------|------|-----|--------------|
| Mean | -6.7×10⁻⁴ | -1.2×10⁻³ | -5.4×10⁻⁴ |
| Std. Dev. | 7.1×10⁻³ | 8.9×10⁻³ | 9.3×10⁻³ |
| Skewness | 0.12 | 0.65 | 0.24 |
| Kurtosis | 0.07 | 0.51 | -1.3 |

**Key Findings:**
- Unbiased estimators (mean ≈ 0)
- 20-30% higher scatter than ground truth
- Small deviations from Gaussianity in higher-order moments

### Power Spectrum Recovery

- **Small scales (ℓ > 40)**: Good agreement with ground truth.
- **Large scales (ℓ < 40)**: Detection threshold leads to more pixels with similar convergence values, thereby enhancing correlation and power, respectively.
- Pseudo power spectrum computed using `anafast` with partial sky coverage corrections

## Data Pipeline

### Input Data
- **Sample size**: 31 million observations
- **Sky coverage**: 1,000 deg²
- **Spatial regions**: 1,145 pixels (HEALPix nside=64 with Minimum Galaxy Constraint)
- **Regional sample size**: 20,000 observations per region for MMD computation

### Quality Control
- **Galaxy size threshold**: 5 arcsec (removes outliers)
- **Minimum galaxies per pixel**: 20,000 (ensures statistical significance)
- **Bootstrap validation**: 15,000 iterations for noise characterization

### Computational Efficiency
- Vectorized MMD computation using NumPy
- Stratified sampling for balanced regional comparisons
- Efficient kernel matrix operations for large-scale analysis

## Model Limitations and Considerations

1. **Detection Threshold**: MMD cannot reliably detect very small distributional shifts (|κ| < 5×10⁻³), creating an artificial gap near zero in predictions and creating additional artificional correlation between pixels.

2. **Variance Inflation**: Predicted maps show 20-30% higher scatter than ground truth, indicating model uncertainty beyond sampling noise.

3. **Scale Limitations**: Power spectrum recovery limited at largest scales due to cosmic variance and detection excess power from detection thereshold.

4. **Non-Gaussianity**: Elevated skewness and kurtosis in predictions suggest residual systematic effects.

## Key Statistical Innovations

- **Kernel-based distribution testing** at scale (31M observations, 1,145 regions)
- **Custom asymmetric kernel design** for intrinsic sign encoding
- **Weighted regression framework** addressing heteroscedastic residuals
- **Multi-scale validation**: pixel-level, distributional, and power spectrum analysis

## Repository Structure

```
.
├── mmd2_*                        # Computed MMD² values for different kernels and maps
├── code.ipynb                    # Full analysis pipeline and visualizations
├── compute_MMDs.py               # Core MMD computation script to run on Euler Cluster
├── cl_kappa_mean_225.txt         # Input power spectrum data
├── pseudo_cl.txt                 # Pseudo power spectrum data for sky patch
├── kappa_*                       # Average convergence per region for for different kernels and maps
└── README.md                     # This file
```
## Data
Our main dataset consists of 31 million galaxies simulated by the GalSBI software. Due to the size of this dataset (~7GB) it was not uploaded to this repository.

The remaining data used are:
- Power spectrum used to create convergence maps (cl_kappa_mean_225.txt)
- Corresponding pseudo power spectrum (pseudo_cl.txt) for the observed sky patch
- Precomputed MMD² values stored in .npy format for reproducibility

## Citation

```
Santiago Brunner (2025)
"An MMD Approach to Inferring Weak-Lensing Convergence from Galaxy Sizes"
Semester Project, ETH Zurich, Cosmology Group
Supervised by Prof. Dr. Alexandre Refregier and Veronika Oehl
```

## Contact

**Santiago Brunner**  
Email: santiago.brunner@yahoo.com  
LinkedIn: [linkedin.com/in/santiagobrunner](https://linkedin.com/in/santiagobrunner)  
GitHub: [github.com/santiagobrunner](https://github.com/santiagobrunner)

---


