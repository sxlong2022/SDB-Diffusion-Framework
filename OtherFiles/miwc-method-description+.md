# Multi-temporal image weighted composition (MIWC) method - Enhanced Version

## Overview

The MIWC method performs weighted composition of multi-temporal images based on near-infrared (NIR) band information to obtain high-quality composite images. This enhanced version particularly considers seasonal variations and tidal influences in coastal areas.

## Method Rationale

1. NIR band reflectance shows strong correlation with water body features
2. Seasonal variations affect water quality and reflectance characteristics
3. Tidal changes influence coastal water body boundaries
4. Water bodies show distinct statistical patterns in NIR reflectance

## Implementation Steps

### Step 1: Negative Pixels Mask

The negative pixels of the images in the atmospheric-corrected multi-temporal remote sensing image collection C0 are masked out to obtain the image collection C1:

R_C1 = {R_i | R_i > 0}, for i = 1, 2, ..., n_C0
Where:
- R_i represents the reflectance values in image i
- n_C0 is the number of images in collection C0

### Step 2: Seasonal Water Quality Adjustment

Apply seasonal adjustment factors to account for water quality variations:

R_adj_i = R_i × S_f
Where S_f is:
- 0.9 for spring (March-May)
- 0.8 for summer (June-August)
- 1.0 for other seasons

### Step 3: Calculation of Common Segmentation Threshold

For the image collection C1:

1. Calculate detailed percentile statistics for each image:
   - P = {P5, P15, P25, P50, P75, P85, P95}
   Where P_n represents the nth percentile of NIR reflectance.

2. Calculate interquartile range (IQR):
   IQR = P75 - P25

3. Calculate tidal influence:
   TR = P95 - P25  (Tidal Range)
   TF = TR / P50   (Tidal Factor)

4. Calculate threshold for each image:
   th_i = P50 × (1 + 0.1 × TF)
   Where:
   - i is the image index
   - th_i is the threshold for image i

5. Remove outlier thresholds according to:
   T_C2 = {th_i | q_low ≤ th_i ≤ q_up}, for i = 1, 2, ..., n_C1
   Where:
   q_low = P25_T - 1.5 × (P75_T - P25_T)
   q_up = P75_T + 1.5 × (P75_T - P25_T)
   P25_T and P75_T are the 25th and 75th percentiles of all th_i

6. Take the median value th_med_C2 of T_C2 as the common segmentation threshold:
   th_med_C2 = median(T_C2)

### Step 4: NIR Limits Calculation

For each image in C2:

1. Apply water mask using common threshold:
   WM_i = {1 if R_NIR_i ≤ th_med_C2, 0 otherwise}

2. Calculate water statistics:
   Q1_i = percentile(R_NIR_i[WM_i = 1], 25)
   Q3_i = percentile(R_NIR_i[WM_i = 1], 75)
   IQR_i = Q3_i - Q1_i

3. Calculate NIR upper limit:
   R_NIR_upper_i = Q3_i + 1.5 × IQR_i

4. Final NIR limit:
   R_NIR_limit = median(R_NIR_upper_i), for i = 1, 2, ..., n_C2

### Step 5: Multi-temporal Weighted Composition

1. Weight calculation for each pixel (u,v):
   W_i(u,v) = 1 / (max(R_NIR_i(u,v), R_NIR_limit))^p
   
2. Weight normalization:
   W_norm_i(u,v) = W_i(u,v) / Σ(W_i(u,v)), for i = 1, 2, ..., n_C2
   Where Σ represents summation over all valid images

3. Final reflectance calculation:
   R_new(u,v) = Σ(W_norm_i(u,v) × R_i(u,v)), for i = 1, 2, ..., n_C2
   Where:
   - R_i(u,v) is the reflectance of pixel (u,v) in image i
   - p is the power parameter (default = 4)

### Step 6: Post-processing

1. Holes interpolation:
   For each hole pixel h(u,v):
   R_h(u,v) = mean(R_valid(u±1,v±1))
   Where R_valid represents valid neighboring pixels

2. Mean filtering:
   R_final(u,v) = Σ(R(u+i,v+j)) / 9
   Where i,j ∈ {-1,0,1}

## Key Parameters

- Power parameter (p): 4
- Seasonal adjustment factors:
  - Spring: 0.9
  - Summer: 0.8
  - Other seasons: 1.0
- Mean filter kernel size: 3×3
- Tidal influence factor: 0.1
- Outlier threshold: 1.5 × IQR

## Implementation Notes

1. Optimized for coastal areas with tidal influences
2. Considers seasonal water quality variations
3. Enhanced boundary detection for island-water interfaces
4. Statistical approach for threshold determination
5. Particularly effective for:
   - Coastal water bodies
   - Multi-temporal analysis
   - Areas with seasonal variations

## Validation and Quality Control

1. Image quality assessment:
   - Cloud coverage filtering
   - Valid pixel percentage
2. Statistical validation:
   - Seasonal distribution analysis
   - Threshold stability check
3. Results include:
   - Processing statistics
   - Seasonal distribution
   - Threshold values
   - Coverage information

## Application Scope

Particularly suitable for:
1. Coastal water body mapping
2. Island shoreline monitoring
3. Seasonal water quality assessment
4. Multi-temporal water body analysis

## Data Requirements

1. Sentinel-2 imagery (recommended)
2. NIR band (B8) availability
3. Multi-temporal coverage
4. Cloud cover < 50%

## References

1. Original MIWC method
2. Seasonal water quality assessment methods
3. Coastal water body mapping techniques
4. Multi-temporal image processing approaches
