# GLONET with Custom Initialization Data

GLONET accepts custom oceanographic data for forecast initialization. Instead of automatically fetching data from Copernicus Marine, you can provide your own initialization files from alternative sources or scenarios.

## Prerequisites

- EDITO account with process execution permissions
- Research partner access ([contact us](mailto:glonet@mercator-ocean.eu))
- S3 bucket with your custom initialization files
- 3 NetCDF files: **in1.nc**, **in2.nc**, **in3.nc**

## Quick Start

### 1. Access the Service
Navigate to **Process** → Search **"Glonet-inference"** → Click **Launch**

### 2. Configure Resources

Same requirements as standard GLONET:

| Resource | Minimum Required |
|----------|-----------------|
| CPU | 7000m (7 cores) |
| Memory | 25 Gi |
| GPU | 1 NVIDIA GPU |

### 3. Configure Parameters

| Parameter | Format | Required | Example |
|-----------|--------|----------|---------|
| **FORECAST_DATE** | YYYY-MM-DD | Yes | 2025-01-08 |
| **S3_OUTPUT_FOLDER** | BUCKET/FOLDER | Yes | oidc-ykuzaa/forecasts |
| **BUCKET_NAME** | bucket-name | Yes | oidc-ykuzaa |
| **INIT_FILE_1_URL** | s3://bucket/path/in1.nc | Yes | s3://bucket/custom-data/in1.nc |
| **INIT_FILE_2_URL** | s3://bucket/path/in2.nc | Yes | s3://bucket/custom-data/in2.nc |
| **INIT_FILE_3_URL** | s3://bucket/path/in3.nc | Yes | s3://bucket/custom-data/in3.nc |

### 4. Launch

Click **Launch** and monitor execution (~5 minutes).

⚡ **Performance:** Custom init data reduces execution time from ~13 min to ~5 min (no Copernicus fetch).

## Data Requirements

### File Structure Overview

You must provide **3 separate NetCDF files** corresponding to different ocean depth layers:

| File | Depth Coverage | Variables | Notes |
|------|---------------|-----------|-------|
| **in1.nc** | Surface (0.49m) | zos, thetao, so, uo, vo | 5 channels |
| **in2.nc** | Intermediate (50-763m) | thetao, so, uo, vo | 4 channels × 10 levels |
| **in3.nc** | Deep (902-5274m) | thetao, so, uo, vo | 4 channels × 10 levels |

### Detailed Specifications

#### IN1 - Surface Layer

**Variables:**
- `zos`: Sea surface height above geoid (m)
- `thetao`: Sea water potential temperature (°C)
- `so`: Salinity (PSU × 10⁻³)
- `uo`: Eastward velocity (m/s)
- `vo`: Northward velocity (m/s)

**Dimensions:**
```python
time: 2      # 2 consecutive days
ch: 5        # 5 channels (zos + thetao + so + uo + vo)
lat: 672     # -78°N to 90°N at 0.25° resolution
lon: 1440    # -180°E to 180°E at 0.25° resolution
```

**Data array shape:** `(2, 5, 672, 1440)`

**Depth level:** 0.49m

#### IN2 - Intermediate Depths

**Variables:**
- `thetao`: Sea water potential temperature (°C)
- `so`: Salinity (PSU × 10⁻³)
- `uo`: Eastward velocity (m/s)
- `vo`: Northward velocity (m/s)

**Dimensions:**
```python
time: 2       # 2 consecutive days
ch: 40        # 4 variables × 10 depth levels
lat: 672      # -78°N to 90°N at 0.25° resolution
lon: 1440     # -180°E to 180°E at 0.25° resolution
```

**Data array shape:** `(2, 40, 672, 1440)`

**Depth levels:** 50, 100, 150, 222, 318, 380, 450, 540, 640, 763 (meters)

**Channel order:** thetao[0-9], so[10-19], uo[20-29], vo[30-39]

#### IN3 - Deep Ocean

**Variables:**
- `thetao`: Sea water potential temperature (°C)
- `so`: Salinity (PSU × 10⁻³)
- `uo`: Eastward velocity (m/s)
- `vo`: Northward velocity (m/s)

**Dimensions:**
```python
time: 2       # 2 consecutive days
ch: 40        # 4 variables × 10 depth levels
lat: 672      # -78°N to 90°N at 0.25° resolution
lon: 1440     # -180°E to 180°E at 0.25° resolution
```

**Data array shape:** `(2, 40, 672, 1440)`

**Depth levels:** 902, 1245, 1684, 2225, 3220, 3597, 3992, 4405, 4833, 5274 (meters)

**Channel order:** thetao[0-9], so[10-19], uo[20-29], vo[30-39]

### Critical Requirements

✅ **Mandatory:**
- NetCDF4 format with **h5netcdf** engine compatibility
- 0.25° spatial resolution (1/4 degree)
- Geographic coverage: -78°N to 90°N, -180°E to 180°E
- 2 timesteps (consecutive days before FORECAST_DATE)
- NaN values for land/missing data
- Variable names exactly as specified

❌ **Common Errors:**
- Single combined file instead of 3 separate files
- Wrong dimensions or channel ordering
- Missing `zos` variable in in1.nc
- Incorrect spatial resolution
- Single timestep instead of 2


## Output

Results are saved to: `s3://{BUCKET_NAME}/{FOLDER_PATH}/{FORECAST_DATE}/`

Same output structure as standard GLONET:
- Main forecast NetCDF (10 days)
- Thumbnails for visualization
- Copy of initialization files used

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Invalid dimensions" | Verify shapes: in1=(2,5,672,1440), in2/in3=(2,40,672,1440) |
| "Missing zos variable" | in1.nc must include sea surface height as first channel |
| "Channel mismatch" | Check variable ordering: in2/in3 must be [thetao, so, uo, vo] |
| "h5netcdf error" | Save files with: `ds.to_netcdf(file, engine="h5netcdf")` |
| S3 access denied | Verify init files are in accessible bucket with correct permissions |

## Use Cases

**Scenario Analysis:**
- Climate change impact studies with modified temperature/salinity profiles
- Extreme event simulations (hurricanes, marine heatwaves)
- Data assimilation experiments

**Alternative Data Sources:**
- Regional ocean models
- Satellite-derived observations
- Ensemble forecast members

**Research Applications:**
- Sensitivity analysis to initial conditions
- Downscaling from global to regional models
- Historical reconstruction with custom data

## Limitations

- Custom init data bypasses Copernicus Marine validation
- User responsible for data quality and physical consistency
- No automatic regridding (must be 0.25° resolution)
- Forecast quality depends on initialization accuracy
