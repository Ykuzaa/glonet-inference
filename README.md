# GLONET On-Demand

GLONET On-Demand enables research partners to generate custom 10-day ocean forecasts. Select your forecast date, specify your S3 storage location, and launch the process.

## Prerequisites

- EDITO account with process execution permissions
- Research partner access ([contact us](mailto:glonet@mercator-ocean.eu))
- S3 bucket on EDITO MinIO storage

## Quick Start

### 1. Access the Service
**[Launch GLONET-Inference](https://datalab.dive.edito.eu/process-launcher/process-playground/glonet-inference?name=glonet-inference&version=0.0.9&s3=region-7e02ff37&resources.requests.cpu=%C2%AB7000m%C2%BB&resources.requests.memory=%C2%AB25Gi%C2%BB)**
### 2. Configure Resources ⚠️

**Critical:** Set minimum resource allocation to avoid crashes.

| Resource | Minimum Required |
|----------|-----------------|
| CPU | 7000m (7 cores) |
| Memory | 25 Gi |
| GPU | 1 NVIDIA GPU |

### 3. Configure Parameters

| Parameter | Format | Default | Example |
|-----------|--------|---------|---------|
| **FORECAST_DATE** | YYYY-MM-DD | Yesterday | 2025-01-08 |
| **S3_OUTPUT_FOLDER** | BUCKET/FOLDER | {BUCKET}/glonet-inference | oidc-ykuzaa/forecasts |
| **BUCKET_NAME** | bucket-name | Required | oidc-ykuzaa |

⚠️ Do not include `s3://` prefix in S3_OUTPUT_FOLDER

### 4. Launch

Click **Launch** and monitor execution (~13 minutes).

## Output

Results are saved to: `s3://{BUCKET_NAME}/{FOLDER_PATH}/{FORECAST_DATE}/`

```
GLONET_MOI_{START}_{END}.nc    Main forecast (10 days, 1/4° resolution)
inits/                          Initialization files (in1, in2, in3)
thumbnails/                     Preview images (zos, thetao, so, uo, vo)
```

**Variables:** Temperature, salinity, currents (u/v), sea surface height  
**Coverage:** Global ocean (80°S to 90°N, 21 depth levels)  
**File size:** ~2.5 GB (NetCDF4)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out-of-Memory crash | Increase Memory to minimum 25 Gi |
| S3 upload failed | Verify bucket name and permissions, remove s3:// prefix |
| Data fetch error | Check FORECAST_DATE has available Copernicus Marine data |
| Execution > 20 min | Wait or retry if > 30 minutes |

## Limitations

- **Forecast horizon:** Fixed 10 days
- **Execution time:** ~13 minutes per forecast
- **Valid dates:** Dates with Copernicus Marine GLO12 data availability
