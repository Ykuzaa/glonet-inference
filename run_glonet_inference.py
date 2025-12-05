"""
Run GLONET inference process.

This script orchestrates:
- model synchronization,
- generation/upload of initial condition files,
- running model inference to assemble forecasts,
- uploading the final NetCDF to S3.
"""
import os
import io
import sys
from glonet_forecast import create_forecast
from model import synchronize_model_locally
from get_inits import generate_initial_data
from generate_thumbnails import generate_thumbnail
from datetime import datetime, timedelta 
from s3_upload import get_s3_client, get_s3_endpoint_url_with_protocol
import config

# Default parameters
LOCAL_MODEL_DIR = "/tmp/glonet"

def _parse_config():
    """Parses configuration from environment variables."""
    # Split the S3_OUTPUT_FOLDER into bucket and remaining path
    s3_path_parts = config.S3_OUTPUT_FOLDER.split("/", 1)
    bucket_name = s3_path_parts[0]
    folder_path = s3_path_parts[1] if len(s3_path_parts) > 1 else ""

    return config.FORECAST_DATE, bucket_name, folder_path

def _upload_forecast_to_s3(dataset, bucket_name, file_key):
    """Converts a dataset to NetCDF in memory and uploads it to S3."""
    try:
        print("--- Saving results to S3 (Streaming) ---")
        print("Converting dataset to NetCDF (in-memory)...")

        # Convert dataset to NetCDF in memory
        object_bytes = dataset.to_netcdf()
        print(f"In-memory size: {len(object_bytes) / 1e6:.2f} MB")

        # Create an in-memory file from these bytes
        in_memory_file = io.BytesIO(object_bytes)

        s3_client = get_s3_client()

        print("Streaming upload (multipart) to S3...")
        print(f"  Bucket: {bucket_name}")
        print(f"  Path: {file_key}")

        s3_client.upload_fileobj(
            in_memory_file,
            bucket_name,
            file_key,
        )

    except Exception as e:
        print(f"Error during upload: {e}")
        raise

# Main function: orchestrates the whole inference pipeline
def main():
    print("=" * 60)
    print("GLONET Inference - EDITO Process")
    print("=" * 60)

    forecast_date, bucket_name, folder_path = _parse_config()
    print(f"Forecast date: {forecast_date}")
    print(f"Destination Bucket: {bucket_name}")
    print(f"Destination Path: {folder_path}")

    print(f"\n{'=' * 60}\nStarting inference...\n{'=' * 60}\n")

    # 1. Download models
    print("--- Downloading models ---")
    synchronize_model_locally(LOCAL_MODEL_DIR)

    # 2. Define paths and URLs
    forecast_start = forecast_date + timedelta(days=1)
    forecast_end = forecast_date + timedelta(days=10)
    s3_endpoint_url = get_s3_endpoint_url_with_protocol()
    
    file_name = f"GLONET_MOI_{forecast_start}_{forecast_end}.nc"
    final_file_key = f"{folder_path}/{forecast_date}/{file_name}".lstrip("/")
    forecast_netcdf_url = f"{s3_endpoint_url}/{bucket_name}/{final_file_key}"

    # 3. Generate initial data (from Copernicus Marine to S3)
    print("\n--- Initial data ---")
    in1_url, in2_url, in3_url = generate_initial_data(
        bucket_name=bucket_name,
        forecast_netcdf_file_url=forecast_netcdf_url,
    )
    print(f"in1: {in1_url}\nin2: {in2_url}\nin3: {in3_url}")

    # 4. Run inference
    print("\n--- Inference ---")
    dataset = create_forecast(
        forecast_netcdf_file_url=forecast_netcdf_url,
        model_dir=LOCAL_MODEL_DIR,
        initial_file_1_url=in1_url,
        initial_file_2_url=in2_url,
        initial_file_3_url=in3_url,
    )

    # 5. Generate thumbnails
    print("\n--- Generating Thumbnails ---")
    dynamic_img_path = f"{folder_path}/{forecast_date}/thumbnails".lstrip("/")
    thumbnail_urls = {
        var: f"{s3_endpoint_url}/{bucket_name}/{dynamic_img_path}/{var}.png"
        for var in ["zos", "thetao", "so", "uo", "vo"]
    }
    generate_thumbnail(
        bucket_name=bucket_name,
        forecast_netcdf_file_url=forecast_netcdf_url,
        thumbnail_file_urls=thumbnail_urls,
        forecast_dataset=dataset
    )

    # 6. Upload final forecast file
    _upload_forecast_to_s3(dataset, bucket_name, final_file_key)

    # Final summary
    print("=" * 60)
    print("Inference completed successfully!")
    print(f"  Date : {forecast_date}")
    print(f"  Results : s3://{bucket_name}/{final_file_key}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR in main execution: {e}", file=sys.stderr)
        sys.exit(1)
