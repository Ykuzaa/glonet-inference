"""
Run GLONET inference process.

This script orchestrates:
- model synchronization,
- generation/upload of initial condition files,
- running model inference to assemble forecasts,
- uploading the final NetCDF to S3.
"""
from datetime import datetime, timedelta
import boto3
import os
import io
import sys
from glonet_forecast import create_forecast
from model import synchronize_model_locally
from get_inits import generate_initial_data

# Default parameters
LOCAL_MODEL_DIR = "/tmp/glonet"


# Main function: orchestrates the whole inference pipeline
def main():
    print("=" * 60)
    print("GLONET Inference - EDITO Process")
    print("=" * 60)

    # Read date from environment variable (defaults to yesterday)
    default_date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    forecast_date_str_input = os.environ.get("FORECAST_DATE", default_date_str)

    if forecast_date_str_input:
        forecast_date_str = forecast_date_str_input
    else:
        forecast_date_str = default_date_str
        print(f"No DATE specified or empty. Using default (yesterday): {forecast_date_str}")

    try:
        FORECAST_DATE = datetime.strptime(forecast_date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"Error: Environment variable FORECAST_DATE ('{forecast_date_str}') is not in YYYY-MM-DD format")
        sys.exit(1)

    print(f"Forecast date: {FORECAST_DATE}")

    # Read S3 output folder from environment
    s3_output_folder_input = os.environ.get("S3_OUTPUT_FOLDER")

    if s3_output_folder_input:
        S3_OUTPUT_FOLDER = s3_output_folder_input
        print(f"Custom S3 path used: {S3_OUTPUT_FOLDER}")
    else:
        print("No custom S3 path provided. Building default path...")

        user_s3_bucket = os.environ.get("AWS_BUCKET_NAME")

        if not user_s3_bucket:
            print("Fatal error: BUCKET_NAME not found in environment.")
            print("Process cannot determine where to save results.")
            sys.exit(1)

        S3_OUTPUT_FOLDER = f"{user_s3_bucket}/glonet-inference"
        print(f"Default S3 path used: {S3_OUTPUT_FOLDER}")

    print(f"\n{'=' * 60}")
    print("Starting inference...")
    print(f"{'=' * 60}\n")

    # Download models
    print("--- Downloading models ---")
    synchronize_model_locally(LOCAL_MODEL_DIR)

    # Compute forecast period: forecast starts tomorrow and ends in 10 days
    forecast_start = FORECAST_DATE + timedelta(days=1)
    forecast_end = FORECAST_DATE + timedelta(days=10)

    # Build full S3 URL for the final NetCDF file
    s3_output_with_date = f"{S3_OUTPUT_FOLDER}/{FORECAST_DATE}"

    forecast_netcdf_url = (
        f"https://minio.dive.edito.eu/{s3_output_with_date}/"
        f"GLONET_MOI_{forecast_start}_{forecast_end}.nc"
    )

    # Generate initial data: from Copernicus Marine to S3 (in bucket)
    print("--- Initial data ---")

    # Split the S3_OUTPUT_FOLDER into bucket and remaining path
    s3_path_parts = S3_OUTPUT_FOLDER.split("/", 1)
    bucket_name = s3_path_parts[0]
    folder_path = s3_path_parts[1] if len(s3_path_parts) > 1 else ""

    in1_url, in2_url, in3_url = generate_initial_data(
        bucket_name=bucket_name,
        forecast_netcdf_file_url=forecast_netcdf_url,
    )
    print(f"in1: {in1_url}")
    print(f"in2: {in2_url}")
    print(f"in3: {in3_url}")

    # Run inference
    print("--- Inference ---")
    dataset = create_forecast(
        forecast_netcdf_file_url=forecast_netcdf_url,
        model_dir=LOCAL_MODEL_DIR,
        initial_file_1_url=in1_url,
        initial_file_2_url=in2_url,
        initial_file_3_url=in3_url,
    )

    try:
        # Final file name
        file_name = f"GLONET_MOI_{forecast_start}_{forecast_end}.nc"
        file_key = f"{folder_path}/{FORECAST_DATE}/{file_name}"
        file_key = file_key.lstrip("/")
        print("--- Saving results to S3 (Streaming) ---")
        print("Converting dataset to NetCDF (in-memory)...")

        # Convert dataset to NetCDF in memory
        object_bytes = dataset.to_netcdf()

        # Ensure bytes type
        if isinstance(object_bytes, memoryview):
            object_bytes = bytes(object_bytes)

        print(f"In-memory size: {len(object_bytes) / 1e6:.2f} MB")

        # Create an in-memory file from these bytes
        in_memory_file = io.BytesIO(object_bytes)

        # Create S3 client (uses environment variables)
        s3_endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
        if s3_endpoint_url and not s3_endpoint_url.startswith("https://"):
            s3_endpoint_url = "https://" + s3_endpoint_url

        s3_client = boto3.client(
            "s3",
            endpoint_url=s3_endpoint_url,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ["AWS_SESSION_TOKEN"],
        )

        print("Streaming upload (multipart) to S3...")
        print(f"  Bucket: {bucket_name}")
        print(f"  Path: {file_key}")

        s3_client.upload_fileobj(
            in_memory_file,
            bucket_name,
            file_key,
        )

        result_url = f"s3://{bucket_name}/{file_key}"
        print(f"Results saved: {result_url}\n")

    except Exception as e:
        print(f"Error during upload: {e}")
        raise

    # Final summary
    print("=" * 60)
    print("Inference completed successfully!")
    print(f"  Date : {FORECAST_DATE}")
    print(f"  Results : {result_url}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
