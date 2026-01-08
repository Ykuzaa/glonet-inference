from datetime import datetime, timedelta
import boto3
import os
import io
import sys
from glonet_forecast import create_forecast
from model import synchronize_model_locally
from generate_thumbnails import generate_thumbnail

LOCAL_MODEL_DIR = "/tmp/glonet"

def main():
    print("=" * 60)
    print("GLONET Inference - EDITO Process (Custom Init Data)")
    print("=" * 60)

    default_date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    forecast_date_str = os.environ.get("FORECAST_DATE", default_date_str)

    try:
        FORECAST_DATE = datetime.strptime(forecast_date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"Error: FORECAST_DATE ('{forecast_date_str}') must be YYYY-MM-DD format")
        sys.exit(1)

    print(f"Forecast date: {FORECAST_DATE}")

    in1_url = os.environ.get("INIT_FILE_1_URL")
    in2_url = os.environ.get("INIT_FILE_2_URL")
    in3_url = os.environ.get("INIT_FILE_3_URL")

    if not in1_url or not in2_url or not in3_url:
        print("Error: INIT_FILE_1_URL, INIT_FILE_2_URL, INIT_FILE_3_URL required")
        sys.exit(1)

    print(f"in1: {in1_url}")
    print(f"in2: {in2_url}")
    print(f"in3: {in3_url}")

    s3_output_folder = os.environ.get("S3_OUTPUT_FOLDER")
    if not s3_output_folder:
        bucket_name = os.environ.get("AWS_BUCKET_NAME")
        if not bucket_name:
            print("Error: AWS_BUCKET_NAME or S3_OUTPUT_FOLDER required")
            sys.exit(1)
        s3_output_folder = f"{bucket_name}/glonet-inference"

    print(f"Output folder: {s3_output_folder}")

    print("\n--- Downloading models ---")
    synchronize_model_locally(LOCAL_MODEL_DIR)

    forecast_start = FORECAST_DATE + timedelta(days=1)
    forecast_end = FORECAST_DATE + timedelta(days=10)

    s3_output_with_date = f"{s3_output_folder}/{FORECAST_DATE}"
    forecast_netcdf_url = (
        f"https://minio.dive.edito.eu/{s3_output_with_date}/"
        f"GLONET_MOI_{forecast_start}_{forecast_end}.nc"
    )

    print("\n--- Running inference ---")
    dataset = create_forecast(
        forecast_netcdf_file_url=forecast_netcdf_url,
        model_dir=LOCAL_MODEL_DIR,
        initial_file_1_url=in1_url,
        initial_file_2_url=in2_url,
        initial_file_3_url=in3_url,
    )

    s3_path_parts = s3_output_folder.split("/", 1)
    bucket_name = s3_path_parts[0]
    folder_path = s3_path_parts[1] if len(s3_path_parts) > 1 else ""

    print("\n--- Generating thumbnails ---")
    try:
        dynamic_img_path = f"glonet-inference/{FORECAST_DATE}/thumbnails"
        thumbnail_urls = {
            var: f"s3://{bucket_name}/{dynamic_img_path}/{var}.png"
            for var in ["zos", "thetao", "so", "uo", "vo"]
        }
        generate_thumbnail(
            bucket_name=bucket_name,
            forecast_netcdf_file_url=forecast_netcdf_url,
            thumbnail_file_urls=thumbnail_urls,
            forecast_dataset=dataset
        )
        print("Thumbnails generated")
    except Exception as e:
        print(f"Thumbnail generation failed: {e}")

    file_name = f"GLONET_MOI_{forecast_start}_{forecast_end}.nc"
    file_key = f"{folder_path}/{FORECAST_DATE}/{file_name}".lstrip("/")

    print("\n--- Saving to S3 ---")
    object_bytes = dataset.to_netcdf()
    if isinstance(object_bytes, memoryview):
        object_bytes = bytes(object_bytes)

    print(f"Size: {len(object_bytes) / 1e6:.2f} MB")

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

    s3_client.upload_fileobj(io.BytesIO(object_bytes), bucket_name, file_key)

    result_url = f"s3://{bucket_name}/{file_key}"
    print(f"Results saved: {result_url}")

    print("=" * 60)
    print("Inference completed")
    print(f"  Date: {FORECAST_DATE}")
    print(f"  Results: {result_url}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)