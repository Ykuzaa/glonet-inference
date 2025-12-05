"""
Model synchronization utilities for GLONET inference.

This module helps synchronize model and statistics files from a remote
S3-like HTTP endpoint into a local directory.
"""

import os
from pathlib import Path
import requests
import config

MODEL_SOURCE_BUCKET = "project-glonet"
MODEL_PREFIX = "public/glonet_1_4_model/20241112/model/"


# Ensure the model files are synchronized into the given local directory.
def synchronize_model_locally(local_dir: str):
    model_s3_bucket = MODEL_SOURCE_BUCKET
    model_remote_prefix = MODEL_PREFIX

    print(f"Synchronizing from: s3://{model_s3_bucket}/{model_remote_prefix}")

    sync_s3_to_local(
        model_s3_bucket,
        model_remote_prefix,
        local_dir,
    )


def _generate_file_list():
    """Generates the list of all model and statistics files to be downloaded."""
    files = []

    # Define levels and corresponding variables
    surface_level = "L0"
    surface_vars = ["zos", "thetao", "so", "uo", "vo"]

    subsurface_levels = [
        "L50", "L100", "L150", "L222", "L318", "L380", "L450", "L540",
        "L640", "L763", "L902", "L1245", "L1684", "L2225", "L3220",
        "L3597", "L3992", "L4405", "L4833", "L5274"
    ]
    subsurface_vars = ["thetao", "so", "uo", "vo"]

    # Surface statistics files
    for var in surface_vars:
        files.extend([f"{surface_level}/{var}_mean.npy", f"{surface_level}/{var}_std.npy"])

    # Sub-surface statistics files
    for level in subsurface_levels:
        for var in subsurface_vars:
            files.extend([f"{level}/{var}_mean.npy", f"{level}/{var}_std.npy"])

    # Model, reference, and weights files
    files.extend([
        "glonet_p1.pt", "glonet_p2.pt", "glonet_p3.pt",
        "ref1.nc", "ref2.nc", "ref3.nc"
    ])

    # Regridding weights files
    all_levels = [surface_level] + subsurface_levels
    for level in all_levels:
        files.append(f"xe_weights14/{level}.nc")

    return files


# Download listed files from the remote S3-like HTTP endpoint into local_dir.
def sync_s3_to_local(bucket_name, remote_prefix, local_dir):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        s3_endpoint_url = config.AWS_S3_ENDPOINT
        
        if not s3_endpoint_url:
            raise ValueError("The environment variable AWS_S3_ENDPOINT is not set.")

        if s3_endpoint_url and not s3_endpoint_url.startswith("https://"):
            s3_endpoint_url = "https://" + s3_endpoint_url
        
        print("Downloading via direct HTTP requests.")
    except Exception as e:
        print(f"S3 endpoint configuration error: {e}")
        raise

    print(
        f"Checking/Downloading: {s3_endpoint_url}/{bucket_name}/{remote_prefix} -> {local_dir}..."
    )

    files_to_download = _generate_file_list()

    for file_name in files_to_download:
        local_file_path = local_dir / file_name
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        object_key = f"{remote_prefix}{file_name}"

        if not local_file_path.exists():
            try:
                http_url = f"{s3_endpoint_url}/{bucket_name}/{object_key}"
                print(f"Downloading: {http_url} -> {local_file_path}...")

                response = requests.get(http_url, stream=True)
                response.raise_for_status()

                with open(local_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Downloaded: {file_name}")
            except Exception as e:
                print(f"ERROR: Download failed for {file_name} from {http_url}")
                print(f"HTTP/Network error: {e}")
                raise

    print(f"Synchronization completed into {local_dir}")
