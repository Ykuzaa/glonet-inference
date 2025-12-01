"""
Model synchronization utilities for GLONET inference.

This module helps synchronize model and statistics files from a remote
S3-like HTTP endpoint into a local directory.
"""

import os
from pathlib import Path
import requests

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


# Download listed files from the remote S3-like HTTP endpoint into local_dir.
def sync_s3_to_local(bucket_name, remote_prefix, local_dir):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        s3_endpoint_url = os.environ.get("S3_ENDPOINT")
        if not s3_endpoint_url:
            raise ValueError("The environment variable S3_ENDPOINT is not set.")

        s3_endpoint_clean = s3_endpoint_url.replace("https://", "").replace(
            "http://", ""
        ).rstrip("/")
        s3_base_url = f"https://{s3_endpoint_clean}"
        print("Downloading via direct HTTP requests.")
    except Exception as e:
        print(f"S3 endpoint configuration error: {e}")
        raise

    print(
        f"Checking/Downloading: {s3_base_url}/{bucket_name}/{remote_prefix} -> {local_dir}..."
    )

    files_to_download = [
        "L0/zos_mean.npy", "L0/zos_std.npy", "L0/thetao_mean.npy", "L0/thetao_std.npy",
        "L0/so_mean.npy", "L0/so_std.npy", "L0/uo_mean.npy", "L0/uo_std.npy",
        "L0/vo_mean.npy", "L0/vo_std.npy",
        "L50/thetao_mean.npy", "L50/thetao_std.npy", "L50/so_mean.npy", "L50/so_std.npy",
        "L50/uo_mean.npy", "L50/uo_std.npy", "L50/vo_mean.npy", "L50/vo_std.npy",
        "L100/thetao_mean.npy", "L100/thetao_std.npy", "L100/so_mean.npy", "L100/so_std.npy",
        "L100/uo_mean.npy", "L100/uo_std.npy", "L100/vo_mean.npy", "L100/vo_std.npy",
        "L150/thetao_mean.npy", "L150/thetao_std.npy", "L150/so_mean.npy", "L150/so_std.npy",
        "L150/uo_mean.npy", "L150/uo_std.npy", "L150/vo_mean.npy", "L150/vo_std.npy",
        "L222/thetao_mean.npy", "L222/thetao_std.npy", "L222/so_mean.npy", "L222/so_std.npy",
        "L222/uo_mean.npy", "L222/uo_std.npy", "L222/vo_mean.npy", "L222/vo_std.npy",
        "L318/thetao_mean.npy", "L318/thetao_std.npy", "L318/so_mean.npy", "L318/so_std.npy",
        "L318/uo_mean.npy", "L318/uo_std.npy", "L318/vo_mean.npy", "L318/vo_std.npy",
        "L380/thetao_mean.npy", "L380/thetao_std.npy", "L380/so_mean.npy", "L380/so_std.npy",
        "L380/uo_mean.npy", "L380/uo_std.npy", "L380/vo_mean.npy", "L380/vo_std.npy",
        "L450/thetao_mean.npy", "L450/thetao_std.npy", "L450/so_mean.npy", "L450/so_std.npy",
        "L450/uo_mean.npy", "L450/uo_std.npy", "L450/vo_mean.npy", "L450/vo_std.npy",
        "L540/thetao_mean.npy", "L540/thetao_std.npy", "L540/so_mean.npy", "L540/so_std.npy",
        "L540/uo_mean.npy", "L540/uo_std.npy", "L540/vo_mean.npy", "L540/vo_std.npy",
        "L640/thetao_mean.npy", "L640/thetao_std.npy", "L640/so_mean.npy", "L640/so_std.npy",
        "L640/uo_mean.npy", "L640/uo_std.npy", "L640/vo_mean.npy", "L640/vo_std.npy",
        "L763/thetao_mean.npy", "L763/thetao_std.npy", "L763/so_mean.npy", "L763/so_std.npy",
        "L763/uo_mean.npy", "L763/uo_std.npy", "L763/vo_mean.npy", "L763/vo_std.npy",
        "L902/thetao_mean.npy", "L902/thetao_std.npy", "L902/so_mean.npy", "L902/so_std.npy",
        "L902/uo_mean.npy", "L902/uo_std.npy", "L902/vo_mean.npy", "L902/vo_std.npy",
        "L1245/thetao_mean.npy", "L1245/thetao_std.npy", "L1245/so_mean.npy", "L1245/so_std.npy",
        "L1245/uo_mean.npy", "L1245/uo_std.npy", "L1245/vo_mean.npy", "L1245/vo_std.npy",
        "L1684/thetao_mean.npy", "L1684/thetao_std.npy", "L1684/so_mean.npy", "L1684/so_std.npy",
        "L1684/uo_mean.npy", "L1684/uo_std.npy", "L1684/vo_mean.npy", "L1684/vo_std.npy",
        "L2225/thetao_mean.npy", "L2225/thetao_std.npy", "L2225/so_mean.npy", "L2225/so_std.npy",
        "L2225/uo_mean.npy", "L2225/uo_std.npy", "L2225/vo_mean.npy", "L2225/vo_std.npy",
        "L3220/thetao_mean.npy", "L3220/thetao_std.npy", "L3220/so_mean.npy", "L3220/so_std.npy",
        "L3220/uo_mean.npy", "L3220/uo_std.npy", "L3220/vo_mean.npy", "L3220/vo_std.npy",
        "L3597/thetao_mean.npy", "L3597/thetao_std.npy", "L3597/so_mean.npy", "L3597/so_std.npy",
        "L3597/uo_mean.npy", "L3597/uo_std.npy", "L3597/vo_mean.npy", "L3597/vo_std.npy",
        "L3992/thetao_mean.npy", "L3992/thetao_std.npy", "L3992/so_mean.npy", "L3992/so_std.npy",
        "L3992/uo_mean.npy", "L3992/uo_std.npy", "L3992/vo_mean.npy", "L3992/vo_std.npy",
        "L4405/thetao_mean.npy", "L4405/thetao_std.npy", "L4405/so_mean.npy", "L4405/so_std.npy",
        "L4405/uo_mean.npy", "L4405/uo_std.npy", "L4405/vo_mean.npy", "L4405/vo_std.npy",
        "L4833/thetao_mean.npy", "L4833/thetao_std.npy", "L4833/so_mean.npy", "L4833/so_std.npy",
        "L4833/uo_mean.npy", "L4833/uo_std.npy", "L4833/vo_mean.npy", "L4833/vo_std.npy",
        "L5274/thetao_mean.npy", "L5274/thetao_std.npy", "L5274/so_mean.npy", "L5274/so_std.npy",
        "L5274/uo_mean.npy", "L5274/uo_std.npy", "L5274/vo_mean.npy", "L5274/vo_std.npy",
        "glonet_p1.pt", "glonet_p2.pt", "glonet_p3.pt",
        "ref1.nc", "ref2.nc", "ref3.nc",
        "xe_weights14/L0.nc", "xe_weights14/L50.nc", "xe_weights14/L100.nc", "xe_weights14/L150.nc",
        "xe_weights14/L222.nc", "xe_weights14/L318.nc", "xe_weights14/L380.nc", "xe_weights14/L450.nc",
        "xe_weights14/L540.nc", "xe_weights14/L640.nc", "xe_weights14/L763.nc", "xe_weights14/L902.nc",
        "xe_weights14/L1245.nc", "xe_weights14/L1684.nc", "xe_weights14/L2225.nc", "xe_weights14/L3220.nc",
        "xe_weights14/L3597.nc", "xe_weights14/L3992.nc", "xe_weights14/L4405.nc", "xe_weights14/L4833.nc",
        "xe_weights14/L5274.nc",
    ]

    for file_name in files_to_download:
        local_file_path = local_dir / file_name
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        object_key = f"{remote_prefix}{file_name}"

        if not local_file_path.exists():
            try:
                http_url = f"{s3_base_url}/{bucket_name}/{object_key}"
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
