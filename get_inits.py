"""
Initial data preparation utilities for GLONET inference.

This module fetches Copernicus Marine data, regrids it using precomputed
weights, assembles input arrays for different depth groups, and uploads
generated initial NetCDF files to S3 if they do not already exist.
"""
from datetime import datetime, timedelta, date
from xarray import Dataset, concat, merge, open_dataset
import copernicusmarine
import numpy as np
import gc
import os
from s3_upload import save_file_to_s3, get_s3_client, get_s3_endpoint_url_with_protocol
import config
from xesmf import Regridder
from model import synchronize_model_locally


# Fetch data for a given date and depth, and regrid using provided weights file.
def get_data(date, depth, weights_filepath):
    start_datetime = str(date - timedelta(days=1))
    end_datetime = str(date)
    is_surface = abs(depth - 0.49402499198913574) < 0.01

    if is_surface:
        print("yes - surface level with zos")
        mlist = [
            "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        ]
        mvar = [["zos"], ["uo", "vo"], ["so"], ["thetao"]]
    else:
        print("no - subsurface level without zos")
        mlist = [
            "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        ]
        mvar = [["uo", "vo"], ["so"], ["thetao"]]

    dataset = []
    for i in range(0, len(mlist)):
        print(mvar[i])
        data = copernicusmarine.open_dataset(
            dataset_id=mlist[i],
            variables=mvar[i],
            minimum_longitude=-180,
            maximum_longitude=180,
            minimum_latitude=-80,
            maximum_latitude=90,
            minimum_depth=depth,
            maximum_depth=depth,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            username=config.COPERNICUSMARINE_SERVICE_USERNAME,
            password=config.COPERNICUSMARINE_SERVICE_PASSWORD,
        )
        dataset.append(data)

    print("Merging datasets...")
    dataset = merge(dataset)
    print(dataset)
    dataset_out = Dataset(
        {
            "lat": (
                ["lat"],
                np.arange(data.latitude.min(), data.latitude.max(), 1 / 4),
            ),
            "lon": (
                ["lon"],
                np.arange(data.longitude.min(), data.longitude.max(), 1 / 4),
            ),
        }
    )

    print("Loading regridder")
    regridder = Regridder(
        dataset, dataset_out, "bilinear", weights=weights_filepath, reuse_weights=True
    )
    print("Regridder ready")
    dataset_out = regridder(dataset)
    print("Done regridding")

    dataset_out = dataset_out.sel(lat=slice(dataset_out.lat[8], dataset_out.lat[-1]))
    del regridder, dataset, data
    gc.collect()
    return dataset_out, is_surface


# Prepare input for surface group (level 1).
def glo_in1(model_dir: str, date):
    input_dataset, is_surface = get_data(date, 0.49402499198913574, f"{model_dir}/xe_weights14/L0.nc")
    print(input_dataset)
    return input_dataset, is_surface


def _get_multidepth_data(model_dir: str, date, depths: list[int]):
    """Helper to fetch and concatenate data for multiple depths."""
    dataset_list = []
    for depth in depths:
        weights_file = f"{model_dir}/xe_weights14/L{depth}.nc"
        dataset, _ = get_data(date, depth, weights_file)
        dataset_list.append(dataset)
    return concat(dataset_list, dim="depth"), False


# Prepare input for intermediate depths (level 2).
def glo_in2(model_dir: str, date):
    depths = [50, 100, 150, 222, 318, 380, 450, 540, 640, 763]
    return _get_multidepth_data(model_dir, date, depths)


# Prepare input for deep ocean depths (level 3).
def glo_in3(model_dir: str, date):
    depths = [902, 1245, 1684, 2225, 3220, 3597, 3992, 4405, 4833, 5274]
    return _get_multidepth_data(model_dir, date, depths)


# Assemble Dataset with channels (including zos when provided).
def create_data(dataset_out, has_zos):
    thetao = dataset_out["thetao"].data
    so = dataset_out["so"].data
    uo = dataset_out["uo"].data
    vo = dataset_out["vo"].data

    if has_zos and "zos" in dataset_out:
        zos = np.expand_dims(dataset_out["zos"].data, axis=1)
        data_tensor = np.concatenate([zos, thetao, so, uo, vo], axis=1)
    else:
        data_tensor = np.concatenate([thetao, so, uo, vo], axis=1)

    lat = dataset_out.lat.data
    lon = dataset_out.lon.data
    time = dataset_out.time.data

    channel_dataset = Dataset(
        {
            "data": (("time", "ch", "lat", "lon"), data_tensor),
        },
        coords={
            "time": ("time", time),
            "ch": ("ch", range(0, data_tensor.shape[1])),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )
    return channel_dataset


# Create depth data by calling the provided glo_in function.
def create_depth_data(date: date, glo_in, model_dir: str):
    depth_dataset, is_surface = glo_in(model_dir, date)
    return create_data(depth_dataset, is_surface)


# Create the init file on S3 if it does not already exist.
def create_data_if_needed(
    bucket_name: str,
    forecast_directory_url: str,
    date: date,
    in_layer: str,
    glo_in,
) -> str:
    from s3_upload import get_s3_endpoint_url_with_protocol

    base_s3_key_path = forecast_directory_url.partition(bucket_name + "/")[2]
    base_s3_key_path = base_s3_key_path.lstrip("/")

    file_key = f"{base_s3_key_path}/inits/{in_layer}.nc"
    file_key = file_key.lstrip("/")

    s3_endpoint_url_base = get_s3_endpoint_url_with_protocol().rstrip("/")

    init_netcdf_file_url = f"{s3_endpoint_url_base}/{bucket_name}/{file_key}"
    s3_full_path = f"s3://{bucket_name}/{file_key}"

    file_exists = False
    try:
        print(f"Checking existence: {s3_full_path}...")
        s3_client = get_s3_client()
        s3_client.head_object(Bucket=bucket_name, Key=file_key)
        print(f"Init file already exists: {s3_full_path}")
        file_exists = True
    except Exception as check_exception:
        is_not_found = False
        if hasattr(check_exception, "response") and "Error" in check_exception.response:
            if check_exception.response["Error"]["Code"] == "404":
                is_not_found = True
            if check_exception.response["Error"]["Code"] == "403":
                print(
                    "WARNING: 403 Forbidden received. Treating as not found (may require IAM permission fix)."
                )
                is_not_found = True

        if is_not_found:
            print(f"Init file does not exist: {s3_full_path}. Generating...")
            file_exists = False
        else:
            print(f"Error while checking S3 ({s3_full_path}): {check_exception}")
            raise check_exception

    if not file_exists:
        local_dir = "/tmp/glonet"
        synchronize_model_locally(local_dir)
        dataset = create_depth_data(date, glo_in, model_dir=local_dir)

        temp_file_path = f"/tmp/{in_layer}_temp.nc"
        try:
            print(f"Writing temporary file {temp_file_path} (HDF5)...")
            dataset.to_netcdf(temp_file_path, engine="h5netcdf")

            print(f"Uploading to {s3_full_path}...")
            save_file_to_s3(
                bucket_name=bucket_name,
                local_file_path=temp_file_path,
                object_key=file_key,
            )
        finally:
            if os.path.exists(temp_file_path):
                print(f"Removing {temp_file_path}")
                os.remove(temp_file_path)

    return init_netcdf_file_url


# Generate in1, in2 and in3 URLs for a forecast file URL.
def generate_initial_data(bucket_name, forecast_netcdf_file_url: str):
    forecast_directory_url = forecast_netcdf_file_url.rpartition("/")[0]
    day_string = forecast_directory_url.rpartition("/")[2]

    date = datetime.strptime(day_string, "%Y-%m-%d").date()
    in1_url = create_data_if_needed(
        bucket_name, forecast_directory_url, date, "in1", glo_in1
    )
    in2_url = create_data_if_needed(
        bucket_name, forecast_directory_url, date, "in2", glo_in2
    )
    in3_url = create_data_if_needed(
        bucket_name, forecast_directory_url, date, "in3", glo_in3
    )
    return in1_url, in2_url, in3_url
