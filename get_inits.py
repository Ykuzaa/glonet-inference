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
from xesmf import Regridder
from model import synchronize_model_locally


# Fetch data for a given date and depth, and regrid using provided weights file.
def get_data(date, depth, fn):
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

    ds = []
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
            username=os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"),
            password=os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD"),
        )
        ds.append(data)

    print("Merging datasets...")
    ds = merge(ds)
    print(ds)

    ds_out = Dataset(
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
        data, ds_out, "bilinear", weights=fn, reuse_weights=True
    )
    print("Regridder ready")
    ds_out = regridder(ds)
    print("Done regridding")

    ds_out = ds_out.sel(lat=slice(ds_out.lat[8], ds_out.lat[-1]))
    del regridder, ds, data
    gc.collect()
    return ds_out, is_surface


# Prepare input for surface group (level 1).
def glo_in1(model_dir: str, date):
    inp, is_surface = get_data(date, 0.49402499198913574, f"{model_dir}/xe_weights14/L0.nc")
    print(inp)
    return inp, is_surface


# Prepare input for intermediate depths (level 2).
def glo_in2(model_dir: str, date):
    inp = []
    inp.append(get_data(date, 50, f"{model_dir}/xe_weights14/L50.nc")[0])
    inp.append(get_data(date, 100, f"{model_dir}/xe_weights14/L100.nc")[0])
    inp.append(get_data(date, 150, f"{model_dir}/xe_weights14/L150.nc")[0])
    inp.append(get_data(date, 222, f"{model_dir}/xe_weights14/L222.nc")[0])
    inp.append(get_data(date, 318, f"{model_dir}/xe_weights14/L318.nc")[0])
    inp.append(get_data(date, 380, f"{model_dir}/xe_weights14/L380.nc")[0])
    inp.append(get_data(date, 450, f"{model_dir}/xe_weights14/L450.nc")[0])
    inp.append(get_data(date, 540, f"{model_dir}/xe_weights14/L540.nc")[0])
    inp.append(get_data(date, 640, f"{model_dir}/xe_weights14/L640.nc")[0])
    inp.append(get_data(date, 763, f"{model_dir}/xe_weights14/L763.nc")[0])
    inp = concat(inp, dim="depth")
    return inp, False


# Prepare input for deep ocean depths (level 3).
def glo_in3(model_dir: str, date):
    inp = []
    inp.append(get_data(date, 902, f"{model_dir}/xe_weights14/L902.nc")[0])
    inp.append(get_data(date, 1245, f"{model_dir}/xe_weights14/L1245.nc")[0])
    inp.append(get_data(date, 1684, f"{model_dir}/xe_weights14/L1684.nc")[0])
    inp.append(get_data(date, 2225, f"{model_dir}/xe_weights14/L2225.nc")[0])
    inp.append(get_data(date, 3220, f"{model_dir}/xe_weights14/L3220.nc")[0])
    inp.append(get_data(date, 3597, f"{model_dir}/xe_weights14/L3597.nc")[0])
    inp.append(get_data(date, 3992, f"{model_dir}/xe_weights14/L3992.nc")[0])
    inp.append(get_data(date, 4405, f"{model_dir}/xe_weights14/L4405.nc")[0])
    inp.append(get_data(date, 4833, f"{model_dir}/xe_weights14/L4833.nc")[0])
    inp.append(get_data(date, 5274, f"{model_dir}/xe_weights14/L5274.nc")[0])
    inp = concat(inp, dim="depth")
    return inp, False


# Assemble Dataset with channels (including zos when provided).
def create_data(ds_out, has_zos):
    thetao = ds_out["thetao"].data
    so = ds_out["so"].data
    uo = ds_out["uo"].data
    vo = ds_out["vo"].data

    if has_zos and "zos" in ds_out:
        zos = np.expand_dims(ds_out["zos"].data, axis=1)
        tt = np.concatenate([zos, thetao, so, uo, vo], axis=1)
    else:
        tt = np.concatenate([thetao, so, uo, vo], axis=1)

    lat = ds_out.lat.data
    lon = ds_out.lon.data
    time = ds_out.time.data

    bb = Dataset(
        {
            "data": (("time", "ch", "lat", "lon"), tt),
        },
        coords={
            "time": ("time", time),
            "ch": ("ch", range(0, tt.shape[1])),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )
    return bb


# Create depth data by calling the provided glo_in function.
def create_depth_data(date: date, glo_in, model_dir: str):
    dd, is_surface = glo_in(model_dir, date)
    return create_data(dd, is_surface)


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
