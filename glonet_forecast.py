"""
GLONET forecast assembly and inference utilities.

This module provides functions to convert model predictions into NetCDF
datasets, run model inference on GPU, attach metadata, and assemble the
final forecast dataset for a given forecast day.
"""
import gc
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import os
import s3fs 
import torch
import config


# Function that creates NetCDF datasets from model predictions
def make_nc(vars, denormalizer, forecast_date, lead_step, model_dir: str):
    vars = denormalizer(vars) # `vars` here is a tensor, not a dataset, so it remains.
    reference_dataset = xr.open_dataset(model_dir + "/" + "ref1.nc")  # reference file ref1
    reference_dataset = xr.concat([reference_dataset] * vars.shape[1], dim="time")
    reference_dataset["zos"] = reference_dataset["zos"] * vars.numpy()[0, :, 0:1].squeeze()
    reference_dataset["thetao"] = reference_dataset["thetao"] * vars.numpy()[0, :, 1:2]  # multiply reference data by model correction coefficients
    reference_dataset["so"] = reference_dataset["so"] * vars.numpy()[0, :, 2:3]
    reference_dataset["uo"] = reference_dataset["uo"] * vars.numpy()[0, :, 3:4]
    reference_dataset["vo"] = reference_dataset["vo"] * vars.numpy()[0, :, 4:5]

    # Assign dates
    time = np.arange(
        str(forecast_date + timedelta(days=2 * lead_step)),
        str(forecast_date + timedelta(days=2 * lead_step + 2)),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    reference_dataset = reference_dataset.assign_coords(time=time)
    return xr.decode_cf(reference_dataset)


# For intermediate depths
def make_nc2(vars, denormalizer, forecast_date, lead_step, model_dir: str):
    vars = denormalizer(vars) # `vars` here is a tensor, not a dataset, so it remains.
    reference_dataset = xr.open_dataset(model_dir + "/" + "ref2.nc")
    reference_dataset = xr.concat([reference_dataset] * vars.shape[1], dim="time")
    reference_dataset["thetao"] = reference_dataset["thetao"] * vars.numpy()[0, :, 0:10]
    reference_dataset["so"] = reference_dataset["so"] * vars.numpy()[0, :, 10:20]
    reference_dataset["uo"] = reference_dataset["uo"] * vars.numpy()[0, :, 20:30]
    reference_dataset["vo"] = reference_dataset["vo"] * vars.numpy()[0, :, 30:40]

    time = np.arange(
        str(forecast_date + timedelta(days=2 * lead_step)),
        str(forecast_date + timedelta(days=2 * lead_step + 2)),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    reference_dataset = reference_dataset.assign_coords(time=time)
    return xr.decode_cf(reference_dataset)


# For deep depths
def make_nc3(vars, denormalizer, forecast_date, lead_step, model_dir: str):
    vars = denormalizer(vars) # `vars` here is a tensor, not a dataset, so it remains.
    reference_dataset = xr.open_dataset(model_dir + "/" + "ref3.nc")
    reference_dataset = xr.concat([reference_dataset] * vars.shape[1], dim="time")
    reference_dataset["thetao"] = reference_dataset["thetao"] * vars.numpy()[0, :, 0:10]
    reference_dataset["so"] = reference_dataset["so"] * vars.numpy()[0, :, 10:20]
    reference_dataset["uo"] = reference_dataset["uo"] * vars.numpy()[0, :, 20:30]
    reference_dataset["vo"] = reference_dataset["vo"] * vars.numpy()[0, :, 30:40]

    time = np.arange(
        str(forecast_date + timedelta(days=2 * lead_step)),
        str(forecast_date + timedelta(days=2 * lead_step + 2)),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    reference_dataset = reference_dataset.assign_coords(time=time)
    return xr.decode_cf(reference_dataset)


def _run_inference_loop(
    input_dataset,
    forecast_date,
    model_dir: str,
    model_name: str,
    normalizer,
    denormalizer,
    make_nc_function,
):
    """Generic function to run the inference loop for a given depth group."""
    # Mask to ignore NaNs (missing values)
    nan_mask = np.isnan(input_dataset.variables["data"][1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32)

    # Convert data to PyTorch tensor
    data_array = np.nan_to_num(input_dataset.data.data, copy=False)
    vin = torch.tensor(data_array, dtype=torch.float32)
    mask = mask.cpu().detach().unsqueeze(0)
    vin = normalizer(vin)

    vin = vin.cpu().detach().unsqueeze(0)
    vin = vin.contiguous()
    forecast_datasets = []
    del data_array, nan_mask, input_dataset
    gc.collect()  # Force garbage collection

    # Loop 5 times: 5*2 days = 10 days total
    for i in range(1, 6):
        print(i)
        inference_model = torch.jit.load(f"{model_dir}/{model_name}")
        inference_model = inference_model.to("cuda:0")  # send to GPU
        with torch.no_grad():
            inference_model.eval()  # evaluation mode (no training)
            with torch.amp.autocast("cuda"):
                vin = vin * mask.cpu()
                outvar = inference_model(vin.to("cuda:0"))  # inference
                outvar = outvar.detach().cpu()  # bring back to CPU to free GPU memory
        del vin
        gc.collect()

        dataset_item = make_nc_function(outvar, denormalizer, forecast_date, i, model_dir)
        forecast_datasets.append(dataset_item)
        vin = outvar
        vin = vin.cpu().detach()
        del outvar, inference_model
        gc.collect()

    del vin, mask
    gc.collect()
    return forecast_datasets


# Run inference on GPU (5 iterations)
def aforecast(input_dataset, forecast_date, model_dir: str):
    from utility import get_denormalizer1, get_normalizer1
    return _run_inference_loop(
        input_dataset, forecast_date, model_dir, "glonet_p1.pt",
        get_normalizer1(model_dir), get_denormalizer1(model_dir), make_nc
    )


# Run inference for intermediate depths
def aforecast2(input_dataset, forecast_date, model_dir: str):
    from utility import get_denormalizer2, get_normalizer2
    return _run_inference_loop(
        input_dataset, forecast_date, model_dir, "glonet_p2.pt",
        get_normalizer2(model_dir), get_denormalizer2(model_dir), make_nc2
    )


# Run inference for deep depths
def aforecast3(input_dataset, forecast_date, model_dir: str):
    from utility import get_denormalizer3, get_normalizer3
    return _run_inference_loop(
        input_dataset, forecast_date, model_dir, "glonet_p3.pt",
        get_normalizer3(model_dir), get_denormalizer3(model_dir), make_nc3
    )


# Add metadata to the NetCDF dataset
def add_metadata(dataset, date):
    dataset = dataset.rename({"lat": "latitude", "lon": "longitude"})
    # Add global attributes
    dataset.attrs["Conventions"] = "CF-1.8"
    dataset.attrs["area"] = "Global"
    dataset.attrs["Conventions"] = "CF-1.8"
    dataset.attrs["contact"] = "glonet@mercator-ocean.eu"
    dataset.attrs["institution"] = "Mercator Ocean International"
    dataset.attrs["source"] = "MOI GLONET"
    dataset.attrs["title"] = (
        "daily mean fields from GLONET 1/4 degree resolution Forecast updated Daily"
    )
    dataset.attrs["references"] = "www.edito.eu"

    del dataset.attrs["regrid_method"]

    # zos variable
    dataset["zos"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Sea surface height",
        "standard_name": "sea_surface_height_above_geoid",
        "unit_long": "Meters",
        "units": "m",
        "valid_max": 5.0,
        "valid_min": -5.0,
    }

    # latitude variable
    dataset["latitude"].attrs = {
        "axis": "Y",
        "long_name": "Latitude",
        "standard_name": "latitude",
        "step": dataset.latitude.values[1] - dataset.latitude.values[0],
        "unit_long": "Degrees North",
        "units": "degrees_north",
        "valid_max": dataset.latitude.values.max(),
        "valid_min": dataset.latitude.values.min(),
    }

    # longitude variable
    dataset["longitude"].attrs = {
        "axis": "X",
        "long_name": "Longitude",
        "standard_name": "longitude",
        "step": dataset.longitude.values[1] - dataset.longitude.values[0],
        "unit_long": "Degrees East",
        "units": "degrees_east",
        "valid_max": dataset.longitude.values.max(),
        "valid_min": dataset.longitude.values.min(),
    }

    # time variable
    dataset["time"].attrs = {
        "valid_min": str(date + timedelta(days=1)),
        "valid_max": str(date + timedelta(days=10)),
    }

    # depth variable
    dataset["depth"].attrs = {
        "axis": "Z",
        "long_name": "Elevation",
        "positive": "down",
        "standard_name": "elevation",
        "unit_long": "Meters",
        "units": "m",
        "valid_min": 0.494025,
        "valid_max": 5727.917,
    }

    # uo variable
    dataset["uo"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Eastward velocity",
        "standard_name": "eastward_sea_water_velocity",
        "unit_long": "Meters per second",
        "units": "m s-1",
        "valid_max": 5.0,
        "valid_min": -5.0,
    }

    # vo variable
    dataset["vo"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Northward velocity",
        "standard_name": "northward_sea_water_velocity",
        "unit_long": "Meters per second",
        "units": "m s-1",
        "valid_max": 5.0,
        "valid_min": -5.0,
    }

    # so variable
    dataset["so"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Salinity",
        "standard_name": "sea_water_salinity",
        "unit_long": "Practical Salinity Unit",
        "units": "1e-3",
        "valid_max": 50.0,
        "valid_min": 0.0,
    }

    # thetao variable
    dataset["thetao"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Temperature",
        "standard_name": "sea_water_potential_temperature",
        "unit_long": "Degrees Celsius",
        "units": "degrees_C",
        "valid_max": 40.0,
        "valid_min": -10.0,
    }
    return dataset


# Main function: create forecast dataset
def create_forecast(
    forecast_netcdf_file_url: str,
    model_dir: str,
    initial_file_1_url: str,
    initial_file_2_url: str,
    initial_file_3_url: str,
) -> xr.Dataset:
    from s3_upload import get_s3_endpoint_url_with_protocol
    forecast_directory_url = forecast_netcdf_file_url.rpartition("/")[0]
    day_string = forecast_directory_url.rpartition("/")[2]
    date = datetime.strptime(day_string, "%Y-%m-%d").date()

    start_datetime = str(date + timedelta(days=1))
    end_datetime = str(date + timedelta(days=10))
    print(
        f"Creating {day_string} forecast from {start_datetime} to {end_datetime}..."
    )

    print("Configuring S3 client for reading...")
    s3_url_with_protocol = get_s3_endpoint_url_with_protocol()

    s3 = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": s3_url_with_protocol},
        key=config.AWS_ACCESS_KEY_ID,
        secret=config.AWS_SECRET_ACCESS_KEY,
        token=config.AWS_SESSION_TOKEN,
    )
    s3_endpoint_url_base = os.environ.get("AWS_S3_ENDPOINT", "https://minio.dive.edito.eu").rstrip("/") + "/"
    if s3_endpoint_url_base and not s3_endpoint_url_base.startswith("https://"):
            s3_endpoint_url_base = "https://" + s3_endpoint_url_base
            
    def get_s3_path(url):
        if url.startswith(s3_endpoint_url_base):
            return url[len(s3_endpoint_url_base):].lstrip("/")
        else:
            parts = url.replace("https://", "", 1).split("/", 1)
            return parts[1] if len(parts) > 1 else ""

    s3_path1 = get_s3_path(initial_file_1_url)
    s3_path2 = get_s3_path(initial_file_2_url)
    s3_path3 = get_s3_path(initial_file_3_url)

    print(f"Authenticated opening of {s3_path1}...")
    with s3.open(s3_path1, "rb") as f1:
        rdata1 = xr.open_dataset(f1, engine="h5netcdf").load()

    print(f"Authenticated opening of {s3_path2}...")
    with s3.open(s3_path2, "rb") as f2:
        rdata2 = xr.open_dataset(f2, engine="h5netcdf").load()

    print(f"Authenticated opening of {s3_path3}...")
    with s3.open(s3_path3, "rb") as f3:
        rdata3 = xr.open_dataset(f3, engine="h5netcdf").load()

    forecast_dataset_1 = aforecast(rdata1, date - timedelta(days=1), model_dir=model_dir)
    del rdata1
    gc.collect()
    forecast_dataset_2 = aforecast2(rdata2, date - timedelta(days=1), model_dir=model_dir)
    del rdata2
    gc.collect()
    forecast_dataset_3 = aforecast3(rdata3, date - timedelta(days=1), model_dir=model_dir)
    del rdata3
    gc.collect()

    combined1 = xr.concat(forecast_dataset_1, dim="time")
    combined2 = xr.concat(forecast_dataset_2, dim="time")
    combined3 = xr.concat(forecast_dataset_3, dim="time")
    del forecast_dataset_1, forecast_dataset_2, forecast_dataset_3
    gc.collect()

    print("Extracting 2D variable 'zos'...")
    zos_data = combined1["zos"]

    combined1 = combined1.drop_vars("zos")

    print("Merging 3D variables along 'depth'...")
    combined4 = xr.concat([combined1, combined2, combined3], dim="depth", data_vars="minimal")

    print("Re-attaching 'zos' variable...")
    combined4["zos"] = zos_data

    combined4 = add_metadata(combined4, date)
    os.makedirs(end_datetime, exist_ok=True)

    del combined1, combined2, combined3
    gc.collect()

    return combined4
