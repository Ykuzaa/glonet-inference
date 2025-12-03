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
from model import synchronize_model_locally
import torch


# Function that creates NetCDF datasets from model predictions
def make_nc(vars, denormalizer, ti, lead, model_dir: str):
    vars = denormalizer(vars)
    d = xr.open_dataset(model_dir + "/" + "ref1.nc")  # reference file ref1
    d = xr.concat([d] * vars.shape[1], dim="time")
    d["zos"] = d["zos"] * vars.numpy()[0, :, 0:1].squeeze()
    d["thetao"] = d["thetao"] * vars.numpy()[0, :, 1:2]  # multiply reference data by model correction coefficients
    d["so"] = d["so"] * vars.numpy()[0, :, 2:3]
    d["uo"] = d["uo"] * vars.numpy()[0, :, 3:4]
    d["vo"] = d["vo"] * vars.numpy()[0, :, 4:5]

    # Assign dates
    time = np.arange(
        str(ti + timedelta(days=2 * lead)),
        str(ti + timedelta(days=2 * lead + 2)),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    d = d.assign_coords(time=time)
    return xr.decode_cf(d)


# For intermediate depths
def make_nc2(vars, denormalizer, ti, lead, model_dir: str):
    vars = denormalizer(vars)
    d = xr.open_dataset(model_dir + "/" + "ref2.nc")
    d = xr.concat([d] * vars.shape[1], dim="time")
    d["thetao"] = d["thetao"] * vars.numpy()[0, :, 0:10]
    d["so"] = d["so"] * vars.numpy()[0, :, 10:20]
    d["uo"] = d["uo"] * vars.numpy()[0, :, 20:30]
    d["vo"] = d["vo"] * vars.numpy()[0, :, 30:40]

    time = np.arange(
        str(ti + timedelta(days=2 * lead)),
        str(ti + timedelta(days=2 * lead + 2)),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    d = d.assign_coords(time=time)
    return xr.decode_cf(d)


# For deep depths
def make_nc3(vars, denormalizer, ti, lead, model_dir: str):
    vars = denormalizer(vars)
    d = xr.open_dataset(model_dir + "/" + "ref3.nc")
    d = xr.concat([d] * vars.shape[1], dim="time")
    d["thetao"] = d["thetao"] * vars.numpy()[0, :, 0:10]
    d["so"] = d["so"] * vars.numpy()[0, :, 10:20]
    d["uo"] = d["uo"] * vars.numpy()[0, :, 20:30]
    d["vo"] = d["vo"] * vars.numpy()[0, :, 30:40]

    time = np.arange(
        str(ti + timedelta(days=2 * lead)),
        str(ti + timedelta(days=2 * lead + 2)),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    d = d.assign_coords(time=time)
    return xr.decode_cf(d)


# Run inference on GPU (5 iterations)
def aforecast(d, date, model_dir: str):
    from utility import get_denormalizer1, get_normalizer1

    denormalizer = get_denormalizer1(model_dir)
    normalizer = get_normalizer1(model_dir)

    # Mask to ignore NaNs (missing values)
    nan_mask = np.isnan(d.variables["data"][1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32)

    # Convert data to PyTorch tensor
    data = np.nan_to_num(d.data.data, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    mask = mask.cpu().detach().unsqueeze(0)
    vin = normalizer(vin)

    vin = vin.cpu().detach().unsqueeze(0)
    vin = vin.contiguous()
    datasets = []
    del data, nan_mask
    gc.collect()  # Force garbage collection

    # Loop 5 times: 5*2 days = 10 days total
    for i in range(1, 6):
        print(i)
        # Load model p1 from disk
        model_inf = torch.jit.load(model_dir + "/" + "glonet_p1.pt")
        model_inf = model_inf.to("cuda:0")  # send to GPU
        with torch.no_grad():
            model_inf.eval()  # evaluation mode (no training)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                vin = vin * mask.cpu()
                outvar = model_inf(vin.to("cuda:0"))  # inference
                outvar = outvar.detach().cpu()  # bring back to CPU to free GPU memory
        del vin
        gc.collect()

        d = make_nc(outvar, denormalizer, date, i, model_dir)  # create NetCDF dataset from tensor (in RAM)
        datasets.append(d)
        # free memory
        vin = outvar
        vin = vin.cpu().detach()
        del outvar, model_inf
        gc.collect()

    del vin, mask
    gc.collect()
    return datasets


# Run inference for intermediate depths
def aforecast2(d, date, model_dir: str):
    from utility import get_denormalizer2, get_normalizer2

    denormalizer = get_denormalizer2(model_dir)
    normalizer = get_normalizer2(model_dir)

    nan_mask = np.isnan(d.variables["data"][1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32)
    data = np.nan_to_num(d.data.data, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    mask = mask.cpu().detach().unsqueeze(0)
    vin = normalizer(vin)

    vin = vin.cpu().detach().unsqueeze(0)
    vin = vin.contiguous()
    datasets = []
    del data, nan_mask
    gc.collect()  # Force garbage collection

    for i in range(1, 6):
        print(i)
        model_inf = torch.jit.load(model_dir + "/" + "glonet_p2.pt")
        model_inf = model_inf.to("cuda:0")
        with torch.no_grad():
            model_inf.eval()

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                vin = vin * mask.cpu()
                outvar = model_inf(vin.to("cuda:0"))
                outvar = outvar.detach().cpu()

        del vin
        gc.collect()

        d = make_nc2(outvar, denormalizer, date, i, model_dir)
        datasets.append(d)
        vin = outvar
        vin = vin.cpu().detach()
        del outvar, model_inf
        gc.collect()

    del vin, mask
    gc.collect()
    return datasets


# Run inference for deep depths
def aforecast3(d, date, model_dir: str):
    from utility import get_denormalizer3, get_normalizer3

    denormalizer = get_denormalizer3(model_dir)
    normalizer = get_normalizer3(model_dir)

    nan_mask = np.isnan(d.variables["data"][1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32)
    data = np.nan_to_num(d.data.data, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    mask = mask.cpu().detach().unsqueeze(0)
    vin = normalizer(vin)

    vin = vin.cpu().detach().unsqueeze(0)
    vin = vin.contiguous()
    datasets = []
    del data, nan_mask
    gc.collect()  # Force garbage collection

    for i in range(1, 6):
        print(i)
        model_inf = torch.jit.load(model_dir + "/" + "glonet_p3.pt")
        model_inf = model_inf.to("cuda:0")
        with torch.no_grad():
            model_inf.eval()

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                vin = vin * mask.cpu()
                outvar = model_inf(vin.to("cuda:0"))
                outvar = outvar.detach().cpu()

        del vin
        gc.collect()

        d = make_nc3(outvar, denormalizer, date, i, model_dir)
        datasets.append(d)
        vin = outvar
        vin = vin.cpu().detach()
        del outvar, model_inf
        gc.collect()

    del vin, mask
    gc.collect()
    return datasets


# Add metadata to the NetCDF dataset
def add_metadata(ds, date):
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    # Add global attributes
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["area"] = "Global"
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["contact"] = "glonet@mercator-ocean.eu"
    ds.attrs["institution"] = "Mercator Ocean International"
    ds.attrs["source"] = "MOI GLONET"
    ds.attrs["title"] = (
        "daily mean fields from GLONET 1/4 degree resolution Forecast updated Daily"
    )
    ds.attrs["references"] = "www.edito.eu"

    del ds.attrs["regrid_method"]

    # zos variable
    ds["zos"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Sea surface height",
        "standard_name": "sea_surface_height_above_geoid",
        "unit_long": "Meters",
        "units": "m",
        "valid_max": 5.0,
        "valid_min": -5.0,
    }

    # latitude variable
    ds["latitude"].attrs = {
        "axis": "Y",
        "long_name": "Latitude",
        "standard_name": "latitude",
        "step": ds.latitude.values[1] - ds.latitude.values[0],
        "unit_long": "Degrees North",
        "units": "degrees_north",
        "valid_max": ds.latitude.values.max(),
        "valid_min": ds.latitude.values.min(),
    }

    # longitude variable
    ds["longitude"].attrs = {
        "axis": "X",
        "long_name": "Longitude",
        "standard_name": "longitude",
        "step": ds.longitude.values[1] - ds.longitude.values[0],
        "unit_long": "Degrees East",
        "units": "degrees_east",
        "valid_max": ds.longitude.values.max(),
        "valid_min": ds.longitude.values.min(),
    }

    # time variable
    ds["time"].attrs = {
        "valid_min": str(date + timedelta(days=1)),
        "valid_max": str(date + timedelta(days=10)),
    }

    # depth variable
    ds["depth"].attrs = {
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
    ds["uo"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Eastward velocity",
        "standard_name": "eastward_sea_water_velocity",
        "unit_long": "Meters per second",
        "units": "m s-1",
        "valid_max": 5.0,
        "valid_min": -5.0,
    }

    # vo variable
    ds["vo"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Northward velocity",
        "standard_name": "northward_sea_water_velocity",
        "unit_long": "Meters per second",
        "units": "m s-1",
        "valid_max": 5.0,
        "valid_min": -5.0,
    }

    # so variable
    ds["so"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Salinity",
        "standard_name": "sea_water_salinity",
        "unit_long": "Practical Salinity Unit",
        "units": "1e-3",
        "valid_max": 50.0,
        "valid_min": 0.0,
    }

    # thetao variable
    ds["thetao"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Temperature",
        "standard_name": "sea_water_potential_temperature",
        "unit_long": "Degrees Celsius",
        "units": "degrees_C",
        "valid_max": 40.0,
        "valid_min": -10.0,
    }
    return ds


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
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"],
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

    ds1 = aforecast(rdata1, date - timedelta(days=1), model_dir=model_dir)
    del rdata1
    gc.collect()
    ds2 = aforecast2(rdata2, date - timedelta(days=1), model_dir=model_dir)
    del rdata2
    gc.collect()
    ds3 = aforecast3(rdata3, date - timedelta(days=1), model_dir=model_dir)
    del rdata3
    gc.collect()

    combined1 = xr.concat(ds1, dim="time")
    combined2 = xr.concat(ds2, dim="time")
    combined3 = xr.concat(ds3, dim="time")
    del ds1, ds2, ds3
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
