from PIL import Image
import io
from xarray import Dataset
import numpy as np
from matplotlib import pyplot as plt
from s3_upload import save_bytes_to_s3


def save_image_s3(bucket_name: str, image_bytes, object_url: str):
    save_bytes_to_s3(
        bucket_name=bucket_name,
        object_bytes=image_bytes,
        object_key=object_url.partition(bucket_name + "/")[2],
    )


def _create_and_upload_image(
    bucket_name: str,
    dataset: Dataset,
    variable_name: str,
    colormap: str,
    time_index: int,
    depth_index: int | None,
    s3_url: str,
):
    """Helper function to generate and upload a single thumbnail."""
    print(f"Generating thumbnail for '{variable_name}'...")

    # Select data slice based on whether it's a 2D (surface) or 3D variable
    if depth_index is None:
        data_slice = dataset[variable_name][time_index]
    else:
        data_slice = dataset[variable_name][time_index, depth_index]

    # Reproject and prepare data for imaging
    data_slice = data_slice.rio.write_crs("EPSG:4326")
    data_ortho = data_slice.rio.reproject("EPSG:4326")

    # Normalize data to range 0-255 for color mapping
    data_min, data_max = (data_ortho.min().item(), data_ortho.max().item())
    if data_min == data_max:  # Avoid division by zero if data is flat
        data_normalized = np.zeros_like(data_ortho, dtype=np.uint8)
    else:
        data_normalized = (
            (data_ortho - data_min) / (data_max - data_min) * 255
        ).astype(np.uint8)

    # Apply colormap and set transparency for NaN values
    cmap = plt.get_cmap(colormap)
    colored_data = cmap(data_normalized)
    alpha = np.where(np.isnan(data_ortho), 0, 255).astype(np.uint8)
    colored_data[..., 3] = alpha / 255.0  # Set alpha channel

    # Create image and save to an in-memory buffer
    rgba_data = (colored_data * 255).astype(np.uint8)
    img = Image.fromarray(rgba_data, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    # Upload to S3
    save_image_s3(bucket_name, image_bytes, s3_url)


def generate_thumbnail(
    bucket_name: str,
    forecast_netcdf_file_url: str,
    thumbnail_file_urls: dict[str, str],
    forecast_dataset: Dataset,
) -> dict[str, str]:
    try:
        # Configurations for each variable's thumbnail
        configs = {
            "zos":    {"cmap": "seismic",  "time": 9, "depth": None},
            "thetao": {"cmap": "viridis",  "time": 9, "depth": 0},
            "so":     {"cmap": "jet",      "time": 9, "depth": 0},
            "uo":     {"cmap": "coolwarm", "time": 2, "depth": 0},
            "vo":     {"cmap": "coolwarm", "time": 2, "depth": 0},
        }

        for var, config in configs.items():
            _create_and_upload_image(
                bucket_name=bucket_name,
                dataset=forecast_dataset,
                variable_name=var,
                colormap=config["cmap"],
                time_index=config["time"],
                depth_index=config["depth"],
                s3_url=thumbnail_file_urls[var],
            )
    except Exception as exception:
        print(
            f"Failed to generate thumbnails for resource {forecast_netcdf_file_url}: {exception}"
        )
        raise
    return thumbnail_file_urls
