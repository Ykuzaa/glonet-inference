from datetime import datetime, date
from xarray import Dataset, concat, merge, open_dataset
from datetime import timedelta
import copernicusmarine     #Télécharger les donnees Copernicus
import numpy
import gc
import os
from s3_upload import save_file_to_s3, get_s3_client
from xesmf import Regridder  #Changer la résolution de la grille
from s3_upload import save_bytes_to_s3
from model import synchronize_model_locally


#Fct qui telecharge les donnees oceanographiques depuis Copernicus Marine pour une profondeur donnee et une date donnee puis les remet sur une grille uniforme (regridding)et retourne le dataset et un flag indiquant si c'est la surface.
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
    print("merging..")
    ds = merge(ds)
    print(ds)
    # Copernicus donne 0.083° resolution, on veut 0.25° (1/4 degre)
    ds_out = Dataset(
        {
            "lat": (
                ["lat"],
                numpy.arange(data.latitude.min(), data.latitude.max(), 1 / 4),
            ),
            "lon": (
                ["lon"],
                numpy.arange(
                    data.longitude.min(), data.longitude.max(), 1 / 4
                ),
            ),
        }
    )

    print("loading regridder")
    # Charger les poids pre-calcules pour interpolation bilineaire
    regridder = Regridder(
        data, ds_out, "bilinear", weights=fn, reuse_weights=True
    )
    print("regridder ready")
    ds_out = regridder(ds)
    print("done regridding")
    ds_out = ds_out.sel(lat=slice(ds_out.lat[8], ds_out.lat[-1]))
    del regridder, ds, data
    gc.collect()
    return ds_out, is_surface

#Donne données initiales pour p1
def glo_in1(model_dir: str, date):
    inp, is_surface = get_data(date, 0.49402499198913574, f"{model_dir}/xe_weights14/L0.nc")
    print(inp)
    return inp, is_surface


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
    inp.append(get_data(date, 4405, f"{model_dir}/xe_weights14/L4405.nc")[0])
    inp.append(get_data(date, 5274, f"{model_dir}/xe_weights14/L5274.nc")[0])
    inp = concat(inp, dim="depth")
    return inp, False


def create_data(ds_out, has_zos):
    thetao = ds_out["thetao"].data
    so = ds_out["so"].data
    uo = ds_out["uo"].data
    vo = ds_out["vo"].data
    
    if has_zos and "zos" in ds_out:
        zos = numpy.expand_dims(ds_out["zos"].data, axis=1)
        tt = numpy.concatenate([zos, thetao, so, uo, vo], axis=1)
    else:
        tt = numpy.concatenate([thetao, so, uo, vo], axis=1)

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


def create_depth_data(date: date, glo_in, model_dir: str):
    dd, is_surface = glo_in(model_dir, date)
    return create_data(dd, is_surface)

#Appelle glo_in, convertit en bytes et sauvegarde sur S3 
def create_data_if_needed(
    bucket_name: str,
    forecast_directory_url: str, # Attend une URL HTTPS comme https://minio.../bucket/folder/DATE
    date: date,
    in_layer: str,
    glo_in,
) -> str:
    # --- DÉFINIR LES CHEMINS ET URLS EN PREMIER ---
    # Extrait le chemin de base relatif au bucket
    # ex: glonet-inference/2025-10-27
    base_s3_key_path = forecast_directory_url.partition(bucket_name + '/')[2]
    base_s3_key_path = base_s3_key_path.lstrip('/')

    # Construit la clé S3 spécifique pour ce fichier initial
    # ex: glonet-inference/2025-10-27/inits/in1.nc
    file_key = f"{base_s3_key_path}/inits/{in_layer}.nc"
    file_key = file_key.lstrip('/')

    # Construit l'URL HTTPS complète (pour la valeur de retour)
    s3_endpoint_url_base = os.environ.get("S3_ENDPOINT", "https://minio.dive.edito.eu").rstrip('/')
    init_netcdf_file_url = f"{s3_endpoint_url_base}/{bucket_name}/{file_key}" # <- DÉFINITION ICI
    s3_full_path = f"s3://{bucket_name}/{file_key}" # Pour les logs
    # --- FIN DÉFINITION ---

    # --- Vérifie si le fichier existe sur S3 ---
    file_exists = False
    try:
        print(f"Vérification existence : {s3_full_path}...")
        s3_client = get_s3_client()
        s3_client.head_object(Bucket=bucket_name, Key=file_key)
        print(f"Fichier initial existe déjà : {s3_full_path}")
        file_exists = True
    except Exception as check_exception:
        # Gère spécifiquement l'erreur "fichier non trouvé"
        is_not_found = False
        if hasattr(check_exception, 'response') and 'Error' in check_exception.response:
            if check_exception.response['Error']['Code'] == '404':
                is_not_found = True

        if is_not_found:
             print(f"Fichier initial n'existe pas : {s3_full_path}. Génération...")
             file_exists = False
        else:
             # Si autre erreur (ex: permission), on arrête
             print(f"Erreur lors de la vérification S3 ({s3_full_path}): {check_exception}")
             raise check_exception
    # --- Fin Vérification ---

    # --- Génère le fichier si nécessaire ---
    if not file_exists:
        # ... (le reste du code pour générer, écrire en temp, uploader) ...
        local_dir = "/tmp/glonet"
        synchronize_model_locally(local_dir)
        dataset = create_depth_data(date, glo_in, model_dir=local_dir)

        temp_file_path = f"/tmp/{in_layer}_temp.nc"
        try:
            print(f"Écriture temporaire {temp_file_path} (HDF5)...")
            dataset.to_netcdf(temp_file_path, engine="h5netcdf")

            print(f"Upload vers {s3_full_path}...")
            save_file_to_s3(
                bucket_name=bucket_name,
                local_file_path=temp_file_path,
                object_key=file_key,
            )
        finally:
            if os.path.exists(temp_file_path):
                print(f"Suppression {temp_file_path}")
                os.remove(temp_file_path)
    # --- Fin Génération ---

    # Retourne l'URL HTTPS comme attendu
    return init_netcdf_file_url


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