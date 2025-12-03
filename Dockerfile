
FROM mambaorg/micromamba:1.5.6-focal-cuda-12.1.1

COPY --chown=$MAMBA_USER:$MAMBA_USER ./environment.yml /tmp/env.yml

RUN micromamba install -y -n base -f /tmp/env.yml && \ 
    micromamba clean --all --yes

WORKDIR /app 

COPY ./glonet_forecast.py .
COPY ./model.py .
COPY ./utility.py .
COPY ./get_inits.py .
COPY ./s3_upload.py .
COPY ./generate_thumbnails.py .
COPY ./run_glonet_inference.py .


CMD [ "python", "run_glonet_inference.py" ]