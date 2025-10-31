
FROM mambaorg/micromamba:1.5.6-focal-cuda-12.1.1

#COPY --chown=$MAMBA_USER:$MAMBA_USER ./environment.yml /tmp/env.yml

#RUN micromamba install -y -n base -f /tmp/env.yml && \ 
 #   micromamba clean --all --yes

RUN pip install -r requirements.txt

COPY ./glonet_forecast.py /app/glonet_forecast.py
COPY ./model.py /app/model.py
COPY ./utility.py /app/utility.py
COPY ./get_inits.py /app/get_inits.py
COPY ./s3_upload.py /app/s3_upload.py
COPY ./run_glonet_inference.py /app/run_glonet_inference.py

WORKDIR /app

CMD [ "python", "run_glonet_inference.py" ]