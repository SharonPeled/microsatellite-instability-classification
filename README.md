# microsatellite-instability-classification

conda env (linux)
conda env create -f environment.yml

gdc-client (from gdc conda env)
conda create -n gdc -c bioconda -c conda-forge gdc-client
conda activate gdc

from data folder and gdc conda env
gdc-client download -m manifest_COAD_READ_MSIStatus_DX.txt

