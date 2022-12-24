# microsatellite-instability-classification

conda env
conda env create -f environment.yml

openslide (not mendatory)
conda install -c conda-forge openslide

gdc-client (from gdc conda env)
conda create -n gdc -c bioconda -c conda-forge gdc-client

from data folder and gdc conda env
gdc-client download -m manifest_COAD_READ_MSIStatus_DX.txt


conda install scikit-image

