# microsatellite-instability-classification

## Environment setup
conda env create -f environment.yml

## Data download
Since there are conflicts with the project's environment, you need to create a new conda env
just for downloading the data from TCGA using gdc-client.
Creating new env and installing gdc-client:
conda create -n gdc -c bioconda -c conda-forge gdc-client
Activating the env just created:
conda activate gdc
Downloading the data using TCGA manifest:
Navigate to data/slides or data/labels and run command:
gdc-client download -m <manifest file>

