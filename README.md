# microsatellite-instability-classification
## Introduction
Microsatellite instability (MSI) is a type of genetic alteration that can occur in certain types of cancer and is important for diagnosis and treatment. In this research project, we propose to classify MSI status from whole slide images (WSI) in the Cancer Genome Atlas (TCGA) dataset using a deep learning model that incorporates global features.


One of the challenges in using deep learning for WSI analysis is that WSIs can be gigapixel in size, which makes it difficult to use standard approaches. To overcome this challenge, we propose to develop a model that can effectively process and analyze the large amount of data contained in WSIs by extracting meaningful global features for use in classification. By utilizing the power of deep learning and focusing on global rather than local features, we hope to achieve high levels of accuracy and robustness in our MSI classification model.

## Environment Setup
To set up the project environment, run the following commands:

```conda env create -f environment.yml```

```conda activate MSI```

## Data Download
In this project, we are specifically focusing on the COAD and READ cohorts. These cohort are publicily available as part of the TCGA dataset. 

To download the data, you will need to create a new conda environment just for the gdc-client. To do this, run the following commands:

```conda create -n gdc -c bioconda -c conda-forge gdc-client```

```conda activate gdc```

Next, navigate to the data/slides or data/labels directory and use the gdc-client to download the data using the TCGA manifest file:

```gdc-client download -m <manifest file>```

## Preprocessing
Before we can use the WSIs for our deep learning model, we need to perform some preprocessing steps. These include:

* Scaling to the target magnification power (MPP) of 0.5.
* Tiling the WSIs into 512x512 or 256x256 tiles.
* Filtering out background tiles using Otsu's method, as well as tiles that contain dark or black spots or pen marks that are not needed for deep learning.
* Applying Macenko color normalization to standardize the color of the tiles
* Saving the processed tiles with their spatial information into a dedicated folder for the next phases of the analysis.

## References
* [wsi-tile-cleanup] https://github.com/lucasrla/wsi-tile-cleanup

* [wsi-preprocessing] https://github.com/lucasrla/wsi-preprocessing

