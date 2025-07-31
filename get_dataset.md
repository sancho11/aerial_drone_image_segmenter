# get\_dataset.md

This document explains how to download and prepare the dataset required for running this project from the Kaggle competition.

## Prerequisites

1. **Kaggle account**: You need a Kaggle account to access the competition data. If you don't have one, register at [https://www.kaggle.com](https://www.kaggle.com).
2. **Kaggle API token**: Generate an API token:

   * Go to your Kaggle account settings ([https://www.kaggle.com/\`](https://www.kaggle.com/`)<your-username>\`/account).
   * Scroll to **API** and click **Create New API Token**.
   * A file named `kaggle.json` will be downloaded.
3. **Kaggle CLI**: Install the Kaggle command-line interface:

   ```bash
   pip install kaggle
   ```
4. **Place the token**: Move `kaggle.json` to the configuration directory:

   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/downloaded/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Downloading the Dataset

### Using Kaggle CLI

1. **Download competition data**:

   ```bash
   kaggle competitions download \
     -c opencv-pytorch-segmentation-project-round2 \
     -p ./dataset/
   ```

   This will download a zip file (e.g., `opencv-pytorch-segmentation-project-round2.zip`) into the project's root folder.
2. **Unzip the files**:

   ```bash
   cd dataset/
   unzip opencv-pytorch-segmentation-project-round2.zip
   ```
3. **Verify file structure**. After extraction, you should see:

   ```text
   dataset/
   ├── mask/         # segmented mask
   ├── imgs/         # images
   ├── train.csv     # Train Dataset Images
   ├── test.csv      # Test Dataset Images
   └── sampleSubmission.csv 
   
   ```

### Alternative Download (Without Kaggle CLI)

1. Open your web browser and navigate to the competition page:
   [https://www.kaggle.com/competitions/opencv-pytorch-segmentation-project-round2](https://www.kaggle.com/competitions/opencv-pytorch-segmentation-project-round2)
2. Log in with your Kaggle account and accept the competition rules.
3. On the **Data** tab, click **Download All** to download the ZIP file.
4. Place the downloaded ZIP into project's root directory:

   ```bash
   mkdir dataset/
   mv ~/Downloads/opencv-pytorch-segmentation-project-round2.zip ./dataset/
   ```
5. Extract the archive:

   ```bash
   cd dataset/
   unzip opencv-pytorch-segmentation-project-round2.zip
   ```
6. Confirm you have:

   ```text
   dataset/
   ├── mask/         # segmented mask
   ├── imgs/         # images
   ├── train.csv     # Train Dataset Images
   ├── test.csv      # Test Dataset Images
   └── sampleSubmission.csv 
   
   ```
