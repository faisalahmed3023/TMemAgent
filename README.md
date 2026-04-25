## How to Create Data

Before running this file, first download the **review** and **meta** files for the corresponding domains from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/). Then place those downloaded `.jsonl` files in the same directory as this script.

This code uses the raw Amazon review and metadata files to generate the processed data files required for the analysis.

### Steps

1. Download the required domain-wise **review** and **meta** files from the Amazon Reviews 2023 dataset.
2. Put all downloaded files in the same folder as `run_data.py`.

Run the script:

```bash
python run_data.py
```

## Load Data

After generating the processed files, load the dataset using the `Data_Structure` class available in `Data.py`.

## Train Recommender Model

After loading the data, create the recommender environment using `recommendation_model.py`.

The supported recommender models include:

- `MF`
- `LightGCN`
- `SASRec`

An example of the recommender environment setup is provided in `environment.py`.
