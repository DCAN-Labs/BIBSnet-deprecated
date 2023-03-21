# BIBSnet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7106148.svg)](https://doi.org/10.5281/zenodo.7106148)

Quickly and accurately segments an optimally-aligned T1 and T2 pair with a deep neural network trained via nnU-Net and SynthSeg with a large 0 to 8 month old infant MRI brain dataset.

## Command-Line Arguments
```
usage: run.py [-h] --input INPUT --output OUTPUT [--nnUNet NNUNET]
  [--task TASK] [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Valid path to existing input directory following valid
                        nnU-Net naming conventions (T1w files end with
                        _0000.nii.gz and T2w end with _0001.nii.gz). There
                        should be exactly 1 T1w file and exactly 1 T2w file in
                        this directory.
  --output OUTPUT, -o OUTPUT
  --nnUNet NNUNET, -n NNUNET
                        Valid path to existing executable file to run nnU-
                        Net_predict. By default, this script will assume that
                        nnU-Net_predict will be in the same directory as this
                        script:
  --task TASK, -t TASK  Task ID, which should be a 3-digit positive integer
                        starting with 5 (e.g. 512).
--model MODEL, -m MODEL
```

## BIBSnet Segmentation Models

`data/models.csv` lists all available BIBSnet models to run. Below are the default BIBSnet models, all trained on manually-segmented 0- to 8-month-old BCP subjects' segmentations. 

| Model | Description |
|:-:|:--|
| 512 | Model Used within BIBSNet manuscript |
| 551 | Default T1w and T2w model |
| 514 | Default T1w-only model |
| 515 | Default T2w-only model |

Additionally, see the "location" column within `data/models.csv` to download the models. The models are publically available and can be pulled down with CLI tools such as wget. For example to pull down model 512 something like this work: `wget https://s3.msi.umn.edu/CABINET_data/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip`

<br />