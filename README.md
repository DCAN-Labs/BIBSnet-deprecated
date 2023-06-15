# BIBSnet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7106148.svg)](https://doi.org/10.5281/zenodo.7106148)

Quickly and accurately segments an optimally-aligned T1 and T2 pair with a deep neural network trained via nnU-Net and SynthSeg with a large 0 to 8 month old infant MRI brain dataset. Please note that this only runs BIBSnet, not pre- or post-BIBSnet. 

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

## Inputs and Outputs
If there is both a T1 and T2, BIBSnet expects them to be aligned with each other, otherwise the segmentation will not work properly. 

Outputs may need to be chirality corrected, which can be done with post-BIBSnet. The outputs will be in the space that the anatomical was aligned with, i.e. if the T1 was aligned to the T2, the output will be in T2 space and vice versa. 

## Container
When running CABINET using a GPU, the job typically takes about 4 minutes, 2 tasks, and one node with 20 gb of memory to run effectively.

This has been primarily tested in Singularity. We are less able to provide technical support for Docker execution.

### Singularity

#### Download
`singularity pull docker://dcanumn/BIBSNet`

#### Usage
```
singularity run --nv --cleanenv --no-home \
-B /path/to/input:/input \
-B /path/to/output:/output \
/path/to/BIBSNet.sif \
--input /input --output /output --task <task ID> --model 3d_fullres 
```

## BIBSnet Segmentation Models

For choosing the model, be sure to choose the model according to what anatomicals you are providing. The task id you provide with the --task flag needs to correlate with the model number.

`data/models.csv` lists all available BIBSnet models to run. Below are the default BIBSnet models, all trained on manually-segmented 0- to 8-month-old BCP subjects' segmentations. 

| Model | Description |
|:-:|:--|
| 512 | Model Used within BIBSNet manuscript |
| 551 | Default T1w and T2w model |
| 514 | Default T1w-only model |
| 515 | Default T2w-only model |

Additionally, see the "location" column within `data/models.csv` to download the models. The models are publically available and can be pulled down with CLI tools such as wget. For example to pull down model 512 something like this work: `wget https://s3.msi.umn.edu/CABINET_data/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip`

<br />
