#!/bin/bash

#SBATCH --job-name=Detec_S3
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mlaval@osupytheas.fr
#SBATCH --partition=seq
#SBATCH --output=output.log
#SBATCH --error=error.log


python3 applySargasumNetFai21B.py --input  ../../Image12B/JacquesV15/S3B_OL_1_EFR____20190129T133807_20190129T134107_20200111T174613_0179_021_238_2700_MR1_R_NT_002/S3B_OL_1_EFR____20190129T133807_20190129T134107_20200111T174613_0179_021_238_2700_MR1_R_NT_002_21b.tif --output ImageTest.tif --weights Sargassum21b_128_FAI_V15_Aug4.pth
