#!/bin/bash

CONFIGNAME=$1

. include_maconfig.sh $CONFIGNAME

mkdir -p  ${SARGASSES_DIR}/

cd ${SARGASSES_DIR}/

for totalfilename in ${B12_DIR}/*/*_21b.tif
do

	cp ${SOURCES_DIR}/include_maconfig.sh .
	echo $totalfilename
	filename=`basename $totalfilename`
	echo $filename
        #poids par defaut V3
        #python3 ${SOURCES_DIR}/applySargasumNetFai21B.py --input $totalfilename --output ${SARGASSES_DIR}/$filename --cuda
	#poids par defaut V9
        #python3 ${SOURCES_DIR}/applySargasumNetFai21B_v2.py --input $totalfilename --output ${SARGASSES_DIR}/$filename --weights ${SOURCES_DIR}/Sargassum21b_128_FAI_V9_Aug4.pth --cuda
        # poids V15
        python3 ${SOURCES_DIR}/applySargasumNetFai21B_v2.py --input $totalfilename --output ${SARGASSES_DIR}/$filename --weights ${SOURCES_DIR}/Sargassum21b_128_FAI_V15_Aug4.pth --thresold 0 --cuda
	
	# DMCI
	#python3 ${SOURCES_DIR}/Sargassum2D_SEN3_DMCI/applydMci.py --input $totalfilename --output ${SARGASSES_DIR}/$filename --weights ${SOURCES_DIR}/Sargassum2D_SEN3_DMCI/Sargassum21b_128_DMCI.pth

done
