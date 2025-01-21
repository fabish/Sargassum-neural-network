#!/bin/bash

CONFIGNAME=$1

. include_maconfig.sh $CONFIGNAME

mkdir -p  ${GEOREF_DIR}/

cd ${GEOREF_DIR}/

for totalfilename in ${SARGASSES_DIR}/*.tif
do
        cp ${SOURCES_DIR}/include_maconfig.sh .
        echo 'totalfilename:' $totalfilename
        filename=`basename $totalfilename`
        echo 'filename:' $filename
	#dirname="${filename%.*}"
	dirname=${filename::-8}
	echo 'dirname:' $dirname
	
	cp ${B12_DIR}/$dirname.SEN3/geo_coordinates.nc .
#	bash ${SOURCES_DIR}/georefS3_V3.sh $totalfilename $filename
	echo ${SOURCES_DIR}/georefS3_V3_Float32.sh $totalfilename $filename
	bash ${SOURCES_DIR}/georefS3_V3_Float32.sh $totalfilename $filename

done
