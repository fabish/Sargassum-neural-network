#!/bin/bash

CONFIGNAME=$1

. include_maconfig.sh $CONFIGNAME

mkdir -p ${B12_DIR}

cd ${B12_DIR}

cp ${SOURCES_DIR}/include_maconfig.sh .

for filename in ${BRUTES_DIR}/*.SEN3.zip
do
	echo "file name: $filename"
	bash ${SOURCES_DIR}/sentinel3OLCI_EFR_GTIFF.sh $filename
done



