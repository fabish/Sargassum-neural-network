#!/bin/bash

CONFIGNAME=$1

. ./Sources/include_maconfig.sh $CONFIGNAME

mkdir -p  ${GEOREF_DIR}/

# Construction fichier donnees dans Sargasses dir

cd ${SARGASSES_DIR}/
if [ -f output_temp.txt ]; then
         rm output_temp.txt
fi
if [ -f output_SargDir.txt ]; then
        rm output_SargDir.txt
fi

ls -lh S3* > output_temp.txt
while read LINE;do
       echo $LINE >& line.txt
       size=`awk -F ' ' '{print $5}' line.txt`
       if [ $size = 0 ]; then
               # tant qu'on a des fichiers de taille nulle, on continue
               echo 0
       else
               filename=`awk -F ' ' '{print $9}' line.txt`
               echo "${filename:0: -8}" >> output_SargDir.txt
       fi
done < output_temp.txt

# Construction fichier donnees dans GeoRef 
cd ${GEOREF_DIR}/
if [ -f output_temp.txt ]; then
        rm output_temp.txt
fi
if [ -f output_GeoRefDir.txt ]; then
        rm output_GeoRefDir.txt
 fi

ls -lh S3* > output_temp.txt
while read LINE;do
         echo $LINE >& line.txt
         size=`awk -F ' ' '{print $5}' line.txt`
         if [ $size = 0 ]; then
                 # tant qu'on a des fichiers de taille nulle, on continue
                 echo 0
         else
                 filename=`awk -F ' ' '{print $9}' line.txt`
                 echo "${filename:0: -8}" >> output_GeoRefDir.txt
         fi
done < output_temp.txt

# Comparaison output_SargDir.txt et output_B12Dir
cp $SARGASSES_DIR/output_SargDir.txt output_SargDir.txt

if [ -f output_GeoRefDir.txt ]; then
   while read filenameOK;do
        lineOK=`grep -n $filenameOK output_GeoRefDir.txt`
        numline=`echo $lineOK | awk -F ':' '{print $1}'`
        echo numero:$numline
        sed -i ${numline}d output_GeoRefDir.txt
   done < output_SargDir.txt
fi

cd ${GEOREF_DIR}/

while read totalfilename;do
        cp ${SOURCES_DIR}/include_maconfig.sh .
        #echo $totalfilename '.tif' > totalfilename
        echo 'totalfilename:' $totalfilename
        totalfilenametif=${SARGASSES_DIR}/${totalfilename}_21b.tif 
        echo 'totalfilename:' $totalfilenametif
        #filename=`basename $totalfilename`
        filename=${totalfilename}_21b.tif 
        echo 'filename:' $filename
	#dirname="${filename%.*}"
	dirname=${totalfilename}
	echo 'dirname:' $dirname
	
	cp ${B12_DIR}/$dirname.SEN3/geo_coordinates.nc .
#	bash ${SOURCES_DIR}/georefS3_V3.sh $totalfilename $filename
	echo ${SOURCES_DIR}/georefS3_V3_Float32.sh $totalfilenametif $filename
	bash ${SOURCES_DIR}/georefS3_V3_Float32.sh $totalfilenametif $filename
done < output_SargDir.txt
