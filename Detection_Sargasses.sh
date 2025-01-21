#!/bin/bash

CONFIGNAME=$1

. ./Sources/include_maconfig.sh $CONFIGNAME

mkdir -p  ${SARGASSES_DIR}/
if [ ! -d ${SARGASSES_DIR} ];then
    echo "Création du dosser1 !";
    mkdir -p ${SARGASSES_DIR}
fi

cd $B12_DIR

if [ -f output_temp.txt ]; then
        rm output_temp.txt
fi
if [ -f output_B12Dir.txt ]; then
        rm output_B12Dir.txt
fi

ls -lh > output_temp.txt

while read LINE;do
        echo $LINE >& line.txt
        size=`awk -F ' ' '{print $5}' line.txt`
        filename=`awk -F ' ' '{print $9}' line.txt`
        if [ "${filename:0:2}" = "S3" ]; then
	echo CC1
	echo ${filename:0:2}
	echo ${filename: -2}
          if [ "${filename: -2}" != "N3" ]; then
	echo CC2
            if [ $size -le 23 ]; then
                echo 'taille :'  $size
            else
                echo 'OK -- Taille :'  $size $filename
                filename=`awk -F ' ' '{print $9}' line.txt`
                echo $filename >> output_B12Dir.txt
            fi
          fi
        fi
done < output_temp.txt

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

# Comparaison output_SargDir.txt et output_B12Dir
cp $B12_DIR/output_B12Dir.txt output_B12Dir.txt

if [ -f output_SargDir.txt ]; then
   while read filenameOK;do
        lineOK=`grep -n $filenameOK output_B12Dir.txt`
        numline=`echo $lineOK | awk -F ':' '{print $1}'`
        #echo numero:$numline
        sed -i ${numline}d output_B12Dir.txt
   done < output_SargDir.txt
fi


while read filename;do
        echo Cris
        totalfilename=${B12_DIR}/${filename}/${filename}_21b.tif > totalfilename
        #ls ${B12_DIR}/$filename/*_21b.tif > totalfilename
	cp ${SOURCES_DIR}/include_maconfig.sh .
	echo $totalfilename
	#filename=`basename $totalfilename`
	echo $filename
        #poids par defaut V3
        #python3 ${SOURCES_DIR}/applySargasumNetFai21B.py --input $totalfilename --output ${SARGASSES_DIR}/$filename
	#poids par defaut V9
        #python3 ${SOURCES_DIR}/applySargasumNetFai21B_v2.py --input $totalfilename --output ${SARGASSES_DIR}/$filename --weights ${SOURCES_DIR}/Sargassum21b_128_FAI_V9_Aug4.pth
        # poids V15
        #python3 ${SOURCES_DIR}/applySargasumNetFai21B_v2.py --input $totalfilename --output ${SARGASSES_DIR}/$filename --weights ${SOURCES_DIR}/Sargassum21b_128_FAI_V15_Aug4.pth --thresold 0
	
	# DMCI
	echo ${SOURCES_DIR}/Sargassum2D_SEN3_DMCI/applydMci.py input $totalfilename output ${SARGASSES_DIR}/${filename}_21b.tif weights ${SOURCES_DIR}/Sargassum2D_SEN3_DMCI/Sargassum21b_128_DMCI.pth
	python3 ${SOURCES_DIR}/Sargassum2D_SEN3_DMCI/applydMci.py --input $totalfilename --output ${SARGASSES_DIR}/${filename}_21b.tif --weights ${SOURCES_DIR}/Sargassum2D_SEN3_DMCI/Sargassum21b_128_DMCI.pth

done < output_B12Dir.txt
