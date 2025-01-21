#!/bin/bash

CONFIGNAME=$1

. ./Sources/include_maconfig.sh $CONFIGNAME

if [ ! -d ${B12_DIR} ];then
    echo "Création du dosser1 !";
    mkdir -p ${B12_DIR}
fi

cd $BRUTES_DIR

if [ -f output_temp.txt ]; then
        rm output_temp.txt
fi
if [ -f output_BruteDir.txt ]; then
        rm output_BruteDir.txt
fi

ls -lh S3* > output_temp.txt

while read LINE;do
        echo $LINE >& line.txt
        size=`awk -F ' ' '{print $5}' line.txt`
        filename=`awk -F ' ' '{print $9}' line.txt`
        #echo $filename
        #echo ${filename: -2}
            if [ "${filename: -2}" = "ip" ]; then
        if [ $size = 0 ]; then
                # tant qu'on a des fichiers de taille nulle, on continue
                echo 0
        else
                echo ${filename:0: -9} >> output_BruteDir.txt
        fi
        fi
done < output_temp.txt


cd ${B12_DIR}
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
          if [ "${filename: -2}" != "N3" ]; then
            if [ $size -le 23 ]; then
                # tant qu'on a des fichiers de taille nulle, on continue
                echo 0
            else
                filename=`awk -F ' ' '{print $9}' line.txt`
                echo $filename >> output_B12Dir.txt
            fi
          fi
        fi
done < output_temp.txt

# Comparaison output_B12Dir.txt et output_BruteDir
cp $BRUTES_DIR/output_BruteDir.txt output_BruteDir.txt

echo Cristele
if [ -f output_B12Dir.txt ]; then
   while read filenameOK;do
        lineOK=`grep -n $filenameOK output_BruteDir.txt`
        numline=`echo $lineOK | awk -F ':' '{print $1}'`
        echo numero:$numline
        sed -i ${numline}d output_BruteDir.txt
   done < output_B12Dir.txt
fi

cp ${SOURCES_DIR}/include_maconfig.sh .

while read filename;do
        filenametot=${BRUTES_DIR}/${filename}.SEN3.zip
	echo "file name: $filenametot"
        pwd
	bash ${SOURCES_DIR}/sentinel3OLCI_EFR_GTIFF.sh ${filenametot}
done < output_BruteDir.txt

