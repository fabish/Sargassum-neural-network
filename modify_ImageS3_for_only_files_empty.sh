#!/bin/bash

#==================================================================
# recuperation d'un listing avec les fichiers S3* de taille non nulle
# et mise a jour du script ImageS3 avec uniquement les fichiers non 
# telecharges
# output:ImageS3.sh maj
#==================================================================

# lecture du listing des noms de fichiers zip 
# contenu dans output_total.txt obtenu:

CONFIGNAME=$1
. ./include_maconfig.sh $CONFIGNAME

cd $BRUTES_DIR

if [ -f output_total.txt ]; then
	rm output_total.txt
fi
if [ -f output_OK.txt ]; then
	rm output_OK.txt
fi

ls -lh S3* > output_total.txt

echo 1 > $ROOTDIR/continue.txt
while read LINE;do 
	echo $LINE >& line.txt
	size=`awk -F ' ' '{print $5}' line.txt`
	if [ $size = 0 ]; then
                # tant qu'on a des fichiers de taille nulle, on continue
                echo 0 > $ROOTDIR/continue.txt
	else
	        filename=`awk -F ' ' '{print $9}' line.txt`
		echo $filename >> output_OK.txt
	fi
done < output_total.txt



# read and compare each line of imageS3.sh to the database of output_OK.sh
cp ImageS3_ori.sh ImageS3.sh
while read filenameOK;do
	lineOK=`grep -n $filenameOK ImageS3.sh`
	numline=`echo $lineOK | awk -F ':' '{print $1}'`
#	echo numero:$numline
        sed -i ${numline}d ImageS3.sh
done < output_OK.txt

echo "============================================================================"
echo "Mise a jour de ImageS3.sh avec uniquement les fichiers restant a telecharger"
echo "============================================================================"

cd $ROOTDIR


