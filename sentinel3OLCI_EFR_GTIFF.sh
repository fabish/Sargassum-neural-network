#!/bin/bash

echo "ATTENTION PAS DE PYTHON-ENV"
echo

function nc2tif() {
	
	
  echo "nc2tif $1 $2"
  
  nompath=$1
  base=`basename $nompath`
  nomf=${base%%.*}
  extf=${base##*.}
	
 # echo $base	
 # echo $nomf
 # echo  $extf
   	
	
	
  if [ ! -e $2/${nomf}.tif ] ; then
	
	echo "/opt/snap/bin/pconvert  -f tif -b 1 -o $2 $1 "
	#echo "gdalwarp -overwrite -s_srs EPSG:32630 -r cubic -ts 10980 10980 -of GTiff -ot Uint16 $1 $2/${nomf}.tif"
	#gdalwarp -overwrite -s_srs EPSG:32630 -r cubic -ts 10980 10980 -of GTiff -ot Uint16 $1 $2/${nomf}.tif
#	gdal_translate -of GTiff $1 $2/${nomf}.tif
#	/opt/snap/bin/pconvert  -f tif -b 1 -o /tmp $1 
	gdal_translate -of GTiff   $1  $2/${nomf}.tif 
  fi	
  
  

}


 


if [ $# -eq 0 ] ; then
   echo "$0 usage file.zip "
   exit 1
fi
 

file=$1

base=`basename $file`
nom=${base%%.*}
ext=${base##*.}

rep=${nom}".SEN3"
echo "REP:$rep"

echo "EXT:$ext" 

 
if [ "X$ext" == "Xzip" -o "X$ext" == "XZIP" ] ; then
	if [ ! -d $rep ] ; then
		unzip  $file 
	fi    
else
   if [ "X$ext" != "XSEN3" ] ; then
	echo "$0 usage file.zip OR rep.SEN3"
	exit 1
   fi	
fi


if [ !  -d $nom ] ; then
  mkdir $nom
fi

 
size=`gdalinfo $rep/Oa01_radiance.nc | grep Size | cut -f 3,4 -d " "`
sizeX=`echo $size | cut -f 1 -d ','`
sizeY=`echo $size | cut -f 2 -d ' '`

echo "SIZE $sizeX $sizeY"
 
  
 
for img in `ls $rep/Oa*radiance.nc` ; do
   echo $img
   
   nc2tif  $img $nom $sizeX sizeY
done


mkdir ./tmp
mkdir ./tmp/$$
echo
echo "Merge21"
ls $nom/Oa*radiance.tif > ./tmp/$$/merge.txt
echo "gdal_merge.py -separate -ot Uint16 -of GTiff -o $nom/$nom.tif"  
gdal_merge.py -separate -ot Uint16 -of GTiff -o $$_21b.tif --optfile ./tmp/$$/merge.txt
mv $$_21b.tif $nom/${nom}_21b.tif

 
 echo IRC 1-b


ls -1 $nom/*Oa17_radiance.tif  > ./tmp/$$/merge.txt
ls -1 $nom/*Oa06_radiance.tif >> ./tmp/$$/merge.txt
ls -1 $nom/*Oa03_radiance.tif >> ./tmp/$$/merge.txt
 
cat ./tmp/merge.txt  
#ls -1 $nom/tmp/*B08.tif $nom/tmp/*B04.tif $nom/tmp/*B03.tif   > ./tmp/merge.txt
gdal_merge.py -separate -ot Uint16 -of GTiff -o $nom/${nom}_irc16.tif --optfile ./tmp/$$/merge.txt
echo ITC  8b
gdal_translate -of GTiff -ot Byte -scale 0 32560 0 255 $nom/${nom}_irc16.tif $nom/${nom}_irc8.tif
rm  -f $nom/${nom}_irc16.tif
rm -r ./tmp/$$
