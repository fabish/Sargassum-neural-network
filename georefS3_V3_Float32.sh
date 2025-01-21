#/bin/bash


if [ $# -lt 2  ] ; then
   echo "$0 usage   file  output"
   exit 1
fi
 
file=$1
output=$2

if [ !  -f geo_coordinates.nc ] ; then
  echo "manque le fichier geo_coordinates.nc  (voir lans le repertoire .SEN3"
  exit 2
fi  

base=`basename $1`
nom=$base
gdal_translate -of VRT NETCDF:"geo_coordinates.nc":latitude lat.vrt
gdal_translate -of VRT NETCDF:"geo_coordinates.nc":longitude lon.vrt



 

ls  $file
cat << EOF > coord.vrt
<VRTDataset rasterXSize="4865" rasterYSize="4090">
 <metadata domain="GEOLOCATION">
 <mdi key="X_DATASET">lon.vrt</mdi>
 <mdi key="X_BAND">1</mdi>
 <mdi key="Y_DATASET">lat.vrt</mdi>
 <mdi key="Y_BAND">1</mdi>
 <mdi key="PIXEL_OFFSET">0</mdi>
 <mdi key="LINE_OFFSET">0</mdi>
 <mdi key="PIXEL_STEP">1</mdi>
 <mdi key="LINE_STEP">1</mdi>
 </metadata> 
 
   <VRTRasterBand dataType="Float32" band="1"> 

    <SimpleSource>
      <SourceFilename relativeToVRT="1">$file</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="4865" RasterYSize="4091" dataType="Float32"/>
      <SrcRect xOff="0" yOff="0" xSize="4865" ySize="4091" />
      <DstRect xOff="0" yOff="0" xSize="4865" ySize="4091" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
EOF

function convert(){
	pre=9
	val=$1
	nbl=`echo ${#val}`
	ent=`echo ${val::-pre}`
	let pos=$nbl-$pre
	dec=`echo ${val:$pos:pre}`
	echo ${ent}.$dec
  
}

cat coord.vrt


echo "gdalwarp -geoloc -t_srs EPSG:4326 coord.vrt $output -srcnodata 0 -overwrite" 

time gdalwarp -multi -geoloc -t_srs EPSG:4326 coord.vrt /tmp/geo$$.tif -srcnodata 0 -overwrite


gdalinfo  /tmp/geo$$.tif > /tmp/info$$.txt

ul=`grep  'Upper Left' /tmp/info$$.txt`
lon=`echo $ul | sed 's/(//' |sed 's/)//'|sed 's/,/ /'|sed 's/\.//g' | sed  's/ \+/ /g' | cut -d " " -f 3`
lon=`convert $lon`
lat=`echo $ul | sed 's/(//' |sed 's/)//'|sed 's/,/ /'|sed 's/\.//g' | sed 's/ \+/ /g' | cut -d " " -f 4`
lat=`convert $lat`


lr=`grep  'Lower Right' /tmp/info$$.txt`
lon2=`echo $lr | sed 's/(//' |sed 's/)//'|sed 's/,/ /'|sed 's/\.//g'  | sed 's/ \+/ /g' |cut -d " " -f 3`
lon2=`convert $lon2`
lat2=`echo $lr | sed 's/(//' |sed 's/)//'|sed 's/,/ /'|sed 's/\.//g'  | sed 's/ \+/ /g' | cut -d " " -f 4`
lat2=`convert $lat2`

echo $lon $lat $lon2 $lat2
 
 
 


gdal_translate -of GTiff -a_ullr $lon $lat $lon2 $lat2 -a_srs  EPSG:4326 /tmp/geo$$.tif  $output


echo $output


# tranform
# gdaltransform -s_srs  EPSG:32630  -t_srs  EPSG:4326
