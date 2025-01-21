#wget --quiet --no-check-certificate  --user=mlaval --password=Sargassescop.7  --continue --output-document=./S3A_OL_1_EFR____20191220T135127_20191220T135427_20191220T153134_0179_053_010_2700_LN1_O_NR_002.SEN3.zip "https://apihub.copernicus.eu/apihub/odata/v1/Products('4c8ca749-cfb3-4259-9b17-9b937614f5c1')/\$value"

#Descarga
wget --quiet --no-check-certificate --continue  --user=orangefire123 --password=1xa75y38q7   --output-document=/mnt/c/Users/upa.gonzalez.irvin/desktop/Sargazo/CNN_Sargasse/S3B_OL_1_EFR____20190330T142306_20190330T142606_20200112T052821_0180_023_324_2700_MR1_R_NT_002.SEN3.zip "https://apihub.copernicus.eu/apihub/odata/v1/Products('60f5ae14-57aa-4c72-864e-69cb4190c75f')/\$value"

#Busqueda
#wget --no-check-certificate  --user=mlaval --password=Sargassescop.7 --output-document=./encontrados.txt "https://scihub.copernicus.eu/dhus/api/stub/products?filter=S3B_OL_1_EFR____20190330T142306_20190330T142606_20200112T052821_0180_023_324_2700_MR1_R_NT_002&offset=0&limit=25&sortedby=ingestiondate&order=desc"
#wget --no-check-certificate  --user=mlaval --password=Sargassescop.7 --output-document=./encontrados.txt "https://scihub.copernicus.eu/dhus/search?q=ingestiondate:[NOW-1DAY TO NOW] AND producttype:SLC&rows=100&start=0&format=xml"

#Busqueda de una sola imagen por nombre
#wget --no-check-certificate  --user=mlaval --password=Sargassescop.7 --output-document=./encontrados.txt "https://scihub.copernicus.eu/dhus/search?q=identifier:S3B_OL_1_EFR____20190330T142306_20190330T142606_20200112T052821_0180_023_324_2700_MR1_R_NT_002&format=xml"