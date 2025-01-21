#!/bin/bash
#Ejecuta toda la red neuronal

#Importacion de script que contabiliza el tiempo
source time.sh

## Inicia conteo de tiempo
start_time=$(date +%s)
cd /mnt/c/Users/fabia/Desktop/Sargazo/CNN_Sargasse/Extraction_Sargasses_S3_2023/SOURCES

echo "Procesamiento de imagenes sentinel-3"
#Codigos
echo "#-------------- Create_12B_OK.sh--------------#"
bash Create_12B_OK.sh



echo "#-------------- Detection_Sargasses_OK.sh--------------#"
bash Detection_Sargasses_OK.sh


echo "#-------------- SargassesGeoRef_OK.sh--------------#"
bash SargassesGeoRef_OK.sh

## Finaliza conteo de tiempo
end_time=$(date +%s)

##Calculo de tiempo
elapsed_seconds=$((end_time - start_time))
elapsed_time=$(timef $elapsed_seconds)

echo "Tiempo transcurrido: $elapsed_time"