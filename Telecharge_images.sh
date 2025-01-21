#!/bin/bash
#set -x 

CONFIGNAME=$1
. ./Sources/include_maconfig.sh $CONFIGNAME


echo 'Telechargement Images Brutes S3'
pwd
cd $BRUTES_DIR
sed -i "s/XXXXXXXX/$mylogin/g" ImageS3_ori.sh
sed -i "s/%%%%%%%%/$mypassword/g" ImageS3_ori.sh
cd $SOURCES_DIR
echo 'Recherche Fichiers Manquants'
bash modify_ImageS3_for_only_files_empty.sh $CONFIGNAME
echo 'Telecharge Fichiers Manquants'
cd $BRUTES_DIR
bash ImageS3.sh
cd $ROOTDIR
echo '====>>> fin telechargement Images Brutes'



