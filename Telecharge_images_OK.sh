
#!/bin/bash

CONFIGNAME=$1
. include_maconfig.sh $CONFIGNAME


echo 'Raw images download'
cd $BRUTES_DIR
pwd
sed -i "s/XXXXXXXX/$mylogin/g" ImageS3.sh
sed -i "s/%%%%%%%%/$mypassword/g" ImageS3.sh
bash ImageS3.sh
echo '====>>> End of raw image download'

