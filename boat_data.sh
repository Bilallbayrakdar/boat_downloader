gdown 1qbn-ftvHdcgiZFi68vTRfVDt0k5-CCEB
gdown 1PuS6k2gGqN_wEndMdPX9ijpqn-K1KJy_
gdown 1qmPcyKGF11pVvJYLJF2R0RwDj5NEHdEr

unzip ./Bosch1_11Agustos_Omer.zip
unzip ./Bosch2_11Agustos_Omer.zip
unzip ./OneDrive_6_9-6-2022.zip -d ./test

mkdir ./images
mv ./test/* ./images/
mv ./Bosch1_11Agustos_Omer ./images/Bosch1_11Agustos_Omer
mv ./Bosch2_11Agustos_Omer ./images/Bosch2_11Agustos_Omer

rm -rf ./OneDrive_6_9-6-2022
rm -rf ./Bosch1_11Agustos_Omer.zip
rm -rf ./Bosch2_11Agustos_Omer.zip
rm -rf ./OneDrive_6_9-6-2022.zip

rm -rf ./datasets
rm -rf ./datasets/boats
rm -rf ./datasets/boats/train_images
rm -rf ./datasets/boats/train_gts
rm -rf ./datasets/boats/test_images
rm -rf ./datasets/boats/test_gts

mkdir ./datasets
mkdir ./datasets/boats
mkdir ./datasets/boats/train_images
mkdir ./datasets/boats/train_gts
mkdir ./datasets/boats/test_images
mkdir ./datasets/boats/test_gts



