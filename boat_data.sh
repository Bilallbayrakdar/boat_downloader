gdown 1qbn-ftvHdcgiZFi68vTRfVDt0k5-CCEB
gdown 1PuS6k2gGqN_wEndMdPX9ijpqn-K1KJy_
gdown 1qmPcyKGF11pVvJYLJF2R0RwDj5NEHdEr
gdown 1Jx9U0VbWotMIm_CtHcYtYvRdDXG0zKmN

mkdir ./images
mkdir ./datasets

unzip ./Bosch1_11Agustos_Omer.zip -d ./images
unzip ./Bosch2_11Agustos_Omer.zip -d ./images
unzip ./OneDrive_6_9-6-2022.zip -d ./images
unzip ./OneDrive_1_9-13-2022.zip -d ./images

mv ./images/12Eylul/yalova_bosch1 ./images/yalova_bosch1_12Eylul
mv ./images/12Eylul/yalova_bosch2 ./images/yalova_bosch2_12Eylul
mv ./images/12Eylul/yalova_hikvision ./images/yalova_hikvision_12Eylul

mv ./images/1Eylul/yalova_bosch1 ./images/yalova_bosch1_1Eylul
mv ./images/1Eylul/yalova_bosch2 ./images/yalova_bosch2_1Eylul
mv ./images/1Eylul/yalova_hikvision ./images/yalova_hikvision_1Eylul

mv './images/20Temmuz/Bosch 1 - Omer _ 20Temmuz' ./images/Bosch1-Omer_20Temmuz
mv './images/20Temmuz/Bosch 2 - Mina _ 20Temmuz' ./images/Bosch2-Mina_20Temmuz
mv './images/20Temmuz/Hikvision - Bilal _ 20Temmuz' ./images/Hikvision-Bilal_20Temmuz

mv './images/4Agustos/Bosch1_4Agustos - Omer' ./images/Bosch1_4Agustos-Omer
mv './images/4Agustos/Bosch2_4Agustos - Omer' ./images/Bosch2_4Agustos-Omer
mv './images/4Agustos/Hik_4Agustos - Omer' ./images/Hik_4Agustos-Omer

rm -rf ./images/20Temmuz
rm -rf ./images/4Agustos
rm -rf ./images/1Eylul
rm -rf ./images/12Eylul

rm -rf ./Bosch1_11Agustos_Omer.zip
rm -rf ./Bosch2_11Agustos_Omer.zip
rm -rf ./OneDrive_6_9-6-2022.zip
rm -rf ./OneDrive_1_9-13-2022.zip