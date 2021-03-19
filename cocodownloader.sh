#!/bin/sh

start = 'date + %s'

echo "Prepare to download train-val2017 anotation zip file..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

echo "Prepare to download train2017 image zip file..."
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm -f train2017.zip

echo "Prepare to download test2017 image zip file..."
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm -f val2017.zip

end = 'date + %s'
runtime = $((end - start))

echo "Download completed in " $runtime  " second"1~#!/bin/sh

echo "Prepare to download VQA train annotations2017 image zip file..."
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
rm -f v2_Annotations_Train_mscoco.zip
echo "Prepare to download VQA validation annotations2017 image zip file..."
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
rm -f v2_Annotations_Val_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
rm -f v2_Questions_Train_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
rm -f v2_Questions_Val_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip v2_Questions_Test_mscoco.zip
rm -f v2_Questions_Test_mscoco.zip
wget -c http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm -f train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm -f val2014.zip
wget -c http://images.cocodataset.org/zips/test2015.zip
unzip test2015.zip
rm -f test2015.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip
unzip v2_Complementary_Pairs_Train_mscoco.zip
rm -f v2_Complementary_Pairs_Train_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip
unzip v2_Complementary_Pairs_Val_mscoco.zip
rm -f v2_Complementary_Pairs_Val_mscoco.zip

end = 'date + %s'
runtime = $((end - start))

git clone https://github.com/openai/CLIP.git
echo "Download completed in " $runtime  " second"

