import os.path
import shutil

import pandas
import pandas as pd

data = pd.read_csv('E:/urbandsounds8k/metadata/UrbanSound8K.csv')

main_path = "E:/urbandsounds8k/spectograms/v25_4_sound_features_50_10000noBGnoise/train/"
folded_classes_path = 'E:/urbandsounds8k/10fold_cross_val_datasets/'
counter = 0
whoops_counter = 1

for index, row in data.iterrows():
    try:
        print(str(counter) + row['slice_file_name'] + "  " + row['class'])
        counter += 1
        i = 1
        for i in range(1, 11):
            if int(row['fold']) == i:
                shutil.copy(os.path.join(main_path, row['class'], str(row['slice_file_name']).split('.')[0]+".png"),
                            os.path.join(folded_classes_path, str(i),"test",row['class'], str(row['slice_file_name']).split('.')[0]+".png"))
            else:
                shutil.copy(os.path.join(main_path, row['class'], str(row['slice_file_name']).split('.')[0]+".png"),
                            os.path.join(folded_classes_path, str(i), "train", row['class'],
                                         str(row['slice_file_name']).split('.')[0] + ".png"))
    except Exception as e:
        print(e)
        print("whoops" + str(whoops_counter))
        whoops_counter += 1
