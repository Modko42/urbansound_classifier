import os.path
import shutil

import pandas
import pandas as pd

data = pd.read_csv('E:/urbandsounds8k/metadata/UrbanSound8K.csv')

main_path = 'E:/urbandsounds8k/all_files/'
classes_path = 'E:/urbandsounds8k/classes/'
counter = 0
whoops_counter = 1

for index,row in data.iterrows():
    try:
        print(str(counter) + row['slice_file_name'] +"  "+ row['class'])
        counter += 1
        shutil.copy(os.path.join(main_path,row['slice_file_name']), os.path.join(classes_path,row['class'],row['slice_file_name']))
    except:
        print("whoops"+str(whoops_counter))
        whoops_counter += 1