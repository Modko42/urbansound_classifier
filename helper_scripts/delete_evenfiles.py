import os
import glob

files = glob.glob("E:/temp_location/spec3s/**/*.png",recursive=True)

i = 0
sum = 0
for f in files:
    if i % 10 == 0:
        os.remove(f)
        sum = sum + 1
    i = i + 1

print(sum)
