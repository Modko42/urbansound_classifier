import os
import random
import shutil

genres = 'blues classical country disco hiphop metal pop reggae rock'
genres = genres.split()

directory = "E:/temp_location/train/"
for g in genres:
  filenames = os.listdir(os.path.join(directory,f"{g}"))
  for f in filenames:
    random.shuffle(filenames)
  test_files = filenames[0:45]

  for f in test_files:

    shutil.move(directory + f"{g}"+ "/" + f,"E:/temp_location/test/" + f"{g}")