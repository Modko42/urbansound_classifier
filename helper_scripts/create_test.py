import os
import random
import shutil

genres = 'air_conditioner car_horn children_playing dog_bark drilling engine_idling gun_shot jackhammer siren street_music'
genres = genres.split()

directory = "E:/temp_location/train/"
for g in genres:
  filenames = os.listdir(os.path.join(directory,f"{g}"))
  for f in filenames:
    random.shuffle(filenames)
  test_files = filenames[0:45]

  for f in test_files:

    shutil.move(directory + f"{g}"+ "/" + f,"E:/temp_location/test/" + f"{g}")