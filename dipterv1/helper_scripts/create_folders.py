import os

os.makedirs("C:/Users/beni1/Desktop/Önlab/spectograms5s_v2_overlap4s/train")
os.makedirs("C:/Users/beni1/Desktop/Önlab/spectograms5s_v2_overlap4s/test")

genres = 'air_conditioner car_horn children_playing dog_bark drilling engine_idling gun_shot jackhammer siren street_music'
genres = genres.split()
for g in genres:
  path_audio = os.path.join('/content/audio3sec',f'{g}')
  os.makedirs(path_audio)
  path_train = os.path.join("C:/Users/beni1/Desktop/Önlab/spectograms5s_v2_overlap4s/train",f'{g}')
  path_test = os.path.join("C:/Users/beni1/Desktop/Önlab/spectograms5s_v2_overlap4s/test",f'{g}')
  os. makedirs(path_train)
  os. makedirs(path_test)