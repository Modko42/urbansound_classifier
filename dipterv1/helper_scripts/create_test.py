import os
import random
import shutil
import statistics

import audioread

genres = 'air_conditioner car_horn children_playing dog_bark drilling engine_idling gun_shot jackhammer siren street_music'
genres = genres.split()

directory = "E:/temp_location/v26/train/"
for g in genres:
    genre_avg = []
    filenames = os.listdir(os.path.join(directory, f"{g}"))
    # for f in filenames:
    #     if f.split('.')[-1] == 'wav':
    #         with audioread.audio_open(directory + f"{g}" + "/" + f) as file:
    #             genre_avg.append(file.duration)
    #         # random.shuffle(filenames)
    #
    # print(g + '  ' + str(round(len(filenames) * 0.1)))
    # print(g + '  avg length ' + str(round(statistics.mean(genre_avg), 2)))

    test_files = filenames[0:round(len(filenames) * 0.1)]
    for f in test_files:
        shutil.move(directory + f"{g}" + "/" + f, "E:/temp_location/v26/test/" + f"{g}")
