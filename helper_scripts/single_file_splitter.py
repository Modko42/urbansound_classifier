import os
from pydub import AudioSegment

song = "Z:/Egyetem/önlab2_msc/raw_audio/test_songs/mountainking.wav"
#number_of_splits = int(input("Number of splits: "))
number_of_splits = 20

filename = song.split('/')[-1].split('.')[0]

for w in range(0, number_of_splits):
    t1 = 3 * (w) * 1000
    t2 = t1 + 6000
    new_audio = AudioSegment.from_wav(song)
    new = new_audio[t1:t2]
    new.export(f'Z:/Egyetem/önlab2_msc/testing/new/mountainking/audio_files/{filename+ "_" + str(t1)}.wav',
               format="wav")

