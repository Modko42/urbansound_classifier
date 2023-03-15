import os
from pydub import AudioSegment
import threading

genres = 'blues classical country disco hiphop metal pop reggae rock'
genres = genres.split()


class myThread(threading.Thread):
    def __init__(self, genre):
        threading.Thread.__init__(self)
        self.genre = genre

    def run(self):
        print("Started " + self.genre)
        split_files(self.genre)
        print("Finished " + self.genre)


def split_files(g):
    j = 0
    for filename in os.listdir(os.path.join('Z:/Egyetem/önlab2_msc/raw_audio/yt_dataset_2', f"{g}")):
        song = os.path.join(f'Z:/Egyetem/önlab2_msc/raw_audio/yt_dataset_2/{g}', f'{filename}')
        j = j + 1
        for w in range(0, 1000):
            t1 = 3 * (w) * 1000
            t2 = t1 + 6000
            new_audio = AudioSegment.from_wav(song)
            new = new_audio[t1:t2]
            new.export(f'Z:/Egyetem/önlab2_msc/splitted_audiofiles/audio6sec_overlap3s_yt_dataset2/{g}/{g + str(j) + str(w)}.wav',
                       format="wav")


threads = []
for g in genres:
    threads.append(myThread(g))

for t in threads:
    t.start()
