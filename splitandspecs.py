# import os
#
# import numpy as np
# import scipy
# from pydub import AudioSegment
# import threading
#
# genres = 'blues classical country disco hiphop metal pop reggae rock'
# genres = genres.split()
#
# #For original wavs
# current_name = input("Current name: ")
# original_wavs_path = 'E:/original_wav_location/'+current_name
# split_path = 'E:/splitted_location/'+current_name
# spec_path = 'E:/spec_location/'+current_name
#
# class myThread(threading.Thread):
#     def __init__(self, genre):
#         threading.Thread.__init__(self)
#         self.genre = genre
#
#     def run(self):
#         print("Started " + self.genre)
#         split_files(self.genre)
#         print("Finished " + self.genre)
#
#
# def split_files(g):
#     j = 0
#     for filename in os.listdir(os.path.join(original_wavs_path, f"{g}")):
#         song = original_wavs_path+'/'+g+'/'+filename
#         #song = os.path.join(f'Z:/Egyetem/önlab2_msc/raw_audio/yt_dataset_2/{g}', f'{filename}')
#         j = j + 1
#         for w in range(0, 1000):
#             t1 = 3 * (w) * 1000
#             t2 = t1 + 6000
#             new_audio = AudioSegment.from_wav(song)
#             new = new_audio[t1:t2]
#             export_location = split_path+'/'+g+'/spec'+str(j)+str(w)+'.wav'
#             new.export(export_location,format="wav")
#
# threads = []
# for g in genres:
#     threads.append(myThread(g))
#
# for t in threads:
#     t.start()
#
# from scipy.io import wavfile
#
# for genre in genres:
#     path = split_path = 'E:/splitted_location/'+current_name+'/'+genre+'/'
#     files = os.listdir(path)
#     for z in range(0, len(files)):
#         [audioIn, Fs] = wavfile.read(files(z).folder+'/'+files(z).name)
#         sizeofAudio = sizeofAudio(1)
#         windowlength = 2500
#         window = scipy.io.signal.windows.hamming(windowlength)
#         df = Fs / windowlength
#         S = np.zeros(round(windowlength / 2) + 1, round(3 * Fs / windowlength))
#         k = 1
#         j = 1
#         stepsize = 400
#         overlap = windowlength - stepsize
#         while k < sizeofAudio - windowlength - 2:
#             if j > 1:
#                 y = audioIn[k - overlap:k + windowlength - overlap - 1]
#                 k = k - overlap;
#             else:
#                 y = audioIn[k:k + windowlength - 1]
#             spect = np.fft.fft(y*window)
#             S(:, j) = spect[1: round(windowlength / 2) + 1]
#             k = k + windowlength;
#             j = j + 1;
#
#         dBS = db(S) - max(max(db(S)));
#         dBS(dBS < -70) = -70;
#         X = 1:size(dBS, 2);
#         Y = Fs * (0:windowlength / 2) / (windowlength);
#         surf(X, Y, dBS, 'EdgeColor', 'none')
#         pbaspect([1 1 1])
#         % image(dBS, 'CDataMapping', 'scaled')
#         axis
#         xy;
#         axis
#         tight;
#         axis([1 max(X) 50 10000])
#         colormap("gray");
#         % colormap("default");
#         view(0, 90);
#         axis
#         off;
#         set(gca, 'YScale', 'log');
#         % export_fig('Z:\Egyetem\önlab_msc\doksi\v4_doksi.png', '-dpng', '-m1', '-transparent', '-r300');
#         name = strcat(
#             "Z:\Egyetem\önlab2_msc\spectograms\spectograms6s_overlap3s_newdataset2\train\",string(genre),'\spec_yt2_',num2str(z-2),'.png');
#         export_fig(name, '-dpng', '-m0.8', '-transparent', '-r300');
#         z = z + 1;
#         end;
#         end;
#         % figure;
#         % plot(fvect, db(spect(1: windowlength / 2 + 1)));
#
#
#
#
#
#
#
