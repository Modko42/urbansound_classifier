[audioBG,FsBg] = audioread('/Users/modko42/Desktop/template/downtown_traffic.wav');

[audioIN,Fs] = audioread('/Users/modko42/Desktop/template/classes/car_horn/19026-1-0-0.wav');

random_number = rand();

padding = audioBG(10*Fs:14*Fs-1);

padding(20000:20000+size(audioIN)-1) = audioIN;



song = audioplayer(padding,Fs);



play(song);