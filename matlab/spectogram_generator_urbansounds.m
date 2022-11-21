genres = ["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"];
%genres = ["car_horn"];


[audioBG,FsBg] = audioread('/Users/modko42/Desktop/template/downtown_traffic.wav');
import_location = "/Users/modko42/Desktop/template/classes/";
training_export_location = "/Users/modko42/Desktop/template/spectograms/";
version = "v5";


[bL,aL] = butter(2,4000/(22050/2));
[bH,aH] = butter(2,200/(22050/2),'high');

for x=1:length(genres)
    genre = genres(x);
    path = strcat(import_location,genre,'/');
    Files=dir(path);
    avg_duration = [];
    duration = 0;
    
    for z=1:length(Files)
        tic;
        [filepath,name,ext] = fileparts([Files(z).folder,'/',Files(z).name]);
        if ext == ".wav"
            [audioIn,Fs] = audioread([Files(z).folder,'/',Files(z).name]);
            
               
            sizeofAudio = size(audioIn);
            sizeofAudio = sizeofAudio(1);
            
            windowlength = 2500;
            window = hamming(windowlength);
            df = Fs / windowlength;
            S = zeros(round(windowlength/2)+1,round(4*Fs/windowlength));
            k = 1;
            j = 1;
            stepsize = 400;
            overlap = windowlength-stepsize;
            random_offset_in_steps = 0;
            random_start_position = 0;
            if sizeofAudio+0.2*Fs < 4*Fs
               random_offset_in_steps = floor(((4*Fs - sizeofAudio)) * rand() / windowlength);
               random_start_position = (100 + rand() * 2000)*Fs;
            end
            padded_audio = audioBG(1+random_start_position:random_start_position+4*Fs);
            loudness_difference = 2+rand()*3;%acousticLoudness(audioIn,Fs) / acousticLoudness(padded_audio,FsBg);
            if loudness_difference > 4
               loudness_difference = 4;
            end
            padded_audio = loudness_difference * padded_audio;
            padded_audio(1+random_offset_in_steps*stepsize:random_offset_in_steps*stepsize+sizeofAudio) = audioIn;
            audioIn = padded_audio;
            sizeofAudio = 4*Fs;
               
            
            avg_duration(end+1) = duration;
            disp(z+" spec "+random_offset_in_steps+" offset, bg noise x"+loudness_difference+" render time: "+duration+" avg: "+(sum(avg_duration) / length(avg_duration)))

            while k < sizeofAudio-windowlength-2
                    if j>1
                        y = audioIn(k-overlap:k+windowlength-overlap-1);
                        k = k - overlap;
                    else
                        y = audioIn(k:k+windowlength-1);
                    end
                  spect = fft(y.*window);
                  S(:,j) = spect(1:round(windowlength/2)+1);
                  k = k + windowlength;
                  j = j+1;
            end    
            
            dBS = db(S) - max(max(db(S)));
            dBS(dBS < -70) = -70;
            size(dBS);
            X = 1:size(dBS,2);
            Y = Fs*(0:windowlength/2)/(windowlength);
            surf(X,Y,dBS,'EdgeColor','none')
            pbaspect([1 1 1])
            %image(dBS,'CDataMapping','scaled')
            axis xy; 
            axis tight; 
            axis([1 max(X) 50 10000]) 
            colormap("gray"); 
            %colormap("default");
            view(0,90);
            axis off;
            set(gca, 'YScale', 'log');
            name = strcat(training_export_location,version,'/train/',string(genre),'/spec_',num2str(z-2),'.png');
            export_fig(name, '-dpng','-m0.7', '-transparent', '-r300');
            %customWriteImage(S,name);
            z = z + 1;
            duration = toc;
        end
    end
end


%customWriteImage(S,'/Users/modko42/Desktop/teszt.png');


function customWriteImage(S,filepath)
    normalized = normalize(db(S),'range',[0 1]);
    normalized = flip(imresize(normalized,(256/size(normalized,2)),'bilinear'));
    logscaled = zeros(256,256);
        
    custom_scale = logspace(0.7,3.097,256);

    for row=1:size(logscaled,1)
        for col=1:size(logscaled,2)
            logscaled(row,col) = normalized(round(custom_scale(row)),col);
        end
        %disp("logscaled("+row+","+col+") = normalized("+round(custom_scale(row))+","+col+")")
    end
    
    imwrite(flip(logscaled),filepath);
end



 



