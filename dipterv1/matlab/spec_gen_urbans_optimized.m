genres = ["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"];
%genres = ["mixed_audio"];

windows = true;
separator = '\';

if windows
    [audioBG,FsBg] = audioread("E:\urbandsounds8k\downtown_traffic.wav");
    import_location = "E:\urbandsounds8k\classes\";
    training_export_location = "E:\urbandsounds8k\spectograms\";
    separator = '\';
else
    [audioBG,FsBg] = audioread('/Users/modko42/Desktop/template/downtown_traffic.wav');
    import_location = "/Users/modko42/Desktop/template/classes/";
    training_export_location = "/Users/modko42/Desktop/template/spectograms/";
    separator = '/';
end

upper_freq_limit = 10000;
lower_freq_limit = 50;

version = "v26_3_sound_features_"+lower_freq_limit+"_"+upper_freq_limit+"noBGnoise";

generateFolders(training_export_location,version,genres,separator);

[bL,aL] = butter(2,4000/(22050/2));
[bH,aH] = butter(2,200/(22050/2),'high');

for x=1:length(genres)
    genre = genres(x);
    path = strcat(import_location,genre,separator);
    Files=dir(path);
    duration = 0;
    
    for z=1:length(Files)
        tic;
        [filepath,name,ext] = fileparts([Files(z).folder,separator,Files(z).name]);
        if ext == ".wav"
            [audioIn,Fs] = audioread([Files(z).folder,separator,Files(z).name]);
            
               
            sizeofAudio = size(audioIn);
            sizeofAudio = sizeofAudio(1);
            
            windowlength = 2500;
            window = hamming(windowlength);
            df = Fs / windowlength;
            S = zeros(round(windowlength/2)+1,round(4*Fs/windowlength));
            C = zeros(209,1);
            ZCR = zeros(209,1);
            RMS = zeros(209,1);
            k = 1;
            j = 1;
            stepsize = floor((4*Fs-2500)/256);
            overlap = windowlength-stepsize;
            random_offset_in_steps = 0;
            random_start_position = 0;
            if sizeofAudio+0.2*Fs < 4*Fs
               random_offset_in_steps = floor(((4*Fs - sizeofAudio)) * rand() / windowlength);
               random_start_position = round((100 + rand() * 2000)*Fs);
            end
            %padded_audio = audioBG(1+random_start_position:random_start_position+4*Fs);
            %loudness_difference = 1+rand()*2;%acousticLoudness(audioIn,Fs) / acousticLoudness(padded_audio,FsBg);
            %padded_audio = loudness_difference * padded_audio;
            padded_audio = zeros(4*Fs,1);
            padded_audio(1+random_offset_in_steps*stepsize:random_offset_in_steps*stepsize+sizeofAudio) = audioIn;
            %padded_audio(1:sizeofAudio) = audioIn;
            audioIn = padded_audio;
            sizeofAudio = 4*Fs;
     
            %disp(z+" spec "+random_offset_in_steps+" offset, bg noise x"+loudness_difference+" render time: "+duration)
            disp(z+" spec "+random_offset_in_steps+" offset"+" render time: "+duration)

            while k < sizeofAudio-windowlength-2
                    if j>1
                        y = audioIn(k-overlap:k+windowlength-overlap-1);
                        k = k - overlap;
                    else
                        y = audioIn(k:k+windowlength-1);
                    end
                  spect = fft(y.*window);
                  S(:,j) = spect(1:round(windowlength/2)+1);
                  C(j) = customSpectalCentroid(y);
                  ZCR(j) = zerocrossrate(y);
                  RMS(j) = rms(y);
                  k = k + windowlength;
                  j = j+1;
            end
            %SE = pentropy(audioIn,Fs);

            splitted_name = split(Files(z).name,'.');
            name = strcat(training_export_location,version,separator,'train',separator,string(genre),separator,splitted_name(1),'.png');
            customWriteImage(S,C,ZCR,SE,RMS,name,lower_freq_limit,upper_freq_limit,df);
            z = z + 1;
            duration = toc;
        end
    end
end

function customWriteImage(S,centroid,zerocrossingrate,spectral_entropy,root_mean_square,filepath,bottomFreq,topFreq,df)
    normalized = normalize(db(S),'range',[0 1]);
    normalized = imresize(normalized,[1251 256]);
    %inv_normalized = flip(imresize(normalized,[1251 256]));
    logscaled = zeros(256,256,2);
        
    centroid = imresize(centroid,256/length(centroid));
    norm_centroid = normalize(centroid,'range',[0 1]);

    zerocrossingrate = imresize(zerocrossingrate,256/length(zerocrossingrate));
    norm_zerocrossingrate = normalize(zerocrossingrate,'range',[0 1]);

    %spectral_entropy = imresize(spectral_entropy,256/length(spectral_entropy));
    %norm_sentropy = normalize(spectral_entropy,'range',[0 1]);

    root_mean_square = imresize(root_mean_square,256/length(root_mean_square));
    norm_rms = normalize(root_mean_square,'range',[0 1]);

    musical_features = zeros(256,256);

    custom_scale = logspace(log10(bottomFreq),log10(topFreq),256);
    log_constant = custom_scale(2) / custom_scale(1);

    for row=1:size(logscaled,1)
        start_bucket = round((1/sqrt(log_constant))*custom_scale(row)/df);
        end_bucket = round(sqrt(log_constant)*custom_scale(row)/df);
        for col=1:size(logscaled,2)   
            logscaled(row,col,1) = mean(normalized(start_bucket:end_bucket,col));
            %logscaled(row,col,2) = mean(inv_normalized(start_bucket:end_bucket,col));
            
           if row < 1*32+1
               musical_features(row,col) = norm_centroid(col);
            else 
               if row < 2*32+1
                    musical_features(row,col) = norm_zerocrossingrate(col);
                else 
                    if row < 3*32+1
                        %musical_features(row,col) = norm_sentropy(col);
                    else 
                        if row < 4*32+1
                            musical_features(row,col) = norm_rms(col);
                        end
                    end
               end 
            end
        end
    end

    rgb_img = zeros(256,256,3);
    rgb_img(:,:,1) = flip(logscaled(:,:,1));
    %rgb_img(:,:,2) = flip(logscaled(:,:,2));
    rgb_img(:,:,3) = musical_features;
    
    imwrite(rgb_img,filepath);
end


function generateFolders(path_,v_,classes,separator)
    mkdir(strcat(path_,v_))
    for c=1:length(classes)
        mkdir(strcat(path_,v_,separator,"train",separator,classes(c)))
        mkdir(strcat(path_,v_,separator,"test",separator,classes(c)))
    end
end

function spectral_centroid = customSpectalCentroid(signal)
    spectrum = abs(fft(signal));
    normalized_spectrum = spectrum / sum(spectrum);
    normalized_frequencies = linspace(0, 1, length(spectrum));
    spectral_centroid = sum(normalized_frequencies * normalized_spectrum);
end
 



