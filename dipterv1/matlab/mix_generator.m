classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"];
source_location = "E:\urbandsounds8k\classes_for_mixing\test\";
export_folder = "E:\temp_location\mixed_test\";
for first_class=1:length(classes)
    for second_class=first_class:length(classes)
        if first_class ~= second_class
            first_path = strcat(source_location,classes(first_class),'\');
            first_directory=dir(first_path);

            second_path = strcat(source_location,classes(second_class),'\');
            second_directory = dir(second_path);
            for counter=1:100
                first_file = randi([3,length(first_directory)],1);
                [filepath1,name1,ext1] = fileparts(strcat(first_path,first_directory(first_file).name));
                while ext1 ~= ".wav"
                    first_file = randi([3,length(first_directory)],1);
                    [filepath1,name1,ext1] = fileparts(strcat(first_path,first_directory(first_file).name));
                end
                
                [firstaudioIn,Fs] = audioread(strcat(first_path,first_directory(first_file).name));
            
                
                second_file = randi([3,length(second_directory)],1);
                [filepath2,name2,ext2] = fileparts(strcat(second_path,second_directory(second_file).name));
                while ext2 ~= ".wav"
                    second_file = randi([3,length(second_directory)],1);
                    [filepath2,name2,ext2] = fileparts(strcat(second_path,second_directory(second_file).name));
                end
                [secondaudioIn, ] = audioread(strcat(second_path,second_directory(second_file).name));
            
                sizeoffirst = size(firstaudioIn);
                sizeoffirst = sizeoffirst(1);
            
                sizeofsecond = size(secondaudioIn);
                sizeofsecond = sizeofsecond(1);
            
                firstaudioIn(4*Fs) = 0;
                secondaudioIn(4*Fs) = 0;

                firstaudioIn = firstaudioIn(1:88200);
                secondaudioIn = secondaudioIn(1:88200);


                mixed_audio = firstaudioIn + secondaudioIn;
                
                filename = strcat(export_folder,classes(first_class),'-',classes(second_class),'\',int2str(counter),'_',classes(first_class),'-',classes(second_class),'.wav')
                audiowrite(filename,mixed_audio,Fs);
            end
        end
    end
end


