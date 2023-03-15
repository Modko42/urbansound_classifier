classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"];
test_folder = "E:\urbandsounds8k\classes\";
export_folder = "E:\urbandsounds8k\classes\mixed_audio\";
for i=1:100
    first_class = randi([1,10],1);
    second_class = randi([1,10],1);
    
    first_path = strcat(test_folder,classes(first_class),'\');
    first_directory=dir(first_path);
    first_file = randi([3,length(first_directory)],1);
    [firstaudioIn,Fs] = audioread(strcat(first_path,first_directory(first_file).name));

    second_path = strcat(test_folder,classes(second_class),'\');
    second_directory = dir(second_path);
    second_file = randi([3,length(second_directory)],1);
    [secondaudioIn, ] = audioread(strcat(second_path,second_directory(second_file).name));

    sizeoffirst = size(firstaudioIn);
    sizeoffirst = sizeoffirst(1);

    sizeofsecond = size(secondaudioIn);
    sizeofsecond = sizeofsecond(1);

    firstaudioIn(4*Fs) = 0;
    secondaudioIn(4*Fs) = 0;

    mixed_audio = firstaudioIn + secondaudioIn;
    
    filename = strcat(export_folder,int2str(i),'_',classes(first_class),'_',classes(second_class),'.wav')
    audiowrite(filename,mixed_audio,Fs);
    
end


