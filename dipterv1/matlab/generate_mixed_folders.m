classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"];
parent_folder = "E:\temp_location\mixed_test_data_v2\";
already_created_combos = [];

all_classes = classes;

for first_class=1:10
    for second_class=first_class:10
        if first_class ~= second_class
            mkdir(parent_folder+classes(first_class)+'-'+classes(second_class));
            disp("["+int2str(first_class-1)+","+int2str(second_class-1)+"],");
        end
    end
end