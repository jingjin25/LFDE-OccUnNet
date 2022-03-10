%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate test data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%% path
data_folder = '.\Dataset\LF_Dataset\Dataset_HCI';
savefolder = 'test_data_HCI_RGB';
if ~exist(savefolder,'dir')
    mkdir(savefolder);
end

data_list = {'backgammon','pyramids','dots','stripes'};
% data_list = {'boxes','cotton','dino','sideboard'};
% data_list = {'bedroom','bicycle','herbs','origami'};


an = 9;

count = 0;

%% read datasets

%%% read lfs
for i_lf = 1:length(data_list)
    lfname = data_list{i_lf};

    read_path = fullfile(data_folder,lfname);
    lf_rgb = read_hci(read_path,9,an);

    LF = permute(lf_rgb,[4,5,3,1,2]);

    savepath = fullfile(savefolder,[lfname,'.mat']);
    save(savepath, 'LF');
end


