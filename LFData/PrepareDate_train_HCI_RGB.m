%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate training data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;
%%
savepath = 'train_HCI_12LF_RGB.h5';
an = 9;
data_img = zeros(512,512,3,an,an,1,'uint8');
count = 0;

%% HCI

dataset = 'hci';
folder ='.\Dataset\LF_Dataset\Dataset_HCI';
listname = 'list/Train_HCI.txt';
f = fopen(listname);
if( f == -1 )
    error('%s does not exist!', listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
data_list = C{1};
fclose(f); 

for i_lf = 1:length(data_list)
    lfname = data_list{i_lf};

    read_path = fullfile(folder,lfname);
    lf_rgb = read_hci(read_path,9,an);
%     lf_ycbcr = rgb2ycbcr_5d(lf_rgb);

%     lf_y = lf_ycbcr(:,:,1,:,:);

    count = count +1;
    data_img(:,:,:,:,:,count) = lf_rgb;     
end
    

%% generate data
order = randperm(count);
data_img = permute(data_img(:, :, :, :, :, order),[2,1,3,5,4,6]); %[h,w,c,ah,aw,N] -> [w,h,c,aw,ah,N]  

%% writing to HDF5
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

h5create(savepath, '/img_HR', size(data_img), 'Datatype', 'uint8'); % width, height, channels, number 

h5write(savepath, '/img_HR', data_img);

h5disp(savepath);
