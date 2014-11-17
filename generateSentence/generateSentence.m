clear all;
clc;
%%
% path_01  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim61_CTskp_allFrame_369sign\test_52\';
% dim334_CTskp_allFrame_369sign
path_01  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim61_CTskp_allFrame_1000sign_7group\test_19\';

sentence_names = importdata('sentences_100.txt');

%读取用单词ID集合表示的句子
sentences_meaning_number_Path = 'sentence_meaning_ID_random_100.txt';
sentences_meaning_number = ChineseDataread(sentences_meaning_number_Path);

 fileFolder=['mkdir ' 'output\groundTruth'];    
 system(fileFolder);  
 fileFolder=['mkdir ' 'output\test'];    
 system(fileFolder);
%%
for s = 1:length(sentence_names)
    clc;
    fprintf('Sentence Index: %d \n', s);
    sentence = sentences_meaning_number{1+s};
    filename = ['output\test\' sentence_names{s} '.txt'];
    fid = fopen(filename,'wt');
    filename_groundTruth = ['output\groundTruth\' sentence_names{s} '.txt'];
    fid_groundTruth = fopen(filename_groundTruth,'wt');
    nframes = 0;
    for i=1:size(sentence,2)
        sign_index = str2double(sentence{i}) + 1;
        if str2double(sentence{i})<10
            name_ID_w = ['w000' sentence{i}];
        elseif str2double(sentence{i})<100
            name_ID_w = ['w00' sentence{i}];
        elseif str2double(sentence{i})<1000
            name_ID_w = ['w0' sentence{i}];
        elseif str2double(sentence{i})<10000
            name_ID_w = ['w' sentence{i}];
        end
        data = importdata([path_01 name_ID_w '.txt'], ' ', 1);
        [frame,dim] = size(data.data);
        nframes = nframes + frame;
    end
    fprintf(fid,'%d %d\n',nframes, dim);
    fprintf(fid_groundTruth,'%d %d\n',nframes, 1);
    for i=1:size(sentence,2)
        sign_index = str2double(sentence{i}) + 1;
        if str2double(sentence{i})<10
            name_ID_w = ['w000' sentence{i}];
        elseif str2double(sentence{i})<100
            name_ID_w = ['w00' sentence{i}];
        elseif str2double(sentence{i})<1000
            name_ID_w = ['w0' sentence{i}];
        elseif str2double(sentence{i})<10000
            name_ID_w = ['w' sentence{i}];
        end
        data = importdata([path_01 name_ID_w '.txt'], ' ', 1);

        data_content = data.data;
        for row = 1:size(data_content,1)
            for col = 1:size(data_content,2)
                fprintf(fid,'%f ',data_content(row, col));    
            end
            fprintf(fid_groundTruth, '%d \n', str2double(sentence{i}));
            fprintf(fid, '\n');
        end

    end
    fclose(fid);
    fclose(fid_groundTruth);
end
