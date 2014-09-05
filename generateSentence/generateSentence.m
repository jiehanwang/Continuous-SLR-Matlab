clear all;
clc;
%%
path_01  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_369sign\test_51\';

% addpath(genpath('./RF_Class_C/.'));
names = importdata('sign_370.txt');
sentence_names = importdata('sentences_209.txt');

%读取用单词ID集合表示的句子
sentences_meaning_number_Path = 'sentences_meaning_number.txt';
sentences_meaning_number = ChineseDataread(sentences_meaning_number_Path);

dim = 334;
%%
for s = 1:length(sentence_names)
    fprintf('Sentence Index: %d \n', s);
    sentence = sentences_meaning_number{1+s};
    filename = ['output\' sentence_names{s} '.txt'];
    fid = fopen(filename,'wt');
    nframes = 0;
    for i=1:size(sentence,2)
        sign_index = str2double(sentence{i}) + 1;
        data = importdata([path_01 names{sign_index} '.txt'], ' ', 1);
        [frame,dim] = size(data.data);
        nframes = nframes + frame;
    end
    fprintf(fid,'%d %d\n',nframes, dim);
    for i=1:size(sentence,2)
        sign_index = str2double(sentence{i}) + 1;
        data = importdata([path_01 names{sign_index} '.txt'], ' ', 1);

        data_content = data.data;
        for row = 1:size(data_content,1)
            for col = 1:size(data_content,2)
                fprintf(fid,'%f ',data_content(row, col));
            end
            fprintf(fid, '\n');
        end

    end
    fclose(fid);
end
