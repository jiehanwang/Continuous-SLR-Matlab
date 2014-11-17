clear all;
clc;
addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));
path = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_fullFrame_209sentences\test_2\';
testID = 2;
n=5;
% segmentNum = 3;

% 读取用单词ID集合表示的句子
sentences_meaning_number_Path = 'sentence_meaning_ID_209.txt';  
sentences_meaning_number = sentenceIDDataread(sentences_meaning_number_Path);
selectSen = importdata('sentences_209.txt');

% 读物单词分割信息
segment_info_path = 'segManually_P08_02.txt';
segment_info_temp = sentenceIDDataread(segment_info_path);

% 读取需要建模的单词ID
sign_ID(:,1) = importdata('sign_80_num.txt');
sign_ID(:,2) = 0; % 每个词只建模三次，多的忽略

% segment_info = cell(209,1);
sign_count = 1;
for seleID=1:size(selectSen,1)
    ce = str2double(selectSen{seleID,1}(2:5))+1;
    fprintf('%d / %d \n', ce, 209);
    segPoints_str = segment_info_temp{1,ce+1}(1:end-1);
    sizeSeg = size(segment_info_temp{1,ce+1},2)-3;
    segPoints=[];
    segPoints.sentenceID = str2double(segment_info_temp{1,ce+1}(1,1));
    sign_i = 1;
    piont_i = 1;
    goodsentence = 0;
    
    % 找到单词切分点，以及将每个单词分成若干层级的段.
    for i=1:sizeSeg
        if mod(i,2)==1
            iso_label = str2double(sentences_meaning_number{1,1+ce}{1,sign_i});
            [tf, loc] = ismember(iso_label,sign_ID(:,1));
            if tf && sign_ID(loc,2)<6
                sign_ID(loc,2) = sign_ID(loc,2)+1;
                iso_begin = str2double(segment_info_temp{1,ce+1}(1,i+1));
                iso_end = str2double(segment_info_temp{1,ce+1}(1,i+2));
                iso_nframes = iso_end - iso_begin;
                segmentNum = 1; % max(floor(iso_nframes/8),1);  % 防止少于8帧的情况发生
                segP = zeros(1, segmentNum+1);
                segP(1) = iso_begin;
                for seg = 2:segmentNum+1
                    segP(seg) = floor(iso_nframes*(seg-1)/segmentNum)+iso_begin;
                end
                for gap=1:segmentNum
                    for from = 1:(segmentNum+1-gap)
                        p1 = from;
                        p2 = from+gap;
                        seg_begin = segP(p1);
                        seg_end = segP(p2);
                        segPoints.seg(piont_i,1) = seg_begin;
                        segPoints.seg(piont_i,2) = seg_end;
                        segPoints.seg(piont_i,3) = iso_label;
                        piont_i = piont_i + 1;
                    end
                end
                goodsentence = 1;
            end
            sign_i = sign_i + 1;
        end
    end
    
    ce_ID = ce-1;
    if ce_ID<10
        name_ID_w = ['w000' num2str(ce_ID)];
    elseif ce_ID<100
        name_ID_w = ['w00' num2str(ce_ID)];
    elseif ce_ID<1000
        name_ID_w = ['w0' num2str(ce_ID)];
    elseif ce_ID<10000
        name_ID_w = ['w' num2str(ce_ID)];
    end
    data = importdata([path name_ID_w '.txt'], ' ', 1);
    [nframes,dim] = size(data.data);
    Data = (data.data)';
    
    % 首先通过一个函数建立cov快查表
    P = cell(1, nframes);
    Q = cell(1, nframes);
    for t=1 : nframes       % 这里将全部的cov算式组成部分算出，其实没有必要
        P{t} = sum(Data(:,1:t),2);
        Q{t} = Data(:,1:t)*Data(:,1:t)';
    end
    
 
    if goodsentence == 1
        for i = 1:size(segPoints.seg,1)
            Para_ARMA_train{sign_count}.C = grasp_region(segPoints.seg(i,1), segPoints.seg(i,2), P, Q, n);
            training_label(sign_count) = segPoints.seg(i,3);
            sign_count = sign_count+1;
        end
    end
    
end
%% Training
sign_count = sign_count-1;
TrainKernel = kernel(Para_ARMA_train,[],testID);
% ValKernel = kernel_ARMA(Para_ARMA_train,Para_ARMA_test,testID);

TTrainKernel = [(1:sign_count)',TrainKernel];
% VValKernel = [(1:length(names)*segmentNum)',ValKernel'];

model_precomputed = svmtrain(training_label', TTrainKernel, '-t 4');
% [predict_label_P1, accuracy_P1, dec_values_P1] = svmpredict(test_label, VValKernel, model_precomputed);

save('model_noSeg_209sen_86sign_forP0801.mat','model_precomputed', 'Para_ARMA_train', 'training_label');
[B,i,j] = unique(training_label);
class_correlation = TrainKernel(i,i);
class_correlation = class_correlation/max(max(class_correlation));
save('class_correlation_model', class_correlation);