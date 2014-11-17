clear all;
clc;
test_path  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_7group\test_19\';
training_path_01  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_7group\test_10\';
training_path_02  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_7group\test_11\';
training_path_03  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_7group\test_14\';
training_path_04  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_7group\test_15\';
training_path_05  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_7group\test_17\';
training_path_06  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_7group\test_21\';
testID = 19;

addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));
%%
names = importdata('sign_100_7g.txt');
classNum = 1000;
n = 10;
trainNum = 6;
segmentNum = 3;       %将一个sign分成的段数。以后可以用low rank去求解

totalSeg = 0;
for s=1:segmentNum
    totalSeg = totalSeg + s;
end
        
        
training_label = zeros(trainNum*length(names)*totalSeg,1);
test_label = zeros(length(names)*totalSeg,1);

Para_ARMA_train = cell(1,totalSeg*trainNum*length(names));
Para_ARMA_test = cell(1, totalSeg*length(names));
%%
for i = 1 : length(names)
    fprintf('Reading data: P%d------%d\n', testID, i);
    data = cell(1, 7);
    data{1} = importdata([training_path_01 names{i} '.txt'], ' ', 1);
    data{2} = importdata([training_path_02 names{i} '.txt'], ' ', 1);
    data{3} = importdata([training_path_03 names{i} '.txt'], ' ', 1);
    data{4} = importdata([training_path_04 names{i} '.txt'], ' ', 1);
    data{5} = importdata([training_path_05 names{i} '.txt'], ' ', 1);
    data{6} = importdata([training_path_06 names{i} '.txt'], ' ', 1);
    data{7} = importdata([test_path names{i} '.txt'], ' ', 1);
    
    
    for g=1:(trainNum+1)   % g==7作测试用
%         data_norm = (insertFrame(data{g}.data,n))';
        data_norm = (data{g}.data)';
        
        % 首先通过一个函数建立cov快查表
        [~, nframes] = size(data_norm);
        P = cell(1, nframes);
        Q = cell(1, nframes);
        for t=1 : nframes       % 这里将全部的cov算式组成部分算出，其实没有必要
            P{t} = sum(data_norm(:,1:t),2);
            Q{t} = data_norm(:,1:t)*data_norm(:,1:t)';
        end
        
        segP = zeros(1, segmentNum+1);
        segP(1) = 1;
        for seg = 2:segmentNum+1
            segP(seg) = floor(nframes*(seg-1)/segmentNum);
        end
        
        
        
        segS(1,1) = segP(1);
        segS(1,2) = segP(2);
        segS(2,1) = segP(1);
        segS(2,2) = segP(3);
        segS(3,1) = segP(1);
        segS(3,2) = segP(4);
        segS(4,1) = segP(2);
        segS(4,2) = segP(3);
        segS(5,1) = segP(2);
        segS(5,2) = segP(4);
        segS(6,1) = segP(3);
        segS(6,2) = segP(4);
        
        
        
        for seg = 1:totalSeg
            beginP = segS(seg,1);  % 1
            endP = segS(seg,2);
            if g < (trainNum+1)
                Para_ARMA_train{totalSeg*(trainNum*(i-1) + g - 1) + seg}.C = GCM_region(beginP, endP, P, Q, n);
                training_label(totalSeg*(trainNum*(i-1) + g - 1) + seg) = str2double(names{i}(2:5));
            else
                Para_ARMA_test{totalSeg*(i-1) + seg}.C = GCM_region(beginP, endP, P, Q, n);
                test_label(totalSeg*(i-1) + seg) = str2double(names{i}(2:5));
            end
        end
    end
    
end
%% Training
TrainKernel = kernel(Para_ARMA_train,[],testID);
% ValKernel = kernel_ARMA(Para_ARMA_train,Para_ARMA_test,testID);

TTrainKernel = [(1:length(names)*trainNum*totalSeg)',TrainKernel];
% VValKernel = [(1:length(names)*segmentNum)',ValKernel'];

model_precomputed = svmtrain(training_label, TTrainKernel, '-t 4');
% [predict_label_P1, accuracy_P1, dec_values_P1] = svmpredict(test_label, VValKernel, model_precomputed);

save('model_HierarSeg_100sign7g_forP19.mat','model_precomputed', 'Para_ARMA_train', 'training_label');
[B,i,j] = unique(training_label);
class_correlation = TrainKernel(i,i);
class_correlation = class_correlation/max(max(class_correlation));
save('class_correlation_model_276_noseg', 'class_correlation');