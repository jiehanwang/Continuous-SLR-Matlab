% function [Para_ARMA_test,Para_ARMA_train,TTrainKernel, VValKernel,accuracy,test_label, training_label, model_precomputed] ...
%     = GRASP_370(testID, training_path_01,training_path_02,training_path_03,training_path_04,test_path)

test_path  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_369sign\test_52\';
training_path_01  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_369sign\test_51\';
training_path_02  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_369sign\test_50\';
training_path_03  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_369sign\test_53\';
training_path_04 = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_369sign\test_54\';
testID = 52;

addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));
%%
names = importdata('sign_370.txt');
classNum = 370;
n = 5;
trainNum = 4;
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
tic;
for i = 1 : 5%length(names)
    fprintf('Reading data: P%d------%d\n', testID, i);
    data = cell(1, 5);
    data{1} = importdata([training_path_01 names{i} '.txt'], ' ', 1);
    data{2} = importdata([training_path_02 names{i} '.txt'], ' ', 1);
    data{3} = importdata([training_path_03 names{i} '.txt'], ' ', 1);
    data{4} = importdata([training_path_04 names{i} '.txt'], ' ', 1);
    data{5} = importdata([test_path names{i} '.txt'], ' ', 1);
    
    
    for g=1:5   % g==5作测试用
        data_norm = (insertFrame(data{g}.data,n))';
        
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
            if g < 5 
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
toc;
save('model_HierarSeg_370sign_forP3.mat','model_precomputed', 'Para_ARMA_train', 'training_label');
[B,i,j] = unique(training_label);
class_correlation = TrainKernel(i,i);
class_correlation = class_correlation/max(max(class_correlation));
save('class_correlation_model_370', 'class_correlation');