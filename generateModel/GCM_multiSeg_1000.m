test_path  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_zeng\test_31\';
training_path_01  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_zeng\test_30\';
training_path_02  = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_zeng\test_32\';
testID = 31;

addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));
%%
names = importdata('sign_1000_zeng.txt');
classNum = 1000;
n = 10;
trainNum = 2;
segmentNum = 3;       % 3 ��һ��sign�ֳɵĶ������Ժ������low rankȥ���

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
    data = cell(1, 3);
    data{1} = importdata([training_path_01 names{i} '.txt'], ' ', 1);
    data{2} = importdata([training_path_02 names{i} '.txt'], ' ', 1);
    data{3} = importdata([test_path names{i} '.txt'], ' ', 1);
    
    
    for g=1:3   % g==3��������
%         data_norm = (insertFrame(data{g}.data,n))';
        data_norm = (data{g}.data)';
        
        % ����ͨ��һ����������cov����
        [~, nframes] = size(data_norm);
        P = cell(1, nframes);
        Q = cell(1, nframes);
        for t=1 : nframes       % ���ｫȫ����cov��ʽ��ɲ����������ʵû�б�Ҫ
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
            if g < 3 
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

save('model_HierarSeg_1000sign_forP2.mat','model_precomputed', 'Para_ARMA_train', 'training_label');
[B,i,j] = unique(training_label);
class_correlation = TrainKernel(i,i);
class_correlation = class_correlation/max(max(class_correlation));
save('class_correlation_model_1000', 'class_correlation');