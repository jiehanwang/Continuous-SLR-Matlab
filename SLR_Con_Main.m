clear all;
clc;
%%
addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));
addpath('CRF\bin');
addpath('CRF');

load CRF\result_dim10_97_R
load data\Model_ARMA_CTC_334to5_allFrame_370sign_P50
sign_names = importdata('input\signs_283.txt');   % signs_97  signs_283
sentence_names = importdata('input\sentences_209.txt');
teatDataPath = 'dim334_CTskp_fullFrame_209sentences'; 
% dim334_CTskp_fullFrame_209sentences 
% dim10_Tskp_fullFrame_209sentences
% dim61_CTskp_fullFrame_209sentences

idx = strfind(teatDataPath,'_');
dimFinalIdx = idx(1,1)-1;
dim = str2num(teatDataPath(4:dimFinalIdx));
%%
%  A_01=['mkdir ' 'data\output\P1'];  %创建命令    
%  system(A_01);                       %创建文件夹
%%
fid = fopen('result\recognized sentence.txt','wt');
for groupID = 1:5
    groupName = ['D:\iData\Outputs\ftdcgrs_whj_output\' teatDataPath '\test_' num2str(groupID) '\'];
    for sentenceID = 1 : length(sentence_names)
        fprintf('Processing data: G%d--S%d\n', groupID, sentenceID);
        data = importdata([groupName sentence_names{sentenceID} '.txt'], ' ', 1);
        [h, w] = size(data.data);  % h:帧数  w:维数
        TestData = (data.data)';
            % CRF
%          for j=1:h
%                testLabels{1,1}(j) = 1;       % 此处暂时label为1，需要手动标注，其中group 2 有标注信息。
%             for k=1:w
%                testSeqs{1,1}(k,j) = data.data(j,k);
%             end
%          end
%         [R{1}.ll R{1}.labels] = test(R{1}.model, testSeqs, testLabels);
%         fprintf(fid, 'G%d--S%d: \t', groupID, sentenceID);
%         for i=1:h
%             [v, po] = max(R{1,1}.ll{1,1}(:,i));
%             className = sign_names{po,1};      % str2num(names{i}(2:5))
%             fprintf(fid,'%s \t', className); 
%         end
%         fprintf(fid, '\n');
        
             % Grassmann manifold
        %首先通过一个函数建立cov快查表
        for t=1:h
            P{1,t} = sum(TestData(:,1:t),2);
            Q{1,t} = TestData(:,1:t)*TestData(:,1:t)';
        end
        
        start = 1;
        for t=start:h
            fprintf('Current %d--%d \n',t,h);
            for t_=t+1:h
                if t == 1
%                     C{1,t_} = (1/t_-1)*(Q{1,t_}-(1/t_)*(P{1,t_}*(P{1,t_})'));
                    C = (1/t_-1)*(Q{1,t_}-(1/t_)*(P{1,t_}*(P{1,t_})'));
                else
%                     C{t,t_} = (1/(t_-t))*((Q{1,t_}-Q{1,t})-(1/(t_-t+1))...
%                         *((P{1,t_}-P{1,t})*(P{1,t_}-P{1,t})'));
                    C = (1/(t_-t))*((Q{1,t_}-Q{1,t})-(1/(t_-t+1))...   %不保存，每次计算cov
                        *((P{1,t_}-P{1,t})*(P{1,t_}-P{1,t})'));
                end
                    % SVD Cov，得到前5维
                [u,s,v] = svd(C);
                Para_ARMA_test{1}.C = u(:,1:5);
                testID = t;    % 暂时用t代替，注意t代表时间
                test_label(1) = t;
                ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test,testID);
                VValKernel = [(1:1)',ValKernel'];
                [predict_label_P1, accuracy_P1, dec_values_P1] = ...
                    svmpredict(test_label, VValKernel, model_precomputed);
                result(t-start+1,t_) = predict_label_P1;
                
            end
        end
   
        
        %边分割边识别策略

    end
end
fclose(fid);