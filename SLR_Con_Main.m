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
%  A_01=['mkdir ' 'data\output\P1'];  %��������    
%  system(A_01);                       %�����ļ���
%%
fid = fopen('result\recognized sentence.txt','wt');
for groupID = 1:5
    groupName = ['D:\iData\Outputs\ftdcgrs_whj_output\' teatDataPath '\test_' num2str(groupID) '\'];
    for sentenceID = 1 : length(sentence_names)
        fprintf('Processing data: G%d--S%d\n', groupID, sentenceID);
        data = importdata([groupName sentence_names{sentenceID} '.txt'], ' ', 1);
        [h, w] = size(data.data);  % h:֡��  w:ά��
        TestData = (data.data)';
            % CRF
%          for j=1:h
%                testLabels{1,1}(j) = 1;       % �˴���ʱlabelΪ1����Ҫ�ֶ���ע������group 2 �б�ע��Ϣ��
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
        %����ͨ��һ����������cov����
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
                    C = (1/(t_-t))*((Q{1,t_}-Q{1,t})-(1/(t_-t+1))...   %�����棬ÿ�μ���cov
                        *((P{1,t_}-P{1,t})*(P{1,t_}-P{1,t})'));
                end
                    % SVD Cov���õ�ǰ5ά
                [u,s,v] = svd(C);
                Para_ARMA_test{1}.C = u(:,1:5);
                testID = t;    % ��ʱ��t���棬ע��t����ʱ��
                test_label(1) = t;
                ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test,testID);
                VValKernel = [(1:1)',ValKernel'];
                [predict_label_P1, accuracy_P1, dec_values_P1] = ...
                    svmpredict(test_label, VValKernel, model_precomputed);
                result(t-start+1,t_) = predict_label_P1;
                
            end
        end
   
        
        %�߷ָ��ʶ�����

    end
end
fclose(fid);