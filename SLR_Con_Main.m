%����
%����š�����matlab��ʹ�õġ���1��ʼ��
%��ID���Ǿ��ӻ��ߵ��ʱ���ʹ�õģ���w0000��ʼ��
%����֮�������1�Ĺ�ϵ��

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
segPath = 'input\segManually_P08_02.txt';
% dim334_CTskp_fullFrame_209sentences 
% dim10_Tskp_fullFrame_209sentences
% dim61_CTskp_fullFrame_209sentences

idx = strfind(teatDataPath,'_');
dimFinalIdx = idx(1,1)-1;
dim = str2num(teatDataPath(4:dimFinalIdx));

%��ȡ������˼�Ͷ�Ӧ��ID��
ChinesePath = 'input\wordlist_370.txt';
chineseIDandMean = ChineseDataread(ChinesePath);

%��ȡ�õ���ID���ϱ�ʾ�ľ���
sentences_meaning_number_Path = 'input\sentences_meaning_number.txt';
sentences_meaning_number = ChineseDataread(sentences_meaning_number_Path);

classNum = 370;
%%
fid = fopen('result\recognized sentence.txt','wt');
for groupID = 2:2
    groupName = ['D:\iData\Outputs\ftdcgrs_whj_output\' teatDataPath '\test_' num2str(groupID) '\'];
    
    % ��1��ʼ��209�����ӱ�ţ� �����ӵ�ID���Ǵ�w0000��ʼ
    for sentenceID = 2 : 2 % length(sentence_names)    
        fprintf('Processing data: Group %d--Sentence %d\n', groupID, sentenceID);
        data = importdata([groupName sentence_names{sentenceID} '.txt'], ' ', 1);
        [h, w] = size(data.data);  % h:֡��  w:ά��
        TestData = (data.data)';
        
        %%%%%%%%%%%%%%%% CRF
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
        
        %%%%%%%%%%%%%%%% Grassmann manifold
        %����ͨ��һ����������cov����
        for t=1:h
            P{1,t} = sum(TestData(:,1:t),2);
            Q{1,t} = TestData(:,1:t)*TestData(:,1:t)';
        end

        
        VideoPath = ['D:\iData\continousSentence\P08_02\S08_' num2str(sentence_names{sentenceID}(2:5)) '_1_0_20130412.oni\color.avi'];
        videoObj = mmreader(VideoPath);             %����Ƶ�ļ�
        nframes = get(videoObj, 'NumberOfFrames');  %��ȡ��Ƶ�ļ�֡����
        windowSize = 50;
        for k = 1 : nframes
            currentFrame = read(videoObj, k);%��ȡ��i֡
            imshow(currentFrame);
            xlim=get(gca,'xlim');
            ylim=get(gca,'ylim');
            
            % ��ʾ��ȷ����˼
            xShift_true = 150;  %������ͼ�����ĵ�ƫ��
            yShift_true = 200;
            trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
            showText_true = [sentence_names{sentenceID}(2:5) ' Groundtruth: '];
            for sign_i = 1:trueSenLen
                sign_choosen_ID = str2num(sentences_meaning_number{1,1+sentenceID}{1,sign_i});
                showText_true = [showText_true chineseIDandMean{1,sign_choosen_ID+1}{1,2} '/'];
            end
            text(sum(xlim)/2-xShift_true,sum(ylim)/2-yShift_true,showText_true,'horiz','center','color','r');
            
            
            xShift = 200;  %������ͼ�����ĵ�ƫ��
            yShift = 100;
            if k>windowSize/2 && k<nframes - windowSize/2
                t = k-windowSize/2;
                t_= k+windowSize/2;
                C = (1/(t_-t))*((Q{1,t_}-Q{1,t})-(1/(t_-t+1))...  
                        *((P{1,t_}-P{1,t})*(P{1,t_}-P{1,t})'));
                lamda = 0.001 * trace(C);
                C = C + lamda * eye(size(C, 1));   
                    % SVD Cov���õ�ǰ5ά
                [u,s,v] = svd(C);
                Para_ARMA_test{1}.C = u(:,1:5);
                testID = t;    % ��ʱ��t���棬ע��t����ʱ��
                test_label(1) = t;
                ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test,testID);
                VValKernel = [(1:1)',ValKernel'];
                [predict_label_P1, accuracy_P1, dec_values_P1] = ...
                    svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'����ȥ����������Ϣ
                result(k) = predict_label_P1;  %ע�⣬����Ǵ�0��369,�����ʾ����sign��ID�š�
                score = dec_values_score(dec_values_P1, classNum); 
                [score_max, index_max] = max(score);
               
                
                showText = ['Frame: ' num2str(k) '; Sign: '  chineseIDandMean{1,index_max}{1,2} ' / score: ' num2str(score_max)];
                text(sum(xlim)/2-xShift,sum(ylim)/2-yShift,showText,'horiz','center','color','r');
            end
            
            drawnow;    %ʵʱ��������
        end


%         segPosition = dataread(segPath, sentenceID+1);
%         segSignSize = floor((size(segPosition,2)-1)/2);
%         for segSign=1:segSignSize
%             signStart(segSign) = segPosition(2*segSign);
%             signEnd(segSign)= segPosition(1 + 2*segSign);
%         end
%         
%         for segSign=1:segSignSize
%             startFrame = signStart(segSign);
%             endFrame = signEnd(segSign);
%             t = str2num(startFrame{1,1});
%             t_= str2num(endFrame{1,1});
%             %�����棬ÿ�μ���cov����Ȼ������C{t,t_}����
%             C = (1/(t_-t))*((Q{1,t_}-Q{1,t})-(1/(t_-t+1))...  
%                         *((P{1,t_}-P{1,t})*(P{1,t_}-P{1,t})'));
%             % SVD Cov���õ�ǰ5ά
%             [u,s,v] = svd(C);
%             Para_ARMA_test{1}.C = u(:,1:5);
%             testID = t;    % ��ʱ��t���棬ע��t����ʱ��
%             test_label(1) = t;
%             ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test,testID);
%             VValKernel = [(1:1)',ValKernel'];
%             [predict_label_P1, accuracy_P1, dec_values_P1] = ...
%                 svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'����ȥ����������Ϣ
%             result(sentenceID, segSign) = predict_label_P1 + 1;
%         end
        
%         start = 1;
%         for t=start:h
%             fprintf('Current start frame: %d / %d \n',t,h);
%             for t_=t+1:h
%                 if t == 1
%                     C = (1/t_-1)*(Q{1,t_}-(1/t_)*(P{1,t_}*(P{1,t_})'));
%                 else
%                     C = (1/(t_-t))*((Q{1,t_}-Q{1,t})-(1/(t_-t+1))...   %�����棬ÿ�μ���cov����Ȼ������C{t,t_}����
%                         *((P{1,t_}-P{1,t})*(P{1,t_}-P{1,t})'));
%                 end
%                     % SVD Cov���õ�ǰ5ά
%                 [u,s,v] = svd(C);
%                 Para_ARMA_test{1}.C = u(:,1:5);
%                 testID = t;    % ��ʱ��t���棬ע��t����ʱ��
%                 test_label(1) = t;
%                 ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test,testID);
%                 VValKernel = [(1:1)',ValKernel'];
%                 [predict_label_P1, accuracy_P1, dec_values_P1] = ...
%                     svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'����ȥ����������Ϣ
%                 result(t-start+1,t_) = predict_label_P1;
%                 
%             end
%         end
   
        
        %�߷ָ��ʶ�����

    end
end
fclose(fid);


