%����
%����š�����matlab��ʹ�õġ���1��ʼ��
%��ID���Ǿ��ӻ��ߵ��ʱ���ʹ�õģ���w0000��ʼ��
%����֮�������1�Ĺ�ϵ��

clear all;
clc;
%%
addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));
addpath(genpath('D:\iCode\HMM_Matlab\HMMall'));

% ��ȡ class_correlation��������������ϵͼ��
load data\class_correlation_model_242_noseg;   

% ��ȡģ�Ϳ�
load data\HMM_model_s3_m3_dim61_242sign_forP801

% ��ȡ���Կ�
sentence_names = importdata('input\sentences_150_4.txt');
teatDataPath = 'dim61_CTskp_fullFrame_209sentences'; 

% ���ļ���ȷ����ǰ��ά��
idx = strfind(teatDataPath,'_');
dimFinalIdx = idx(1,1)-1;
dim = str2double(teatDataPath(4:dimFinalIdx));

%��ȡ������˼�Ͷ�Ӧ��ID��
ChinesePath = 'input\wordlist_370.txt';
chineseIDandMean = ChineseDataread(ChinesePath);

%��ȡ�õ���ID���ϱ�ʾ�ľ���
sentences_meaning_number_Path = 'input\sentences_meaning_ID_real150_4.txt';
sentences_meaning_number = ChineseDataread(sentences_meaning_number_Path);

% ��ȡ������˼�Ͷ�Ӧ��ID��
ChinesePath = 'input\wordlist_370.txt';
chineseIDandMean = ChineseDataread(ChinesePath);

% ��ȡ���Դʻ�ID
vocabulary = importdata('input\sign_242.txt');
for i=1:242
    vocabulary_new(i) = str2double(vocabulary{i,1}(2:5));
end

classNum = 242;
sample = 3;    %��n֡����
draw = 0;    %1:��ʾ��Ƶ�� 0������ʾ��Ƶ
thre = 1000;
windowSize = 40;   %�������ڵĴ�С
cutRegion = 40;
fidName = ['result\HMM_result' '_NoSegModel_thre' num2str(thre) '_skip' ...
    num2str(sample) '_win' num2str(windowSize) '_real150_242sign_BP2D_G0801_v2.txt' ];
fid = fopen(fidName,'wt');
%%
for groupID =  1:1
    groupName = ['D:\iData\Outputs\ftdcgrs_whj_output\' teatDataPath '\test_' num2str(groupID) '\'];
%     groundTruthFileFolderName = ['D:\iData\Outputs\ftdcgrs_whj_output\' teatDataPath...
%         '\groundTruth_' num2str(groupID) '\'];
    fprintf(fid, 'The test group: G_%s \n', num2str(groupID));
    fprintf(fid, '%s:/%s/%s/%s/%s/%s/%s/%s/%s/%s/%s\n',...
        'sentenceID', 'correctFrame', 'totalFrame',...
        'rate_frame','correctSign', 'groundtruth', 'rate_sign', 'distance',...
        'insert', 'delete', 'substitute');
    totalFrames = 0;
    totalCorrectFrame = 0;
    totalsigns = 0;
    totalCorrectSign = 0;
    totalDistance = 0;
    totalInsert = 0;
    totalDelete = 0;
    totalSubstitute = 0;
    
    
    % ��1��ʼ��209�����ӱ�ţ� �����ӵ�ID���Ǵ�w0000��ʼ
    for sentenceID = 1:length(sentence_names)    
        sign_recognized_ID = [];
        sign_recognized_ID_Final = [];
        fprintf('Processing data: Group %d--Sentence %d\n', groupID, sentenceID);
        data = importdata([groupName sentence_names{sentenceID} '.txt'], ' ', 1);
%         groundTruth_ = importdata([groundTruthFileFolderName sentence_names{sentenceID} '.txt'], ' ', 1);
        [h, w] = size(data.data);  % h:֡��  w:ά��
        TestData = (data.data)';
%         groundTruth = groundTruth_.data;
        nframes = size(TestData, 2);
        correctFrame = 0;
        currentLabel = -1;    % ��Ϊ����ÿ֡����label���������������ÿ֡��label��
        
        
        showText_result1 = 'none';
        showText_result2 = 'none';
        showText_true = 'Truth:';
        % ��ȷ����˼
        trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
%         totalsigns = totalsigns + trueSenLen;
%         sign_groundTruth_ID = zeros(1, trueSenLen) - 1;      % ��ȷ��Sign ID
        recognizeCount = 0;          % sign_recognized_ID   % ʶ������ĵ�Sign ID
        sign_count = 1;
        for sign_i = 1:trueSenLen
            sign_tempID = str2double(sentences_meaning_number{1,1+sentenceID}{1,sign_i});
            if ismember(sign_tempID, vocabulary_new)
                sign_groundTruth_ID(sign_count) = sign_tempID;
                showText_true = [showText_true chineseIDandMean{1,sign_groundTruth_ID(sign_count)+1}{1,2} '/'];
                sign_count = sign_count + 1;
            end
            
        end
        trueSenLen = sign_count-1;
        
        score_all = [];
        
        for k=cutRegion:nframes-cutRegion
            showText_pace = ['Sentence ID: ' sentence_names{sentenceID}(2:5) ', '...
                   num2str(k) '/' num2str(nframes) ' frames, '];
            if k>windowSize/2 && k<nframes - windowSize/2 && mod(k,sample)==0
                t = k - windowSize/2;
                t_= k + windowSize/2;
                
                data_norm = TestData(:,t:t_);
     
                loglik = zeros(1,classNum);
                for d=1:classNum
                    loglik(d) = mhmm_logprob(data_norm, prior{d}, transmat{d}, mu{d}, Sigma{d}, mixmat{d});
                end
                
                [score_sort, index_sort] = sort(loglik,'descend');
%                 index_max = index_sort(1);
%                 predict_label_P1 = index_max-1;
                 predict_label_P1 = str2double(vocabulary{index_sort(1,1),1}(2:5));
                index_max = predict_label_P1+1;
                
                if score_sort(1)>thre
                    score_all = [score_all loglik'];

                    showText_result1 = ['Sign: '...
                        chineseIDandMean{1,index_max}{1,2} ' /score' num2str(score_sort(1))];
                    showText_result2 = ['Candidates: ' chineseIDandMean{1,index_sort(2)}{1,2} '/'...
                                chineseIDandMean{1,index_sort(3)}{1,2} '/'...
                                chineseIDandMean{1,index_sort(4)}{1,2} '/'...
                                chineseIDandMean{1,index_sort(5)}{1,2} ];
                            
                    currentLabel = predict_label_P1;
                    % ���label���ظ��Ļ��ͼ�¼������ȡ����¼��
                     if recognizeCount == 0
                         recognizeCount = recognizeCount + 1;
                         sign_recognized_ID(recognizeCount) = predict_label_P1;
                         labelCount(recognizeCount) = 1;    % ��¼��label���ֵĴ���
                     elseif sign_recognized_ID(recognizeCount) ~= predict_label_P1 
                         recognizeCount = recognizeCount + 1;
                         sign_recognized_ID(recognizeCount) = predict_label_P1;
                         labelCount(recognizeCount) = 1;
                     else
                         labelCount(recognizeCount) = labelCount(recognizeCount) + 1;
                     end
                end
                
            end
            
            
            % ��ȷ��֡��ͳ��
%             if currentLabel == groundTruth(k)
%                 correctFrame = correctFrame + 1;
%             end

            clc;
            fprintf('%s \n%s \n%s \n%s \n', showText_pace, showText_true, showText_result1,showText_result2);
            
        end
        
        
        sign_recognized_ID_Final = BP_2D_HMM(score_all, classNum, vocabulary, class_correlation);
        %-------------------------------------------------------------------
        sign_recognized_ID_Final(sign_recognized_ID_Final==46)=75;
        sign_groundTruth_ID(sign_groundTruth_ID==46)=75;
        totalsigns = totalsigns + trueSenLen;
        %-------------------------------------------------------
        
         % �˴���ʶ���groundTruth�������������Ƚϣ�������ɾ�ĵ���Ŀ��ͳ�ơ� Delete, Insert, Substitue
        [distance, insert, delete, substitute, correctSign] = editDis(sign_groundTruth_ID, sign_recognized_ID_Final); % sign_recognized_ID
        totalCorrectSign = totalCorrectSign + correctSign;
        totalDistance = totalDistance + distance;
        totalInsert = totalInsert+insert;
        totalDelete = totalDelete+delete;
        totalSubstitute = totalSubstitute+substitute;
        
        % ������
        fprintf(fid, 'S%s:\t%d\t%d\t%f\t%d\t%d\t%f\t%d\t%d\t%d\t%d\n', sentence_names{sentenceID}(2:5), correctFrame, nframes-windowSize...
            , correctFrame/(nframes-windowSize),correctSign, trueSenLen, correctSign/trueSenLen, distance, insert, delete, substitute);
        totalFrames = totalFrames + nframes-windowSize;
        totalCorrectFrame = totalCorrectFrame + correctFrame;
    end
    % ������ͳ�ƽ��
    fprintf(fid, 'Ave.:\t%d\t%d\t%f\t%d\t%d\t%f\t%d\t%d\t%d\t%d \n', totalCorrectFrame,...
        totalFrames, totalCorrectFrame/totalFrames, totalCorrectSign, totalsigns,...
        totalCorrectSign/totalsigns, totalDistance,totalInsert,totalDelete,totalSubstitute);
end
fclose(fid);



