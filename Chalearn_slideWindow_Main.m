%����
%����š�����matlab��ʹ�õġ���1��ʼ��
%��ID���Ǿ��ӻ��ߵ��ʱ���ʹ�õģ���w0000��ʼ��
%����֮�������1�Ĺ�ϵ��
clear all;
clc;
%% Settings and Initials
addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));

% ��ȡ class_correlation��������������ϵͼ��
load data\class_correlation_model_Chalearn;   

% ��ȡģ�Ϳ�
load data\model_chalearn
 
% ��ȡ���Կ�
sentence_names = importdata('input\sentences_275_Chalearn.txt');
teatDataPath = 'dim334_Chalearn_sentences'; 


% ��ȡ�õ���ID���ϱ�ʾ�ľ���
sentences_meaning_number_Path = 'input\sentences_meaning_number_Chalearn.txt';  
sentences_meaning_number = ChineseDataread(sentences_meaning_number_Path);

% ���ļ���ȷ����ǰ��ά��
idx = strfind(teatDataPath,'_');
dimFinalIdx = idx(1,1)-1;
dim = str2double(teatDataPath(4:dimFinalIdx));

% ��ȡ������˼�Ͷ�Ӧ��ID��
ChinesePath = 'input\wordlist_20.txt';
chineseIDandMean = ChineseDataread(ChinesePath);

% ��ȡ���Դʻ�ID
vocabulary = importdata('input\sign_20.txt');

classNum = 20;     % 370
subSpaceSize = 10;   % �ӿռ��С
gap = 1;            % ��n֡����
thre = 0.73;        % score>thre ����Ϊ��Ч
draw = 0;           % 1:��ʾ��Ƶ�� 0������ʾ��Ƶ
windowSize = 30;    % �������ڵĴ�С
%%
fid = fopen('result\Chalearn_result.txt','wt');
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
        [h, w] = size(data.data);  % h:֡��  w:ά��
        TestData = (data.data)';
        
%         groundTruth = groundTruth_.data;
        nframes = size(TestData, 2);
        correctFrame = 0;
        currentLabel = -1;    % ��Ϊ����ÿ֡����label���������������ÿ֡��label��
        
        %����ͨ��һ����������cov����
        P = cell(1,h);
        Q = cell(1,h);
        for t=1:h
            P{1,t} = sum(TestData(:,1:t),2);
            Q{1,t} = TestData(:,1:t)*TestData(:,1:t)';
        end

        showText_result1 = 'none';
        showText_result2 = 'none';
        showText_true = 'Truth:';
        % ��ȷ����˼
        trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
        totalsigns = totalsigns + trueSenLen;
        sign_groundTruth_ID = zeros(1, trueSenLen) - 1;      % ��ȷ��Sign ID
        recognizeCount = 0;          % sign_recognized_ID   % ʶ������ĵ�Sign ID
        for sign_i = 1:trueSenLen
            sign_groundTruth_ID(sign_i) = str2double(sentences_meaning_number{1,1+sentenceID}{1,sign_i});
            showText_true = [showText_true chineseIDandMean{1,sign_groundTruth_ID(sign_i)}{1,2} '/'];
        end
            
        TopNindex_ID = zeros(5,300);
        TopNscore_ID = zeros(5,300);
        TopNcount = 1;
        score_all = [];
        for k=1:gap:nframes
            showText_pace = ['Sentence ID: ' sentence_names{sentenceID}(2:5) ', '...
                   num2str(k) '/' num2str(nframes) ' frames, '];
            if draw == 1
                tempshow = zeros(480,640);
                imshow(tempshow);
                xlim=get(gca,'xlim');
                ylim=get(gca,'ylim');
                % ��ʾ��ȷ����˼
                text(sum(xlim)/2-0,sum(ylim)/2-210,showText_true,'horiz','center','color','r');
            end
            
            % ���ټ���Cov�����ӿռ䣬��GCM��
            t = max(k - windowSize/2, 1);
            t_= min(k + windowSize/2, nframes);
            Para_ARMA_test{1}.C = grasp_region(t, t_, P, Q, subSpaceSize);

            test_label(1) = t;
            ValKernel = kernel_Chalearn(Para_ARMA_train,Para_ARMA_test);
            VValKernel = [(1:1)',ValKernel'];
            [predict_label_P1, accuracy_P1, dec_values_P1] = ...
                svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'����ȥ����������Ϣ
%             index_max = predict_label_P1;

            % ����SVM�ĸ��ʣ�����one-to-one����Ϣ.
            score = dec_values_score(dec_values_P1, classNum); 
            [score_sort, index_sort] = sort(score,'descend');
            
            % Rank 1 ������ֵ����Ϊ��Ч
            if score_sort(1) > thre
%                 for topN = 1:5
%                     TopNindex_ID(topN,TopNcount) = index_sort(topN)-1;
%                     TopNscore_ID(topN,TopNcount) = score_sort(topN);
%                 end
%                 TopNcount = TopNcount + 1;
                score_all = [score_all score'];

                showText_result1 = ['Sign: '...
                    chineseIDandMean{1,predict_label_P1}{1,2} ' /score' num2str(score_sort(1))...
                    ' /groundTruth: ' ];
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
            
%             % ��ȷ��֡��ͳ��
%             if currentLabel == groundTruth(k)
%                 correctFrame = correctFrame + 1;
%             end
            

            clc;
            fprintf('%s \n%s \n%s \n%s \n', showText_pace, showText_true, showText_result1,showText_result2);
            
        end
        
        %-------------------------------------------------------------------
        % �Ծ��ӵ�����parsing.
        
        % ɾȥֻ����һ�εĴʡ�
        % sign_recognized_ID_Final = sign_recognized_ID(find(labelCount~=1));
        
        % �����﷨
        %  sign_recognized_ID_Final = addGrammar(sign_recognized_ID);
        
        % Spotting with viterbi_like2
%         sign_recognized_ID_Final = viterbi_like2(score_all, classNum, 30);
        
        % Spotting with BP_2D
%         sign_recognized_ID_Final = BP_2D(score_all, classNum, vocabulary, class_correlation_Chalearn);
        
        % Spotting with noGrammar
        % sign_recognized_ID_Final = noGrammar(score_all);
         
        % noGrammar
%         sign_recognized_ID_Final = sign_recognized_ID;
        %-------------------------------------------------------------------
        
        % �˴���ʶ���groundTruth�������������Ƚϣ�������ɾ�ĵ���Ŀ��ͳ�ơ� Delete, Insert, Substitue
        [distance, insert, delete, substitute, correctSign] = editDis(sign_groundTruth_ID, sign_recognized_ID);
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



