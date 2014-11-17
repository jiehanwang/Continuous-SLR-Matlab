%����
%����š�����matlab��ʹ�õġ���1��ʼ��
%��ID���Ǿ��ӻ��ߵ��ʱ���ʹ�õģ���w0000��ʼ��
%����֮�������1�Ĺ�ϵ��
clear all;
clc;
%% Settings and Initials
addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));

% ��ȡģ�Ϳ�
load data\model_noSeg_209sen_242sign_forP0801;
%model_noSeg_209sen_242sign_forP0801
%model_2Seg_150sen_242sign_forP0801

% ��ȡ class_correlation��������������ϵͼ��
load data\class_correlation_model_242_noseg;   

% ��ȡ���Կ�
sentence_names = importdata('input\sentences_150_4.txt');
teatDataPath = 'dim334_CTskp_fullFrame_209sentences'; 

% ��ȡ�õ���ID���ϱ�ʾ�ľ���
sentences_meaning_number_Path = 'input\sentences_meaning_ID_real150_4.txt';  
sentences_meaning_number = ChineseDataread(sentences_meaning_number_Path);

% ���ļ���ȷ����ǰ��ά��
idx = strfind(teatDataPath,'_');
dimFinalIdx = idx(1,1)-1;
dim = str2double(teatDataPath(4:dimFinalIdx));

% ��ȡ������˼�Ͷ�Ӧ��ID��
ChinesePath = 'input\wordlist_370.txt';
chineseIDandMean = ChineseDataread(ChinesePath);

% ��ȡ���Դʻ�ID
% vocabulary = importdata('input\sign_150_forReal209.txt');
vocabulary = model_precomputed.Label;

classNum = 242;     % 
subSpaceSize = 5;   % �ӿռ��С  5
gap = 3;            % ��n֡����
thre = 0.8;        % score>thre ����Ϊ��Ч
draw = 1;           % 1:��ʾ��Ƶ�� 0������ʾ��Ƶ
windowSize = 30;    % �������ڵĴ�С
cutRegion = 40;
fidName = ['result\result' '_2SegModel_thre' num2str(thre) '_skip' ...
    num2str(gap) '_win' num2str(windowSize) '_Real209_242sign_BP2D_G0801_show.txt' ];
fid = fopen(fidName,'wt');
%%
for groupID =  1:1
    groupName = ['D:\iData\Outputs\ftdcgrs_whj_output\' teatDataPath '\test_' num2str(groupID) '\'];
    groundTruthFileFolderName = ['D:\iData\Outputs\ftdcgrs_whj_output\' teatDataPath...
        '\groundTruth_' num2str(groupID) '\'];
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
    
    % ��1��ʼ�ľ��ӱ�ţ� �����ӵ�ID���Ǵ�w0000��ʼ
    for sentenceID = 3:3%length(sentence_names)    
        sign_recognized_ID = [];
        sign_recognized_ID_Final = [];
        fprintf('Processing data: Group %d--Sentence %d\n', groupID, sentenceID);
        data = importdata([groupName sentence_names{sentenceID} '.txt'], ' ', 1);
        %groundTruth_ = importdata([groundTruthFileFolderName sentence_names{sentenceID} '.txt'], ' ', 1);
        [h, w] = size(data.data);  % h:֡��  w:ά��
        TestData = (data.data)';
        %groundTruth = groundTruth_.data;
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
        
        %����Ƶ�ļ�
        if draw == 1
            VideoPath = ['D:\iData\continousSentence\P08_01\S08_'...
                num2str(sentence_names{sentenceID}(2:5)) '_1_0_20130412.oni\color.avi'];
            videoObj = mmreader(VideoPath);  
        end
                   
        showText_result1 = 'none';
        showText_result2 = 'none';
        showText_true = 'Truth:';
        % ��ȷ����˼
        trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
        sign_groundTruth_ID = [];
        recognizeCount = 0;          % sign_recognized_ID   % ʶ������ĵ�Sign ID
        sign_count = 1;
        for sign_i = 1:trueSenLen
            sign_tempID = str2double(sentences_meaning_number{1,1+sentenceID}{1,sign_i});
            if ismember(sign_tempID, vocabulary)
                sign_groundTruth_ID(sign_count) = sign_tempID;
                showText_true = [showText_true chineseIDandMean{1,sign_groundTruth_ID(sign_count)+1}{1,2} '/'];
                sign_count = sign_count + 1;
            end
            
        end
        trueSenLen = sign_count-1;
        
        score_all = [];
        for k=cutRegion:gap:nframes-cutRegion
            showText_pace = ['Sentence ID: ' sentence_names{sentenceID}(2:5) ', '...
                   num2str(k) '/' num2str(nframes) ' frames, '];
            if draw == 1
                currentFrame = read(videoObj, k);%��ȡ��i֡
                imshow(currentFrame);
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
            ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test);
            VValKernel = [(1:1)',ValKernel'];
            [predict_label_P1, accuracy_P1, dec_values_P1] = ...
                svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'����ȥ����������Ϣ
            index_max = predict_label_P1 + 1;

            % ����SVM�ĸ��ʣ�����one-to-one����Ϣ.
            dec_values_P1(dec_values_P1<-1) = -1;
            dec_values_P1(dec_values_P1>1) = 1;
            score = dec_values_score(dec_values_P1, classNum); 
            [score_sort, index_sort] = sort(score,'descend');
            score_max = max(score);
            
            % Rank 1 ������ֵ����Ϊ��Ч
            if score_max > thre
                score_all = [score_all score'];

                showText_result1 = ['Sign: '...
                    chineseIDandMean{1,index_max}{1,2} ' /score' num2str(score_max)...
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
            
            clc;
            fprintf('%s \n%s \n%s \n%s \n', showText_pace, showText_true, showText_result1,showText_result2);
            
            if draw == 1
                %text(sum(xlim)/2-200,sum(ylim)/2-110,l_count,'horiz','center','color','r');
                text(sum(xlim)/2-200,sum(ylim)/2-150,showText_result1,'horiz','center','color','r');
%                 text(sum(xlim)/2-200,sum(ylim)/2-130,showText_result2,'horiz','center','color','r');
%                 text(sum(xlim)/2-200,sum(ylim)/2-110,showText_result_candidate,'horiz','center','color','r');
                drawnow;    %ʵʱ��������
            end
        end
        
        %-------------------------------------------------------------------
        % Spotting with BP_2D
        if size(score_all,2)<1
            sign_recognized_ID_Final= sign_recognized_ID;
        else
            sign_recognized_ID_Final = BP_2D(score_all, classNum, vocabulary, class_correlation);
        end
        
        %-------------------------------------------------------------------
        sign_recognized_ID_Final(sign_recognized_ID_Final==46)=75;
        sign_groundTruth_ID(sign_groundTruth_ID==46)=75;
        totalsigns = totalsigns + size(sign_groundTruth_ID,2);
        %-------------------------------------------------------------------
        % �˴���ʶ���groundTruth�������������Ƚϣ�������ɾ�ĵ���Ŀ��ͳ�ơ� Delete, Insert, Substitue
        [distance2, insert, delete, substitute, correctSign] = editDis(sign_groundTruth_ID, sign_recognized_ID_Final);
        distance = levenshtein(sign_groundTruth_ID, sign_recognized_ID_Final);
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



