%声明
%“编号”是在matlab中使用的。从1开始。
%“ID”是句子或者单词本身使用的，从w0000开始。
%两者之间有相差1的关系。
clear all;
clc;
%% Settings and Initials
addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));

% 读取 class_correlation变量。即，类间关系图。
load data\class_correlation_model_Chalearn;   

% 读取模型库
load data\model_chalearn
 
% 读取测试库
sentence_names = importdata('input\sentences_275_Chalearn.txt');
teatDataPath = 'dim334_Chalearn_sentences'; 


% 读取用单词ID集合表示的句子
sentences_meaning_number_Path = 'input\sentences_meaning_number_Chalearn.txt';  
sentences_meaning_number = ChineseDataread(sentences_meaning_number_Path);

% 从文件名确定当前的维数
idx = strfind(teatDataPath,'_');
dimFinalIdx = idx(1,1)-1;
dim = str2double(teatDataPath(4:dimFinalIdx));

% 读取中文意思和对应的ID号
ChinesePath = 'input\wordlist_20.txt';
chineseIDandMean = ChineseDataread(ChinesePath);

% 读取测试词汇ID
vocabulary = importdata('input\sign_20.txt');

classNum = 20;     % 370
subSpaceSize = 10;   % 子空间大小
gap = 1;            % 隔n帧采样
thre = 0.73;        % score>thre 的视为有效
draw = 0;           % 1:显示视频。 0：不显示视频
windowSize = 30;    % 滑动窗口的大小
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
    
    % 从1开始的209个句子编号， 而句子的ID都是从w0000开始
    for sentenceID = 1:length(sentence_names)    
        sign_recognized_ID = [];
        sign_recognized_ID_Final = [];
        fprintf('Processing data: Group %d--Sentence %d\n', groupID, sentenceID);
        data = importdata([groupName sentence_names{sentenceID} '.txt'], ' ', 1);
        [h, w] = size(data.data);  % h:帧数  w:维数
        TestData = (data.data)';
        
%         groundTruth = groundTruth_.data;
        nframes = size(TestData, 2);
        correctFrame = 0;
        currentLabel = -1;    % 因为不是每帧都有label，用这个变量分配每帧的label。
        
        %首先通过一个函数建立cov快查表
        P = cell(1,h);
        Q = cell(1,h);
        for t=1:h
            P{1,t} = sum(TestData(:,1:t),2);
            Q{1,t} = TestData(:,1:t)*TestData(:,1:t)';
        end

        showText_result1 = 'none';
        showText_result2 = 'none';
        showText_true = 'Truth:';
        % 正确的意思
        trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
        totalsigns = totalsigns + trueSenLen;
        sign_groundTruth_ID = zeros(1, trueSenLen) - 1;      % 正确的Sign ID
        recognizeCount = 0;          % sign_recognized_ID   % 识别出来的的Sign ID
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
                % 显示正确的意思
                text(sum(xlim)/2-0,sum(ylim)/2-210,showText_true,'horiz','center','color','r');
            end
            
            % 快速计算Cov及其子空间，即GCM。
            t = max(k - windowSize/2, 1);
            t_= min(k + windowSize/2, nframes);
            Para_ARMA_test{1}.C = grasp_region(t, t_, P, Q, subSpaceSize);

            test_label(1) = t;
            ValKernel = kernel_Chalearn(Para_ARMA_train,Para_ARMA_test);
            VValKernel = [(1:1)',ValKernel'];
            [predict_label_P1, accuracy_P1, dec_values_P1] = ...
                svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'用来去除输出结果信息
%             index_max = predict_label_P1;

            % 计算SVM的概率，来自one-to-one的信息.
            score = dec_values_score(dec_values_P1, classNum); 
            [score_sort, index_sort] = sort(score,'descend');
            
            % Rank 1 大于阈值的视为有效
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

                % 如果label不重复的话就记录，否则取消记录。
                 if recognizeCount == 0
                     recognizeCount = recognizeCount + 1;
                     sign_recognized_ID(recognizeCount) = predict_label_P1;
                     labelCount(recognizeCount) = 1;    % 记录该label出现的次数
                 elseif sign_recognized_ID(recognizeCount) ~= predict_label_P1 
                     recognizeCount = recognizeCount + 1;
                     sign_recognized_ID(recognizeCount) = predict_label_P1;
                     labelCount(recognizeCount) = 1;
                 else
                     labelCount(recognizeCount) = labelCount(recognizeCount) + 1;
                 end
            end
            
%             % 正确的帧数统计
%             if currentLabel == groundTruth(k)
%                 correctFrame = correctFrame + 1;
%             end
            

            clc;
            fprintf('%s \n%s \n%s \n%s \n', showText_pace, showText_true, showText_result1,showText_result2);
            
        end
        
        %-------------------------------------------------------------------
        % 对句子的整体parsing.
        
        % 删去只出现一次的词。
        % sign_recognized_ID_Final = sign_recognized_ID(find(labelCount~=1));
        
        % 增加语法
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
        
        % 此处对识别和groundTruth这两个序列作比较，返回增删改的数目并统计。 Delete, Insert, Substitue
        [distance, insert, delete, substitute, correctSign] = editDis(sign_groundTruth_ID, sign_recognized_ID);
        totalCorrectSign = totalCorrectSign + correctSign;
        totalDistance = totalDistance + distance;
        totalInsert = totalInsert+insert;
        totalDelete = totalDelete+delete;
        totalSubstitute = totalSubstitute+substitute;
        
        % 输出结果
        fprintf(fid, 'S%s:\t%d\t%d\t%f\t%d\t%d\t%f\t%d\t%d\t%d\t%d\n', sentence_names{sentenceID}(2:5), correctFrame, nframes-windowSize...
            , correctFrame/(nframes-windowSize),correctSign, trueSenLen, correctSign/trueSenLen, distance, insert, delete, substitute);
        totalFrames = totalFrames + nframes-windowSize;
        totalCorrectFrame = totalCorrectFrame + correctFrame;
    end
    
    % 输出最后统计结果
    fprintf(fid, 'Ave.:\t%d\t%d\t%f\t%d\t%d\t%f\t%d\t%d\t%d\t%d \n', totalCorrectFrame,...
        totalFrames, totalCorrectFrame/totalFrames, totalCorrectSign, totalsigns,...
        totalCorrectSign/totalsigns, totalDistance,totalInsert,totalDelete,totalSubstitute);
end
fclose(fid);



