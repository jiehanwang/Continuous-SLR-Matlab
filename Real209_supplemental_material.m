%声明
%“编号”是在matlab中使用的。从1开始。
%“ID”是句子或者单词本身使用的，从w0000开始。
%两者之间有相差1的关系。
clear all;
clc;
%% Settings and Initials
addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));

% 读取模型库
load data\model_noSeg_209sen_242sign_forP0801;
%model_noSeg_209sen_242sign_forP0801
%model_2Seg_150sen_242sign_forP0801

% 读取 class_correlation变量。即，类间关系图。
load data\class_correlation_model_242_noseg;   

% 读取测试库
sentence_names = importdata('input\sentences_150_4.txt');
teatDataPath = 'dim334_CTskp_fullFrame_209sentences'; 

% 读取用单词ID集合表示的句子
sentences_meaning_number_Path = 'input\sentences_meaning_ID_real150_4.txt';  
sentences_meaning_number = ChineseDataread(sentences_meaning_number_Path);

% 从文件名确定当前的维数
idx = strfind(teatDataPath,'_');
dimFinalIdx = idx(1,1)-1;
dim = str2double(teatDataPath(4:dimFinalIdx));

% 读取中文意思和对应的ID号
ChinesePath = 'input\wordlist_370.txt';
chineseIDandMean = ChineseDataread(ChinesePath);

% 读取测试词汇ID
% vocabulary = importdata('input\sign_150_forReal209.txt');
vocabulary = model_precomputed.Label;

classNum = 242;     % 
subSpaceSize = 5;   % 子空间大小  5
gap = 3;            % 隔n帧采样
thre = 0.8;        % score>thre 的视为有效
draw = 1;           % 1:显示视频。 0：不显示视频
windowSize = 30;    % 滑动窗口的大小
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
    
    % 从1开始的句子编号， 而句子的ID都是从w0000开始
    for sentenceID = 3:3%length(sentence_names)    
        sign_recognized_ID = [];
        sign_recognized_ID_Final = [];
        fprintf('Processing data: Group %d--Sentence %d\n', groupID, sentenceID);
        data = importdata([groupName sentence_names{sentenceID} '.txt'], ' ', 1);
        %groundTruth_ = importdata([groundTruthFileFolderName sentence_names{sentenceID} '.txt'], ' ', 1);
        [h, w] = size(data.data);  % h:帧数  w:维数
        TestData = (data.data)';
        %groundTruth = groundTruth_.data;
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
        
        %读视频文件
        if draw == 1
            VideoPath = ['D:\iData\continousSentence\P08_01\S08_'...
                num2str(sentence_names{sentenceID}(2:5)) '_1_0_20130412.oni\color.avi'];
            videoObj = mmreader(VideoPath);  
        end
                   
        showText_result1 = 'none';
        showText_result2 = 'none';
        showText_true = 'Truth:';
        % 正确的意思
        trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
        sign_groundTruth_ID = [];
        recognizeCount = 0;          % sign_recognized_ID   % 识别出来的的Sign ID
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
                currentFrame = read(videoObj, k);%读取第i帧
                imshow(currentFrame);
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
            ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test);
            VValKernel = [(1:1)',ValKernel'];
            [predict_label_P1, accuracy_P1, dec_values_P1] = ...
                svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'用来去除输出结果信息
            index_max = predict_label_P1 + 1;

            % 计算SVM的概率，来自one-to-one的信息.
            dec_values_P1(dec_values_P1<-1) = -1;
            dec_values_P1(dec_values_P1>1) = 1;
            score = dec_values_score(dec_values_P1, classNum); 
            [score_sort, index_sort] = sort(score,'descend');
            score_max = max(score);
            
            % Rank 1 大于阈值的视为有效
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
            
            clc;
            fprintf('%s \n%s \n%s \n%s \n', showText_pace, showText_true, showText_result1,showText_result2);
            
            if draw == 1
                %text(sum(xlim)/2-200,sum(ylim)/2-110,l_count,'horiz','center','color','r');
                text(sum(xlim)/2-200,sum(ylim)/2-150,showText_result1,'horiz','center','color','r');
%                 text(sum(xlim)/2-200,sum(ylim)/2-130,showText_result2,'horiz','center','color','r');
%                 text(sum(xlim)/2-200,sum(ylim)/2-110,showText_result_candidate,'horiz','center','color','r');
                drawnow;    %实时更新命令
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
        % 此处对识别和groundTruth这两个序列作比较，返回增删改的数目并统计。 Delete, Insert, Substitue
        [distance2, insert, delete, substitute, correctSign] = editDis(sign_groundTruth_ID, sign_recognized_ID_Final);
        distance = levenshtein(sign_groundTruth_ID, sign_recognized_ID_Final);
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



