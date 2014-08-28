%声明
%“编号”是在matlab中使用的。从1开始。
%“ID”是句子或者单词本身使用的，从w0000开始。
%两者之间有相差1的关系。

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

%读取中文意思和对应的ID号
ChinesePath = 'input\wordlist_370.txt';
chineseIDandMean = ChineseDataread(ChinesePath);

%读取用单词ID集合表示的句子
sentences_meaning_number_Path = 'input\sentences_meaning_number.txt';
sentences_meaning_number = ChineseDataread(sentences_meaning_number_Path);

classNum = 370;
rank = 5;   % 考虑前几名有效
l_min = 30;
l_max = 70;
k_pre_step = 10;  % 若是没有检测到合格的sign，前进步长。
fo_la_thre = 0.04;   %large-margin的阈值
subSpaceSize = 5;   %子空间大小
%%
fid = fopen('result\recognized sentence.txt','wt');
for groupID = 2:2
    groupName = ['D:\iData\Outputs\ftdcgrs_whj_output\' teatDataPath '\test_' num2str(groupID) '\'];
    
    % 从1开始的209个句子编号， 而句子的ID都是从w0000开始
    for sentenceID = 2 : 2 % length(sentence_names)    
        fprintf('Processing data: Group %d--Sentence %d\n', groupID, sentenceID);
        data = importdata([groupName sentence_names{sentenceID} '.txt'], ' ', 1);
        [h, w] = size(data.data);  % h:帧数  w:维数
        TestData = (data.data)';
        
        %%%%%%%%%%%%%%%% CRF
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
        
        %%%%%%%%%%%%%%%% Grassmann manifold
        %首先通过一个函数建立cov快查表
        for t=1:h
            P{1,t} = sum(TestData(:,1:t),2);
            Q{1,t} = TestData(:,1:t)*TestData(:,1:t)';
        end

        
        VideoPath = ['D:\iData\continousSentence\P08_02\S08_'...
            num2str(sentence_names{sentenceID}(2:5)) '_1_0_20130412.oni\color.avi'];
        videoObj = mmreader(VideoPath);             %读视频文件
        nframes = get(videoObj, 'NumberOfFrames');  %获取视频文件帧个数
        windowSize = 50;
        %result = zeros(1,nframes) -1;   %初始化result
        
        k_pre = 1;
        showText_result1 = 'none';
        showText_result2 = 'none';
        recognizeSignNum = 1;
        for k=1:nframes
            if k <= l_min + k_pre;
                currentFrame = read(videoObj, k);%读取第i帧
                imshow(currentFrame);
                xlim=get(gca,'xlim');
                ylim=get(gca,'ylim');

                % 显示正确的意思
                trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
                showText_true = ['Sentence ' sentence_names{sentenceID}(2:5) ', '...
                   num2str(k) '/' num2str(nframes) ' frames, '];
                for sign_i = 1:trueSenLen
                    sign_choosen_ID = str2num(sentences_meaning_number{1,1+sentenceID}{1,sign_i});
                    showText_true = [showText_true chineseIDandMean{1,sign_choosen_ID+1}{1,2} '/'];
                end
                text(sum(xlim)/2-0,sum(ylim)/2-210,showText_true,'horiz','center','color','r');
                
                %l_count = '0';
                if k == l_min + k_pre
                    l_range = l_max-l_min+1;
                    score_sort = zeros(l_range,classNum);
                    index_sort = zeros(l_range,classNum)-1;
                    formerRankScore = zeros(l_range,1);
                    latterRankScore = zeros(l_range,1);
                    for l = l_min:l_max
                        t = k_pre;
                        t_= min(k_pre + l,nframes);
                        
                        %l_count = num2str(l);
                        fprintf('Current loop: %d \n', l);
                        
                        % 快速计算cov及其子空间，即GRASP。
                        Para_ARMA_test{1}.C = grasp_region(t, t_, P, Q, subSpaceSize);

                        test_label(1) = t;
                        ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test);
                        VValKernel = [(1:1)',ValKernel'];
                        [predict_label_P1, accuracy_P1, dec_values_P1] = ...
                            svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'用来去除输出结果信息
                        % 计算SVM的概率，来自one-to-one的信息.
                        score = dec_values_score(dec_values_P1, classNum); 

                        [score_sort(l-l_min+1,:),index_sort(l-l_min+1,:)] = sort(score,'descend');
                        formerRankScore(l-l_min+1) = mean(score_sort(l-l_min+1,1:rank));
                        latterRankScore(l-l_min+1) = mean(score_sort(l-l_min+1,rank+1:3*rank));
                    end

                    [maxY_fo_la, maxI_fo_la] = max(formerRankScore - latterRankScore);
                    index_max = index_sort(maxI_fo_la,1);
                    score_max = score_sort(maxI_fo_la,1);
                        %记录结果
                    results(sentenceID,recognizeSignNum) = index_max-1;
                    recognizeSignNum = recognizeSignNum + 1;
                    
                    if maxY_fo_la > fo_la_thre
                        showText_result1 = ['Period: ' num2str(k_pre) '-' num2str(k_pre + l_min + maxI_fo_la)...
                            ' /Sign: ' chineseIDandMean{1,index_max}{1,2}...
                            '  /score: ' num2str(score_max)];

                        showText_result2 = ['former: ' num2str(formerRankScore(maxI_fo_la))...
                            ' /latter: ' num2str(latterRankScore(maxI_fo_la))...
                            ' /fo_la: ' num2str(maxY_fo_la)];
                        k_pre = k_pre + l_min + maxI_fo_la;
                    else
                        k_pre = k_pre + k_pre_step;
                    end
                end
                %text(sum(xlim)/2-200,sum(ylim)/2-110,l_count,'horiz','center','color','r');
                text(sum(xlim)/2-200,sum(ylim)/2-150,showText_result1,'horiz','center','color','r');
                text(sum(xlim)/2-200,sum(ylim)/2-130,showText_result2,'horiz','center','color','r');
                drawnow;    %实时更新命令
            end

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
%             %不保存，每次计算cov。不然可以用C{t,t_}保存
%             C = (1/(t_-t))*((Q{1,t_}-Q{1,t})-(1/(t_-t+1))...  
%                         *((P{1,t_}-P{1,t})*(P{1,t_}-P{1,t})'));
%             % SVD Cov，得到前5维
%             [u,s,v] = svd(C);
%             Para_ARMA_test{1}.C = u(:,1:5);
%             testID = t;    % 暂时用t代替，注意t代表时间
%             test_label(1) = t;
%             ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test,testID);
%             VValKernel = [(1:1)',ValKernel'];
%             [predict_label_P1, accuracy_P1, dec_values_P1] = ...
%                 svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'用来去除输出结果信息
%             result(sentenceID, segSign) = predict_label_P1 + 1;
%         end
        
%         start = 1;
%         for t=start:h
%             fprintf('Current start frame: %d / %d \n',t,h);
%             for t_=t+1:h
%                 if t == 1
%                     C = (1/t_-1)*(Q{1,t_}-(1/t_)*(P{1,t_}*(P{1,t_})'));
%                 else
%                     C = (1/(t_-t))*((Q{1,t_}-Q{1,t})-(1/(t_-t+1))...   %不保存，每次计算cov。不然可以用C{t,t_}保存
%                         *((P{1,t_}-P{1,t})*(P{1,t_}-P{1,t})'));
%                 end
%                     % SVD Cov，得到前5维
%                 [u,s,v] = svd(C);
%                 Para_ARMA_test{1}.C = u(:,1:5);
%                 testID = t;    % 暂时用t代替，注意t代表时间
%                 test_label(1) = t;
%                 ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test,testID);
%                 VValKernel = [(1:1)',ValKernel'];
%                 [predict_label_P1, accuracy_P1, dec_values_P1] = ...
%                     svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'用来去除输出结果信息
%                 result(t-start+1,t_) = predict_label_P1;
%                 
%             end
%         end
   
        
        %边分割边识别策略

    end
end
fclose(fid);

%% 废弃代码
%         k_p = 1;
%         for k = 1 : nframes-l_max
%         k = 1;
%         while k < nframes-l_max
%             
%             currentFrame = read(videoObj, k);%读取第i帧
%             imshow(currentFrame);
%             xlim=get(gca,'xlim');
%             ylim=get(gca,'ylim');
%             
%             % 显示正确的意思
%             trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
%             showText_true = [sentence_names{sentenceID}(2:5) ' Groundtruth: '];
%             for sign_i = 1:trueSenLen
%                 sign_choosen_ID = str2num(sentences_meaning_number{1,1+sentenceID}{1,sign_i});
%                 showText_true = [showText_true chineseIDandMean{1,sign_choosen_ID+1}{1,2} '/'];
%             end
%             text(sum(xlim)/2-0,sum(ylim)/2-210,showText_true,'horiz','center','color','r');
%             
% 
%             l_range = l_max-l_min+1;
%             score_sort = zeros(l_range,classNum);
%             index_sort = zeros(l_range,classNum)-1;
%             formerRankScore = zeros(l_range,1);
%             latterRankScore = zeros(l_range,1);
%             for l = l_min:l_max
%                 t = k;
%                 t_= k + l;
% 
%                 % 快速计算cov及其子空间，即GRASP。
%                 Para_ARMA_test{1}.C = grasp_region(t, t_, P, Q, subSpaceSize);
% 
%                 test_label(1) = t;
%                 ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test);
%                 VValKernel = [(1:1)',ValKernel'];
%                 [predict_label_P1, accuracy_P1, dec_values_P1] = ...
%                     svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'用来去除输出结果信息
%                 % 计算SVM的概率，来自one-to-one的信息.
%                 score = dec_values_score(dec_values_P1, classNum); 
% 
%                 [score_sort(l-l_min+1,:),index_sort(l-l_min+1,:)] = sort(score,'descend');
%                 formerRankScore(l-l_min+1) = mean(score_sort(l-l_min+1,1:rank));
%                 latterRankScore(l-l_min+1) = mean(score_sort(l-l_min+1,rank+1:3*rank));
%             end
% 
%             [maxY_fo_la, maxI_fo_la] = max(formerRankScore - latterRankScore);
%             index_max = index_sort(maxI_fo_la,1);
%             score_max = score_sort(maxI_fo_la,1);
%             if maxY_fo_la > fo_la_thre
%                 showText = ['Frame: ' num2str(k) '; Sign: '...
%                 chineseIDandMean{1,index_max}{1,2} ' / score: ' num2str(score_max)];
%                 text(sum(xlim)/2-200,sum(ylim)/2-150,showText,'horiz','center','color','r');
% 
%                 showText = ['former: ' num2str(formerRankScore(maxI_fo_la))...
%                     ' /latter: ' num2str(latterRankScore(maxI_fo_la))...
%                     ' /fo_la: ' num2str(maxY_fo_la)];
%                 text(sum(xlim)/2-200,sum(ylim)/2-130,showText,'horiz','center','color','r');
%                 
%                 k = k + l_min + maxI_fo_la;
%             else
%                 showText = 'None';
%                 text(sum(xlim)/2-200,sum(ylim)/2-130,showText,'horiz','center','color','r');
%                 
%                 k = k + 10;
%             end
% 
%             
%             
% 
% %             if k>windowSize/2 && k<nframes - windowSize/2
% %                 t = k-windowSize/2;
% %                 t_= k+windowSize/2;
% %                 
% %                 % 快速计算cov及其子空间，即GRASP。
% %                 Para_ARMA_test{1}.C = grasp_region(t, t_, P, Q, subSpaceSize);
% %                 
% %                 test_label(1) = t;
% %                 ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test);
% %                 VValKernel = [(1:1)',ValKernel'];
% %                 [predict_label_P1, accuracy_P1, dec_values_P1] = ...
% %                     svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'用来去除输出结果信息
% %                 result(k) = predict_label_P1;  % 注意，结果是从0到369,这里表示的是sign的ID号。
% %                 score = dec_values_score(dec_values_P1, classNum); 
% %                 [score_max, index_max] = max(score); % 注意，index_max是从1到370。
% %                 [score_sort,index_sort] = sort(score,'descend');
% %                 formerRankScore = mean(score_sort(1:rank));
% %                 latterRankScore = mean(score_sort(rank+1:3*rank));
% %                
% %                 if formerRankScore - latterRankScore > fo_la_thre
% %                     showText = ['Frame: ' num2str(k) '; Sign: '...
% %                     chineseIDandMean{1,index_max}{1,2} ' / score: ' num2str(score_max)];
% %                     text(sum(xlim)/2-200,sum(ylim)/2-150,showText,'horiz','center','color','r');
% % 
% %                     showText = ['former: ' num2str(formerRankScore) ' /latter: ' num2str(latterRankScore)];
% %                     text(sum(xlim)/2-200,sum(ylim)/2-130,showText,'horiz','center','color','r');
% %                 else
% %                     showText = 'None';
% %                     text(sum(xlim)/2-200,sum(ylim)/2-130,showText,'horiz','center','color','r');
% %                 end
% %                 
% %                 
% %             end
%             
%             drawnow;    %实时更新命令
%         end

