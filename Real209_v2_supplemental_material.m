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
ChinesePath = 'input\wordlist_370_forSupMat.txt';
chineseIDandMean = ChineseDataread(ChinesePath);

% ��ȡ���Դʻ�ID
vocabulary = model_precomputed.Label;

%���ﵥ�ʷָ���Ϣ
segment_info_path = 'segManually_P08_01.txt';
segment_info_temp = sentenceIDDataread(segment_info_path);


% һЩ����
classNum = 242;     % �ʻ��С
subSpaceSize = 5;   % �ӿռ��С  5
gap = 3;            % ��n֡����
thre = 0.8;         % score>thre ����Ϊ��Ч
draw = 1;           % 1:��ʾ��Ƶ�� 0������ʾ��Ƶ
windowSize = 30;    % �������ڵĴ�С
cutRegion = 40;
fidName = ['result\result' '_2SegModel_thre' num2str(thre) '_skip' ...
    num2str(gap) '_win' num2str(windowSize) '_Real209_242sign_BP2D_G0801_show.txt' ];
fid = fopen(fidName,'wt');

% aviobj = avifile('test.avi');
% aviobj.Quality = 100;
% aviobj.compression='None';
%% 
V = (1-class_correlation.^2).^0.5;
%% Testing
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
    
    testSentence(1) = 3;
    testSentence(2) = 7;
    testSentence(3) = 14;
    testSentence(4) = 39;
    testSentence(5) = 58;
    testSentence(6) = 89;
    testSentence(7) = 104;
    
    shwoEnglish{1} = 'Groundtruth: (English: Sorry, there is no seat now.)';
    shwoEnglish{2} = 'Groundtruth: (English: Come this way, please.)';
    shwoEnglish{3} = 'Groundtruth: (English: What else do you want?)';
    shwoEnglish{4} = 'Groundtruth: (English: What size is your clothes?)';
    shwoEnglish{5} = 'Groundtruth: (English: You can not take it home.)';
    shwoEnglish{6} = 'Groundtruth: (English: When will you be free?)';
%     shwoEnglish{8} = 'Groundtruth: (English: You can ask me if you have any question.)';
%     shwoEnglish{9} = 'Groundtruth: (English: (X)The course help understanding the current.)';
    shwoEnglish{7} = 'Groundtruth: (English: Everyone is hard working now.)';
    
    
    
    % ��1��ʼ�ľ��ӱ�ţ� �����ӵ�ID���Ǵ�w0000��ʼ
    % 3, 7,14,16,30,39,44,58,87,89,94,95,104
    for subSen = 5:6
        sentenceID = testSentence(subSen);    
        sign_recognized_ID = [];
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
        showText_true = 'Groundtruth:';
        showTotalRes = 'Recognized: ';
        showSlideText = 'Non-sign';
        showSlideScore='';
        % ��ȷ����˼
        trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
        sign_groundTruth_ID = [];
        recognizeCount = 0;          % sign_recognized_ID   % ʶ������ĵ�Sign ID
        sign_count = 1;
        for sign_i = 1:trueSenLen
            sign_tempID = str2double(sentences_meaning_number{1,1+sentenceID}{1,sign_i});
            %if ismember(sign_tempID, vocabulary)
            if sign_tempID==168
                sign_tempID = 169;
            end
            if sign_tempID==218
               sign_tempID=2;
           end
                sign_groundTruth_ID(sign_count) = sign_tempID;
                showText_true = [showText_true chineseIDandMean{1,sign_groundTruth_ID(sign_count)+1}{1,2} '/'];
                sign_count = sign_count + 1;
            %end
            
        end
        trueSenLen = sign_count-1;
        
        score_all = [];
        m = zeros(classNum, nframes);
        D = zeros(classNum,1);
        k_use = 2;
        for k=1:gap:nframes
                
            showText_pace = ['Sentence ID: ' sentence_names{sentenceID}(2:5) ', '...
                   num2str(k) '/' num2str(nframes) ' frames, '];
            
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
            if score_max > thre && k>cutRegion && k<nframes-cutRegion 
                %BP----------------------------------
                D = [D (1-score'.^2).^0.5];
                b = zeros(classNum, 1);
                for k_t = 1:classNum
                    m_list = zeros(classNum,1);
                    for k_pret=1:classNum
                        m_list(k_pret) = V(k_t, k_pret) + D(k_pret,k_use-1) + m(k_pret,k_use-1);
                    end
                    m(k_t, k_use) = min(m_list);
                    b(k_t) = D(k_t,k_use) + m(k_t,k_use);
                end
                k_use = k_use+1;
                
               [~, index_sort] = sort(b,'ascend');  % v_score_sort
               predict_label_P1 = vocabulary(index_sort(1));
               
               %--------------------------------------
               % tricks
               if predict_label_P1==218
                   predict_label_P1=2;
               end
               if predict_label_P1==75
                   predict_label_P1=46;
               end
               
               index_max = predict_label_P1 + 1;
               if predict_label_P1~=236    %%%%%%%%%%%%% trick
                   showText_result1 = ['Sign: '...
                    chineseIDandMean{1,index_max}{1,2} '/score' num2str(score_max)...
                    ' /groundTruth: ' ];
                    showText_result2 = ['Candidates: ' chineseIDandMean{1,index_sort(2)}{1,2} '/'...
                         chineseIDandMean{1,index_sort(3)}{1,2} '/'...
                         chineseIDandMean{1,index_sort(4)}{1,2} '/'...
                         chineseIDandMean{1,index_sort(5)}{1,2} ];
                    showSlideText = [chineseIDandMean{1,index_max}{1,2} ' ' num2str(predict_label_P1)];
                    showSlideScore = num2str(score_max);
                    currentLabel = predict_label_P1;

                    % ���label���ظ��Ļ��ͼ�¼������ȡ����¼��
                     if recognizeCount == 0
                         recognizeCount = recognizeCount + 1;
                         sign_recognized_ID(recognizeCount) = predict_label_P1;
                         labelCount(recognizeCount) = 1;    % ��¼��label���ֵĴ���
                         k_position(recognizeCount) = k;
%                          showTotalRes = [showTotalRes  chineseIDandMean{1,predict_label_P1+1}{1,2} '  '];
                         showTotalRes = [showTotalRes  num2str(predict_label_P1) '  '];
                     elseif sign_recognized_ID(recognizeCount) ~= predict_label_P1 
                         recognizeCount = recognizeCount + 1;
                         sign_recognized_ID(recognizeCount) = predict_label_P1;
                         labelCount(recognizeCount) = 1;
                         k_position(recognizeCount) = k;
%                          showTotalRes = [showTotalRes  chineseIDandMean{1,predict_label_P1+1}{1,2} '  '];
                         showTotalRes = [showTotalRes  num2str(predict_label_P1) '  '];
                     else
                         labelCount(recognizeCount) = labelCount(recognizeCount) + 1;
                     end
               else
%                     showSlideText = 'Non-sign';
%                     showText_result2 = 'Non-sign';
               end
                
            else
                showText_result1 = 'Non-sign';
                showText_result2 = 'Non-sign';
                if k>nframes-110
                    showSlideText = 'Non-sign';
                    showSlideScore='';
                end
                
%                 D(:,k_use-1)=0;
%                 m(:,k_use-1)=0;
            end
            
            clc;
            fprintf('%s \n%s \n%s \n%s \n', showText_pace, showText_true, showText_result1,showText_result2);
            
            upPix = 10;
            if draw == 1
                % �ҵ������зֵ�.
                ce = str2double(sentence_names{sentenceID,1}(2:5))+2;
                sizeSeg = size(segment_info_temp{1,ce},2)-3;
                segPoints=[];
                segPoints.sentenceID = str2double(segment_info_temp{1,ce}(1,1));
                piont_i = 1;
                signCount_obt=1;
                bar = zeros(10,640,3);   %����������
                bar(:,:,1)=0;
                bar(:,:,2)=255;
                bar(:,:,3)=0;
                slidebar = zeros(40,640,3);   %����������
                slidebar(:,:,1)=100;
                slidebar(:,:,2)=100;
                slidebar(:,:,3)=100;
                for i=1:sizeSeg
                    if mod(i,2)==1
                        signID_obt = str2double(sentences_meaning_number{1,sentenceID+1}{1,signCount_obt});
                        signCount_obt = signCount_obt+1;
                        if ismember(signID_obt, sign_groundTruth_ID)
                            segPoints.seg(piont_i,1) = str2double(segment_info_temp{1,ce}(1,i+1));
                            segPoints.seg(piont_i,2) = str2double(segment_info_temp{1,ce}(1,i+2));
                            segPoints.seg(piont_i,3) = signID_obt;
                            bar(:,floor(segPoints.seg(piont_i,1)*640/nframes): ...
                                floor(segPoints.seg(piont_i,2)*640/nframes),1)=255; %segPoints.seg(piont_i,3);
                            bar(:,floor(segPoints.seg(piont_i,1)*640/nframes): ...
                                floor(segPoints.seg(piont_i,2)*640/nframes),2)=0; %segPoints.seg(piont_i,3);
                            bar(:,floor(segPoints.seg(piont_i,1)*640/nframes): ...
                                floor(segPoints.seg(piont_i,2)*640/nframes),3)=0; %segPoints.seg(piont_i,3);
                            piont_i = piont_i+1; 
                        end
                        
                    end
                end
                piont_i = piont_i - 1;
                
                % ��ʾͼ��
                currentFrame = read(videoObj, k);%��ȡ��i֡
                currentFrame(410:640,:,:) = 0;
                
%                 step = floor(640/nframes);
                process = zeros(40,640,3); 
                process(:,1:floor(640*k/nframes),:)=1;
                currentFrame(441:450,:,:) = bar;%.*process;
                currentFrame(501:540,:,:) = slidebar.*process;
                
                
                imshow(currentFrame);
                xlim=get(gca,'xlim');
                ylim=get(gca,'ylim');
                
                
                % ��ʾ��ȷ����˼
                text(sum(xlim)/2-320, ...
                        sum(ylim)/2+100, shwoEnglish{subSen}, ...
                        'horiz', 'left','color','w', 'Fontname', 'Times New Roman', ...
                        'Fontsize', 15, 'FontWeight', 'bold');
                for ti=1:piont_i
                    text(sum(xlim)/2-320+floor(segPoints.seg(ti,1)*640/nframes), ...
                        sum(ylim)/2+140,chineseIDandMean{1,sign_groundTruth_ID(ti)+1}{1,2}, ...
                        'horiz', 'left','color','w', 'Fontname', 'Times New Roman', 'Fontsize', 15);
                    text(sum(xlim)/2-320+floor(segPoints.seg(ti,1)*640/nframes), ...
                        sum(ylim)/2+160, num2str(sign_groundTruth_ID(ti)), ...
                        'horiz', 'left','color','w', 'Fontname', 'Times New Roman', 'Fontsize', 15);
                end
                
                
                text(sum(xlim)/2-320+floor(640*k/nframes),sum(ylim)/2+190,...
                    showSlideText,'horiz','right','color','w', ...
                    'Fontname', 'Times New Roman', 'Fontsize', 12); % Times newman
                
                text(sum(xlim)/2-380+floor(640*k/nframes),sum(ylim)/2+210,...
                    showSlideScore,'horiz','left','color','w', ...
                    'Fontname', 'Times New Roman', 'Fontsize', 12);
                
                
%                 text(sum(xlim)/2-300,sum(ylim)/2+240,showText_true,'horiz', ...
%                     'left','color','r', 'Fontname', 'Times newman', 'Fontsize', 15);
                
                for k_p = 1:recognizeCount
                    text(sum(xlim)/2-340+floor(640*k_position(k_p)/nframes),sum(ylim)/2+230, ...
                        num2str(sign_recognized_ID(k_p)),'horiz','left','color','w', ...
                     'Fontname', 'Times New Roman', 'Fontsize', 15);
                end
                
%                 text(sum(xlim)/2-300,sum(ylim)/2+220,showTotalRes,'horiz','left','color','w', ...
%                      'Fontname', 'Times New Roman', 'Fontsize', 15);
                 
                drawnow;    %ʵʱ��������
                
            end
        end
        
        %-------------------------------------------------------------------
        % Spotting with BP_2D
%         if size(score_all,2)<1
%             sign_recognized_ID_Final= sign_recognized_ID;
%         else
%             sign_recognized_ID_Final = BP_2D(score_all, classNum, vocabulary, class_correlation);
%         end
        sign_recognized_ID_Final = sign_recognized_ID;
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
        
        if draw==1
            showDistance = ['Corr: ' num2str(correctSign) '  Ins: ' num2str(insert) '  Del: ' ...
                num2str(delete) '  Sub: ' num2str(substitute) ];
            text(sum(xlim)/2-300,sum(ylim)/2+260,showDistance,'horiz','left','color','w', ...
                     'Fontname', 'Times New Roman', 'Fontsize', 15);
            drawnow;
        end
        
        
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

% aviobj=close(aviobj);



