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
load data\model_MultiSeg_370sign_forP50_new
sign_names = importdata('input\signs_283.txt');   % signs_97  signs_283
sentence_names = importdata('input\sentences_209.txt');
teatDataPath = 'dim334_CTskp_allFrame_manually_209sentences'; 
% dim334_CTskp_fullFrame_209sentences 
% dim334_CTskp_allFrame_manually_209sentences
segPath = 'input\segManually_P08_02.txt';

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
rank = 2;   % ����ǰ������Ч
subSpaceSize = 5;   %�ӿռ��С
draw = 1;    %1:��ʾ��Ƶ�� 0������ʾ��Ƶ
windowSize = 40;   %�������ڵĴ�С
%%
fid = fopen('result\recognized sentence.txt','wt');
for groupID =  1:1
    groupName = ['D:\iData\Outputs\ftdcgrs_whj_output\' teatDataPath '\test_' num2str(groupID) '\'];
    
    % ��1��ʼ��209�����ӱ�ţ� �����ӵ�ID���Ǵ�w0000��ʼ
    for sentenceID = 50:53 %length(sentence_names)    
        fprintf('Processing data: Group %d--Sentence %d\n', groupID, sentenceID);
        data = importdata([groupName sentence_names{sentenceID} '.txt'], ' ', 1);
        [h, w] = size(data.data);  % h:֡��  w:ά��
        TestData = (data.data)';
        nframes = size(TestData, 2);
        

        %����ͨ��һ����������cov����
        for t=1:h
            P{1,t} = sum(TestData(:,1:t),2);
            Q{1,t} = TestData(:,1:t)*TestData(:,1:t)';
        end

%         VideoPath = ['D:\iData\continousSentence\P08_02\S08_'...
%             num2str(sentence_names{sentenceID}(2:5)) '_1_0_20130412.oni\color.avi'];
%         videoObj = mmreader(VideoPath);             %����Ƶ�ļ�
        
        
        showText_result1 = 'none';
        showText_result2 = 'none';
        for k=1:nframes
            if draw == 1
%                 currentFrame = read(videoObj, k);%��ȡ��k֡
%                 imshow(currentFrame);
                tempshow = zeros(480,640);
%                 tempshow = reshape(TestData(1:324),18,18);
                imshow(tempshow);
                xlim=get(gca,'xlim');
                ylim=get(gca,'ylim');
            end

            % ��ʾ��ȷ����˼
            trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
            showText_true = ['Sentence ' sentence_names{sentenceID}(2:5) ', '...
               num2str(k) '/' num2str(nframes) ' frames, '];
            for sign_i = 1:trueSenLen
                sign_choosen_ID = str2num(sentences_meaning_number{1,1+sentenceID}{1,sign_i});
                showText_true = [showText_true chineseIDandMean{1,sign_choosen_ID+1}{1,2} '/'];
            end
            text(sum(xlim)/2-0,sum(ylim)/2-210,showText_true,'horiz','center','color','r');
            
            if k>windowSize/2 && k<nframes - windowSize/2 && mod(k,5)==0
                t = k - windowSize/2;
                t_= k + windowSize/2;

                % ���ټ���cov�����ӿռ䣬��GRASP��
                Para_ARMA_test{1}.C = grasp_region(t, t_, P, Q, subSpaceSize);

                test_label(1) = t;
                ValKernel = kernel_ARMA_Continuous(Para_ARMA_train,Para_ARMA_test);
                VValKernel = [(1:1)',ValKernel'];
                [predict_label_P1, accuracy_P1, dec_values_P1] = ...
                    svmpredict(test_label, VValKernel, model_precomputed,'-q');  % '-q'����ȥ����������Ϣ
                index_max = predict_label_P1 + 1;
                
                % ����SVM�ĸ��ʣ�����one-to-one����Ϣ.
                score = dec_values_score(dec_values_P1, classNum); 
                [score_sort, index_sort] = sort(score,'descend');
                
                if score_sort(1) > 0.75
                    showText_result1 = ['Frame: ' num2str(k) ' /Sign: '...
                        chineseIDandMean{1,index_max}{1,2} ' /score' num2str(score_sort(1))];
                    showText_result2 = ['Candidates: ' chineseIDandMean{1,index_sort(2)}{1,2} '/'...
                                chineseIDandMean{1,index_sort(3)}{1,2} '/'...
                                chineseIDandMean{1,index_sort(4)}{1,2} '/'...
                                chineseIDandMean{1,index_sort(5)}{1,2} ];
                end
            end
            
            if draw == 1
                text(sum(xlim)/2-200,sum(ylim)/2-150,showText_result1,'horiz','center','color','r');
                text(sum(xlim)/2-200,sum(ylim)/2-130,showText_result2,'horiz','center','color','r');
                drawnow;    %ʵʱ��������
            end
        end
        
    end
end
fclose(fid);


