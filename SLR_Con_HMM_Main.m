%����
%����š�����matlab��ʹ�õġ���1��ʼ��
%��ID���Ǿ��ӻ��ߵ��ʱ���ʹ�õģ���w0000��ʼ��
%����֮�������1�Ĺ�ϵ��

clear all;
clc;
%%
addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));
addpath(genpath('D:\iCode\HMM_Matlab\HMMall'));

load data\HMM_model_s3_m3_dim61_370sign_forP1
sentence_names = importdata('input\sentences_209.txt');
teatDataPath = 'dim61_CTskp_fullFrame_manually_209sentences'; 
% dim61_CTskp_fullFrame_209sentences
% dim61_CTskp_fullFrame_manually_209sentences ֻ��P1����,������û������.

% ���ļ���ȷ����ǰ��ά��
idx = strfind(teatDataPath,'_');
dimFinalIdx = idx(1,1)-1;
dim = str2double(teatDataPath(4:dimFinalIdx));

%��ȡ������˼�Ͷ�Ӧ��ID��
ChinesePath = 'input\wordlist_370.txt';
chineseIDandMean = ChineseDataread(ChinesePath);

%��ȡ�õ���ID���ϱ�ʾ�ľ���
sentences_meaning_number_Path = 'input\sentences_meaning_number.txt';
sentences_meaning_number = ChineseDataread(sentences_meaning_number_Path);

classNum = 370;
sample = 5;    %��n֡����
draw = 0;    %1:��ʾ��Ƶ�� 0������ʾ��Ƶ
windowSize = 40;   %�������ڵĴ�С
%%
fid = fopen('result\recognized sentence.txt','wt');
for groupID =  1:1
    groupName = ['D:\iData\Outputs\ftdcgrs_whj_output\' teatDataPath '\test_' num2str(groupID) '\'];
    
    % ��1��ʼ��209�����ӱ�ţ� �����ӵ�ID���Ǵ�w0000��ʼ
    for sentenceID = 101:101 %length(sentence_names)    
        fprintf('Processing data: Group %d--Sentence %d\n', groupID, sentenceID);
        data = importdata([groupName sentence_names{sentenceID} '.txt'], ' ', 1);
        [h, w] = size(data.data);  % h:֡��  w:ά��
        TestData = (data.data)';
        nframes = size(TestData, 2);

        if draw == 1
            VideoPath = ['D:\iData\continousSentence\P08_02\S08_'...
                num2str(sentence_names{sentenceID}(2:5)) '_1_0_20130412.oni\color.avi'];
            videoObj = mmreader(VideoPath);             %����Ƶ�ļ�
        end
        
        showText_result1 = 'none';
        showText_result2 = 'none';
        for k=1:nframes
            % ��ȷ����˼
            trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
            showText_true = ['Sentence ID: ' sentence_names{sentenceID}(2:5) ', '...
               num2str(k) '/' num2str(nframes) ' frames, '];
            for sign_i = 1:trueSenLen
                sign_choosen_ID = str2double(sentences_meaning_number{1,1+sentenceID}{1,sign_i});
                showText_true = [showText_true chineseIDandMean{1,sign_choosen_ID+1}{1,2} '/'];
            end
            
            if draw == 1
                currentFrame = read(videoObj, k);%��ȡ��k֡
                imshow(currentFrame);
                xlim=get(gca,'xlim');
                ylim=get(gca,'ylim');
                % ��ʾ��ȷ����˼
                text(sum(xlim)/2-0,sum(ylim)/2-210,showText_true,'horiz','center','color','r');
            end

            if k>windowSize/2 && k<nframes - windowSize/2 && mod(k,sample)==0
                t = k - windowSize/2;
                t_= k + windowSize/2;
                
                data_norm = TestData(:,t:t_);
     
                loglik = zeros(1,classNum);
                for d=1:classNum
                    loglik(d) = mhmm_logprob(data_norm, prior{d}, transmat{d}, mu{d}, Sigma{d}, mixmat{d});
                end
                
                [score_sort, index_sort] = sort(loglik,'descend');
                index_max = index_sort(1);
                
                showText_result1 = ['Sign: '...
                    chineseIDandMean{1,index_max}{1,2} ' /score' num2str(score_sort(1))];
                showText_result2 = ['Candidates: ' chineseIDandMean{1,index_sort(2)}{1,2} '/'...
                            chineseIDandMean{1,index_sort(3)}{1,2} '/'...
                            chineseIDandMean{1,index_sort(4)}{1,2} '/'...
                            chineseIDandMean{1,index_sort(5)}{1,2} ];
            end
            
            if draw == 1
                text(sum(xlim)/2-200,sum(ylim)/2-150,showText_result1,'horiz','center','color','r');
                text(sum(xlim)/2-200,sum(ylim)/2-130,showText_result2,'horiz','center','color','r');
                drawnow;    %ʵʱ��������
            else
                clc;
                fprintf('%s \n %s \n %s \n', showText_true, showText_result1,showText_result2);
            end
        end
        
    end
end
fclose(fid);



