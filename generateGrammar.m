clear all;
clc;

%读取用单词ID集合表示的句子
sentences_meaning_number_Path = 'input\sentences_meaning_number.txt';
sentences_meaning_number = ChineseDataread(sentences_meaning_number_Path);

classNum = 370;
sentenceNum = 209;

C_b = zeros(classNum,classNum);
C_a = zeros(classNum,classNum);


for sentenceID = 1:sentenceNum
    trueSenLen = size(sentences_meaning_number{1,1+sentenceID},2);
    sign_groundTruth_ID = zeros(1, trueSenLen) - 1; 
    for sign_i = 1:trueSenLen
        sign_groundTruth_ID(sign_i) = str2double(sentences_meaning_number{1,1+sentenceID}{1,sign_i});
    end
    for i=1:trueSenLen
        signID = sign_groundTruth_ID(i);
        if i==1
            signID_a = sign_groundTruth_ID(i+1);
            C_b(signID+1, signID_a+1) = C_b(signID+1, signID_a+1) + 1;
        elseif i==trueSenLen
            signID_b = sign_groundTruth_ID(i-1);
            C_a(signID+1, signID_b+1) = C_a(signID+1, signID_b+1) + 1;
        else
            signID_b = sign_groundTruth_ID(i-1);
            signID_a = sign_groundTruth_ID(i+1);
            C_a(signID+1, signID_b+1) = C_a(signID+1, signID_b+1) + 1;
            C_b(signID+1, signID_a+1) = C_b(signID+1, signID_a+1) + 1;
        end
    end
    
end
