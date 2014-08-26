clear all;
addpath('bin');
% load sampleData;
% load sign
load data\data_test_dim10_97

paramsData.weightsPerSequence = ones(1,128) ;
paramsData.factorSeqWeights = 1;
%%
paramsNodCRF.normalizeWeights = 1;
R{1}.params = paramsNodCRF;
disp('Training begin!');
T = fix(clock)
[R{1}.model R{1}.stats] = train(trainSeqs, trainLabels, R{1}.params);
disp('Training end!');
T = fix(clock)
[R{1}.ll R{1}.labels] = test(R{1}.model, testSeqs, testLabels);
disp('Test end!');
T = fix(clock)
% x=1:620;
% plot(x,R{1,1}.ll{1,1}(1,1:620),x,R{1,1}.ll{1,1}(2,1:620),x,R{1,1}.ll{1,1}(3,1:620),x,R{1,1}.ll{1,1}(4,1:620),x,R{1,1}.ll{1,1}(5,1:620),x,R{1,1}.ll{1,1}(6,1:620),x,R{1,1}.ll{1,1}(7,1:620),x,R{1,1}.ll{1,1}(8,1:620),x,R{1,1}.ll{1,1}(9,1:620),x,R{1,1}.ll{1,1}(10,1:620));

%% 
% paramsNodHCRF.normalizeWeights = 1;
% R{2}.params = paramsNodHCRF;
% [R{2}.model R{2}.stats] = train(trainCompleteSeqs, trainCompleteLabels, R{2}.params);
% [R{2}.ll R{2}.labels] = test(R{2}.model, testSeqs, testLabels);
%% 
% paramsNodLDCRF.normalizeWeights = 1;
% R{3}.params = paramsNodLDCRF;
% disp('Training begin!');
% T = fix(clock)
% [R{3}.model R{3}.stats] = train(trainSeqs, trainLabels, R{3}.params);
% disp('Training end!');
% T = fix(clock)
% [R{3}.ll R{3}.labels] = test(R{3}.model, testSeqs, testLabels);
% disp('Test end!');
% T = fix(clock)
%%
% correct = 0;
% for i=1:21701
%     if R{1}.ll{1}(1,i)>R{1}.ll{1}(2,i)
%         result(i) = 0;
%     else
%         result(i) = 1;
%     end
%     if result(i) == testLabels{1,1}(i)
%         correct = correct + 1;
%     end
% end

%%
% plotResults(R);
