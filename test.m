clear all;
clc;
load score_all_sentence1;
sign_groundTruth_ID = [301 60 238 22 156 215 314 206 252 365];


score_all_new = cell(3,1);
size_score = size(score_all{1},2);
for i=1:size_score
    maxScore = max(max(max(score_all{1}(:,i)),max(score_all{2}(:,i))), max(score_all{3}(:,i)));
    if   maxScore > 0.77  % max(score_all{2}(:,i)) > 0.75
        for w=1:3
            score_all_new{w} = [score_all_new{w} score_all{w}(:,i)];
        end
    end
end

sign_recognized_ID_Final = BP_3D(score_all_new, 370);
% sign_recognized_ID_Final = BP_2D(score_all_new{2}, 370,40);

[distance, insert, delete, substitute, correctSign] = editDis(sign_groundTruth_ID, sign_recognized_ID_Final);

fprintf('Distance: %d, Correctness: %d \n', distance, correctSign);