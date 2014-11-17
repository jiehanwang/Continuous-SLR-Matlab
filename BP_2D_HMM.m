function recog_sign_ID = BP_2D_HMM(score_all, classNum, vocabulary, class_correlation)
score_all(score_all<=0)=10000;
min_score = min(min(score_all));
score_all(score_all>9999)=0;
max_score = max(max(score_all));
for i=1:size(score_all,1)
    for j=1:size(score_all,2)
        if score_all(i,j)>0
            score_all(i,j) = (score_all(i,j)-min_score)/(max_score-min_score);
        end
    end
end

v_nframes = size(score_all,2);
m = zeros(classNum, v_nframes);
I = eye(classNum,classNum)*0;
class_correlation_update = class_correlation - I;
Vweight = 1;
iterator = 1;

V = Vweight*(1-class_correlation_update.^2).^0.5;
D =(1-score_all.^2).^0.5;

for i=1:iterator
    for t= 2:v_nframes     % beginFrame:endFrame
        fprintf('Processing: %d / %d \n', t, v_nframes);
        for k_t = 1:classNum
            m_list = zeros(classNum,1);
            for k_pret=1:classNum
                m_list(k_pret) = V(k_t, k_pret) + D(k_pret,t-1) + m(k_pret,t-1);
            end
            m(k_t, t) = min(m_list);
        end
    end
end


b = zeros(classNum, v_nframes);
for t=1:v_nframes
    for k = 1:classNum
        b(k,t) = D(k,t) + m(k,t);
    end
end

[~, v_index_sort] = sort(b,'ascend');  % v_score_sort
recog_sign_ID(1) = str2double(vocabulary{v_index_sort(1,1),1}(2:5));
count = 2;
for i=2:v_nframes
    recog_sign = str2double(vocabulary{v_index_sort(1,i),1}(2:5));
    if recog_sign~=recog_sign_ID(count-1)
        recog_sign_ID(count) = recog_sign;
        count = count+1;
    end
end

end