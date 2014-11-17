function recog_sign_ID = BP_2D_real209(score_all, classNum, vocabulary, class_correlation)
v_nframes = size(score_all,2);
m = zeros(classNum, v_nframes);

% 权重设置，目前不生效
I = eye(classNum,classNum)*0;
class_correlation_update = class_correlation - I;
Vweight = 1;
iterator = 1;

V = Vweight*(1-class_correlation_update.^2).^0.5;
D = (1-score_all.^2).^0.5;

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

% 计算Final的cost score
b = zeros(classNum, v_nframes);
for t=1:v_nframes
    for k = 1:classNum
        b(k,t) = D(k,t) + m(k,t);
%         b(k,t) = D(k,t);
    end
end

% Sorting
[~, v_index_sort] = sort(b,'ascend');  % v_score_sort
recog_sign_score = zeros(1,v_nframes) -1;

% Obtain the first rank label for each frame
recog_sign = vocabulary(v_index_sort(1,1));
recog_sign_ID(1) = recog_sign;
recog_sign_score(1) = score_all(v_index_sort(1,1),1);
count = 2;
for i=2:v_nframes
    % 查找在vocabulary中的ID号
    recog_sign = vocabulary(v_index_sort(1,i));
    recog_sign_score_temp = score_all(v_index_sort(1,i),i);
    % 剔除连续重复的ID
    if recog_sign~=recog_sign_ID(count-1)
        recog_sign_ID(count) = recog_sign;
        recog_sign_score(count) = recog_sign_score_temp;
        count = count+1;
    elseif recog_sign == recog_sign_ID(count-1)
        if recog_sign_score(count-1) < recog_sign_score_temp
            recog_sign_score(count-1) = recog_sign_score_temp;
        end
    end
end
count = count - 1;   % 最后的识别出来的手语词数目

if count>10
    result_score = recog_sign_score(1:count);
    [~, index_result_score_sort] = sort(result_score, 'descend');
    [select, ~] = sort(index_result_score_sort(1:10), 'ascend');
    recog_sign_ID_10 = recog_sign_ID(select);
else
    recog_sign_ID_10 = recog_sign_ID;
end

end