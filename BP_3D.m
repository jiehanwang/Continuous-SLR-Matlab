function recog_sign_ID = BP_3D(score_all, classNum)
load data\class_correlation_model;   % class_correlation: 370*370
v_nframes = size(score_all{1},2);

m = zeros(classNum, v_nframes);
m_up = zeros(classNum, v_nframes);
m_down = zeros(classNum, v_nframes);
I = eye(classNum,classNum)*0;
class_correlation_update = class_correlation - I;

iterator = 1;

for i=1:iterator
    for t= 2:v_nframes     % beginFrame:endFrame
        fprintf('Processing: %d / %d \n', t, v_nframes);
        for k_t = 1:classNum
            m_list_up = zeros(classNum,1);
            for k_pret=1:classNum
                V_up = (1-class_correlation_update(k_t, k_pret)^2)^0.5;
                D_up = (1-score_all{1}(k_pret,t-1)^2)^0.5;
                m_list_up(k_pret) = V_up + D_up + m_up(k_pret,t-1);
            end
            m_up(k_t, t) = min(m_list_up);
        end
        
        for k_t = 1:classNum     
            m_list_down = zeros(classNum,1);
            for k_pret=1:classNum
                V_down = (1-class_correlation_update(k_t, k_pret)^2)^0.5;
                D_down = (1-score_all{3}(k_pret,t-1)^2)^0.5;
                m_list_down(k_pret) = V_down + D_down + m_down(k_pret,t-1);
            end
            m_down(k_t, t) = min(m_list_down);
        end
        
        for k_t = 1:classNum
            m_list = zeros(classNum,1);
            for k_pret=1:classNum
                V = (1-class_correlation_update(k_t, k_pret)^2)^0.5;
                D = (1-score_all{2}(k_pret,t-1)^2)^0.5;
                m_list(k_pret) = V + D + m(k_pret,t-1); % + m_up(k_pret, t) + m_down(k_pret, t);
            end    
            m(k_t, t) = min(m_list);
        end
    end
end


b = zeros(classNum, v_nframes);
for t=1:v_nframes
    for k = 1:classNum
        D = (1-score_all{2}(k,t)^2)^0.5;
        b(k,t) = D + m(k,t);
    end
end

b_up = zeros(classNum, v_nframes);
for t=1:v_nframes
    for k = 1:classNum
        D = (1-score_all{1}(k,t)^2)^0.5;
        b_up(k,t) = D + m_up(k,t);
    end
end

b_down = zeros(classNum, v_nframes);
for t=1:v_nframes
    for k = 1:classNum
        D = (1-score_all{3}(k,t)^2)^0.5;
        b_down(k,t) = D + m_down(k,t);
    end
end


[~, v_index_sort] = sort(b,'ascend');  % v_score_sort
[~, v_index_sort_up] = sort(b_up,'ascend');  % v_score_sort
[~, v_index_sort_down] = sort(b_down,'ascend');  % v_score_sort


recog_sign_ID(1) = v_index_sort(1,1)-1;
count = 2;
for i=2:v_nframes
    fprintf('%d \t %d \t %d \n', v_index_sort(1,i), v_index_sort_up(1,i), v_index_sort_down(1,i));
    if (v_index_sort(1,i)-1~=recog_sign_ID(count-1) &&...
            (v_index_sort(1,i) == v_index_sort_down(1,i)) && (v_index_sort(1,i) == v_index_sort_up(1,i)))
        recog_sign_ID(count) = v_index_sort(1,i)-1;
        count = count+1;
    end
end

end