function recog_sign_ID = BP_3D(score_all, classNum, vocabulary, class_correlation)

v_nframes = size(score_all{1},2);

m = zeros(classNum, v_nframes);
m_up = zeros(classNum, v_nframes);
m_down = zeros(classNum, v_nframes);
I = eye(classNum,classNum)*0;
class_correlation_update = class_correlation - I;

iterator = 1;

V = (1-class_correlation_update.^2).^0.5;
D_up = (1-score_all{1}.^2).^0.5;
D = (1-score_all{2}.^2).^0.5;
D_down = (1-score_all{3}.^2).^0.5;

for i=1:iterator
    for t= 2:v_nframes     % beginFrame:endFrame
        fprintf('Processing: %d / %d \n', t, v_nframes);
        for k_t = 1:classNum
            m_list_up = zeros(classNum,1);
            for k_pret=1:classNum
                m_list_up(k_pret) = V(k_t, k_pret) + D_up(k_pret,t-1) + m_up(k_pret,t-1);
            end
            m_up(k_t, t) = min(m_list_up);
        end
        
        for k_t = 1:classNum     
            m_list_down = zeros(classNum,1);
            for k_pret=1:classNum
                m_list_down(k_pret) = V(k_t, k_pret) + D_down(k_pret,t-1) + m_down(k_pret,t-1);
            end
            m_down(k_t, t) = min(m_list_down);
        end
        
        for k_t = 1:classNum
            m_list = zeros(classNum,1);
            for k_pret=1:classNum
                m_list(k_pret) = V(k_t, k_pret) + D(k_pret,t-1) + m(k_pret,t-1); % + m_up(k_pret, t) + m_down(k_pret, t);
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

b_up = zeros(classNum, v_nframes);
for t=1:v_nframes
    for k = 1:classNum
        b_up(k,t) = D_up(k,t) + m_up(k,t);
    end
end

b_down = zeros(classNum, v_nframes);
for t=1:v_nframes
    for k = 1:classNum
        b_down(k,t) = D_down(k,t) + m_down(k,t);
    end
end


[~, v_index_sort] = sort(b,'ascend');  % v_score_sort
[~, v_index_sort_up] = sort(b_up,'ascend');  % v_score_sort
[~, v_index_sort_down] = sort(b_down,'ascend');  % v_score_sort


% recog_sign_ID(1) = v_index_sort(1,1)-1;
recog_sign_ID(1) = vocabulary(v_index_sort(1,1));

count = 2;
for i=2:v_nframes
    fprintf('%d \t %d \t %d \n', v_index_sort(1,i), v_index_sort_up(1,i), v_index_sort_down(1,i));
    label_lay(1) = vocabulary(v_index_sort_up(1,i));
    label_lay(2) = vocabulary(v_index_sort(1,i));
    label_lay(3) = vocabulary(v_index_sort_down(1,i));
    flag = 1;
    
    % 2 vots
    if label_lay(1)~=label_lay(2) && label_lay(1)~=label_lay(3) && label_lay(3)~=label_lay(2)
        flag = 0;
        label_final = -1;
    elseif label_lay(1)==label_lay(2)
        label_final = label_lay(1);
    elseif label_lay(1)==label_lay(3)
        label_final = label_lay(1);
    elseif label_lay(2)==label_lay(3)
        label_final = label_lay(2);
    end

    % 3 vots
%     if label_lay(1)==label_lay(2) && label_lay(1) == label_lay(3)
%         flag =1;
%         label_final = label_lay(1);
%     else
%         flag = 0;
%     end

    if (label_final~=recog_sign_ID(count-1) && flag == 1)
        recog_sign_ID(count) = label_final;
        count = count+1;
    end
end

end