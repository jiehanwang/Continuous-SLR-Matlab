function recog_sign_ID = viterbi_like2(score_all, classNum, formerN)

load data\class_correlation_model;
v_nframes = size(score_all,2);

v_score = zeros(classNum, v_nframes);
v_score(:,1) = score_all(:,1);

restart = 0;
for col = 2:v_nframes
    if restart == 1;
        v_score(:,col) = score_all(:,col);
        restart = 0;
        continue;
    end
    for row = 1:classNum
        temp = zeros(classNum,1);
        for row_pre = 1:classNum
            temp(row_pre) = v_score(row_pre, col-1)*class_correlation(row, row_pre);
        end
        v_score(row, col) = max(temp)*score_all(row, col);
    end
    [v_score_col_sort, v_index_col_sort] = sort(v_score(:, col),'descend');
    [score_col_sort, index_col_sort] = sort(score_all(:, col),'descend');
    
    dis(col)= formerN-size( intersect(v_index_col_sort(1:formerN)', index_col_sort(1:formerN)'),2);
    if dis(col) >= 15
        restart = 1;
    end
end

[v_score_sort, v_index_sort] = sort(v_score,'descend');

recog_sign_ID(1) = v_index_sort(1,1)-1;
count = 2;
for i=2:v_nframes
    if v_index_sort(1,i)-1~=recog_sign_ID(count-1)
        recog_sign_ID(count) = v_index_sort(1,i)-1;
        count = count+1;
    end
end

end

% a_b = zeros(1,v_nframes);
% for i=1:v_nframes
%     a = score_all(:,i)/max(score_all(:,i));
%     b = v_score_sort(:,i)/max(v_score_sort(:,i));
%     a_b(i) = sum(abs(a-b));
% end