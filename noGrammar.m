function recog_sign_ID = noGrammar(score_all)
v_nframes = size(score_all,2);
[v_score_sort, v_index_sort] = sort(score_all,'descend');

recog_sign_ID(1) = v_index_sort(1,1)-1;
count = 2;
for i=2:v_nframes
   % if v_index_sort(1,i)-1~=recog_sign_ID(count-1)
        recog_sign_ID(count) = v_index_sort(1,i)-1;
        count = count+1;
   % end
end