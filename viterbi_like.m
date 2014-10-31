v_nframes = TopNcount-1;
rank = 5;
v_score = ones(5,v_nframes);
cut = 0;    % cur==1: 表示此处为一个spotting

recog_sign_ID=[];
for i=2:v_nframes
    for cur=1:rank
        fprintf('%d--%d \n', i,cur);
        for pre = 1:rank
            if TopNindex_ID(cur,i) == TopNindex_ID(pre,i-1)
                cij = 1;
            else
                cij = 0;
            end
            if cut == 0
                s_temp(pre) = v_score(pre,i-1)*cij*TopNscore_ID(cur,i);
            else
                s_temp(pre) = 1*cij*TopNscore_ID(cur,i);
            end
        end
        v_score(cur,i) = max(s_temp);
    end
    
    if cut == 1
        cut = 0;
    end
    if sum(v_score(:,i)) == 0
        cut = 1;
        [~,I] = max(v_score(:,i-1));
        recog_sign_ID = [recog_sign_ID TopNindex_ID(I(1), i-1)];
    end
end