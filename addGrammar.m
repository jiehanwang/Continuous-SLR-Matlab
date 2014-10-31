function sign_recognized_ID_Final = addGrammar(sign_recognized_ID)
load data\languageModel;

recogN = size(sign_recognized_ID,2);
connectRe = zeros(recogN, recogN);
for row = 1:recogN
    for col = (row+1):recogN
        connectRe(row, col) = C_a(sign_recognized_ID(col)+1, sign_recognized_ID(row)+1);  
        % C_a(a,b): b在a之前的概率
    end
end

recogN_Final = 1;
for i=1:recogN
    if sum(connectRe(i,:)) == 0 && sum(connectRe(:,i)) == 0
    elseif recogN_Final>1 && (sign_recognized_ID_Final(recogN_Final-1) == sign_recognized_ID(i))
    else
        sign_recognized_ID_Final(recogN_Final) = sign_recognized_ID(i);
        recogN_Final = recogN_Final+1;
    end
end 
recogN_Final = recogN_Final - 1;


end