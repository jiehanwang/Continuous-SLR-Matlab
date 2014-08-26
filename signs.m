for i=1:370
    id(i) = 0;
end

for i=1:53
    for j=1:6
        tempID = sentenceID_50(i,j);
        if tempID < 500
            id(tempID+1) = 1;
        end
    end
end