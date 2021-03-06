
fileName = 'sentence_meaning_ID_random_100.txt';
names = importdata('sign_100_7g.txt');

fid = fopen(fileName,'wt');
fprintf(fid, 'NUMBER	100 \n');
temp = randint(100,10,[0 1034]);


for i=1:100
    names_new(i) = str2double(names{i}(2:5));
end

for i=1:100
    for j=1:10
        recordNum = temp(i,j);
        while ~ismember(recordNum, names_new);
            recordNum = randint(1,1,[0 1034]);
        end
        
        if j<10
            fprintf(fid, '%d ', recordNum);
        else
            fprintf(fid, '%d\n', recordNum);
        end
    end
end
fclose(fid);