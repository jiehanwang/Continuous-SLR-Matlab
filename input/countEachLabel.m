data = importdata('sentence_meaning_ID_random_370.txt');
Data = data.data;

b=zeros(370,1);
for i=1:100
    for j=1:10
        b(Data(i,j)+1) = b(Data(i,j)+1)+1;
    end
end