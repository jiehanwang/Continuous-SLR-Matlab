clear all;
training_path_01  = '.\data\P50\';
training_path_02  = '.\data\P51\';
training_path_03  = '.\data\P52\';
training_path_04  = '.\data\P53\';
test_path = '.\data\P54\';
names = importdata('.\data\signs_97.txt');
dim = 10;

testCount = 0;
for i = 1 : length(names)
    disp(i);
    data01 = importdata([training_path_01 names{i} '.txt'], ' ', 1);
    data02 = importdata([training_path_02 names{i} '.txt'], ' ', 1);
    data03 = importdata([training_path_03 names{i} '.txt'], ' ', 1);
    data04 = importdata([training_path_04 names{i} '.txt'], ' ', 1);
    data = importdata([test_path names{i} '.txt'], ' ', 1);
    
    %--------------------------------------
    %Training data
    [h, w] = size(data01.data);
    for j=1:h
           trainLabels{1,4*i-3}(j) = str2num(names{i}(2:5));
        for k=1:dim
           trainSeqs{1,4*i-3}(k,j) = data01.data(j,k);
        end
    end
    
    [h, w] = size(data02.data);
     for j=1:h
           trainLabels{1,4*i-2}(j) = str2num(names{i}(2:5));
        for k=1:dim
           trainSeqs{1,4*i-2}(k,j) = data02.data(j,k);
        end
     end
     
     [h, w] = size(data03.data);
     for j=1:h
           trainLabels{1,4*i-1}(j) = str2num(names{i}(2:5));
        for k=1:dim
           trainSeqs{1,4*i-1}(k,j) = data03.data(j,k);
        end
     end
     
     [h, w] = size(data04.data);
     for j=1:h
           trainLabels{1,4*i}(j) = str2num(names{i}(2:5));
        for k=1:dim
           trainSeqs{1,4*i}(k,j) = data04.data(j,k);
        end
     end
    %-------------------------------------
    % Testing data
    [h, w] = size(data.data);
     for j=1:h
           testLabels{1,1}(testCount + j) = str2num(names{i}(2:5));
        for k=1:dim
           testSeqs{1,1}(k,testCount + j) = data.data(j,k);
        end
     end
     testCount = testCount + h;
end