function score = dec_values_score(dec_values, classNum)
% SVM结果的概率估计方法来自 "http://blog.csdn.net/zhzhl202/article/details/7438313"
% 这也是一种比较常用的方法

% length = size(dec_values,2);
signNumber = classNum;

pointer = 1;
dec_matrix = zeros(signNumber,signNumber);
for i=1:signNumber
    for j=i+1:signNumber
        dec_matrix(i,j) = dec_values(pointer);
        pointer = pointer + 1;
    end
end

dec_matrix = dec_matrix + (-dec_matrix');

dec_sum = zeros(1,signNumber);
score = zeros(1,signNumber);
for j=1:signNumber
    for i=1:signNumber
        if dec_matrix(i,j)<0
            dec_sum(j) = dec_sum(j) + 1;
            score(j) = score(j) + abs(dec_matrix(i,j));
        end
    end
    score(j) = score(j)/(2*dec_sum(j)+0.0000001)+dec_sum(j)/(2*signNumber);
end