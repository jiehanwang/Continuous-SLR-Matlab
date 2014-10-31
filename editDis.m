function [distance, insert, delete, substitute, correct] = editDis(t,r)
m = size(t,2);
n = size(r,2);
insert = 0;   % 0
delete = 0;   % 1
substitute = 0;  % 2
correct = 0;         % 3

% 累积距离矩阵
dis = ones(m+1,n+1) * realmax;
route = ones(m+1,n+1,3);
for i=1:m+1
    dis(i,1) = i-1;
end
for j=1:n+1
    dis(1,j) = j-1;
end

for i=2:m+1
    for j=2:n+1
        if t(i-1) == r(j-1)
            tij = 0;
        else
            tij = 1;
        end
        dis(i,j) = min(min(dis(i-1,j)+1,dis(i,j-1)+1), dis(i-1,j-1)+tij);
        if dis(i,j) == dis(i-1,j)+1
            route(i,j,1) = i-1;
            route(i,j,2) = j;
            route(i,j,3) = 1;
        elseif dis(i,j) == dis(i,j-1)+1
            route(i,j,1) = i;
            route(i,j,2) = j-1;
            route(i,j,3) = 0;
        elseif dis(i,j) == dis(i-1,j-1)+tij
            route(i,j,1) = i-1;
            route(i,j,2) = j-1;
            if tij == 0
                route(i,j,3) = 3;
            else
                route(i,j,3) = 2;
            end
        end
    end 
end




i = m+1;
j = n+1;
while i>=1 && j>=1
    if route(i,j,3) == 3
        correct = correct +1;
    elseif route(i,j,3) == 2
        substitute = substitute+1;
    elseif route(i,j,3) == 1
        delete = delete+1;
    elseif route(i,j,3) == 0
        insert = insert +1;
    end

    if i==1 && j==1
        delete = delete - 1;    %回溯到最开始的位置，因为初始化是1,1指代delete，会多计算一次。故减去。
        break;
    elseif i==1 || j==1
        break;
    else
        temp_i = i;
        temp_j = j;
        i = route(temp_i,temp_j,1);
        j = route(temp_i,temp_j,2);
    end
end

distance = dis(m+1,n+1);

end
