function dist = dtw_forSignResult(t,r)
n = size(t,2);
m = size(r,2);
% Ö¡Æ¥Åä¾àÀë¾ØÕó
d = zeros(n,m);
for i = 1:n
    for j = 1:m
%         d(i,j) = sum((t(i,:)-r(j,:)).^2);
%         d(i,j) = sqrt(sum((t(i,:)-r(j,:)).^2));
        if t(i) == r(j)
            d(i,j) = 0;
        else
            d(i,j) = 1000;
        end
    end
end
%%
% ÀÛ»ı¾àÀë¾ØÕó
D = ones(n,m) * realmax;
D(1,1) = d(1,1);
% ¶¯Ì¬¹æ»®
for i = 2:n
    for j = 1:m
        D1 = D(i-1,j);
        if j>1
            D2 = D(i-1,j-1);
        else
            D2 = realmax;
        end
        if j>2
            D3 = D(i-1,j-2);
        else
            D3 = realmax;
        end
        D(i,j) = d(i,j) + min([D1,D2,D3]);
    end
end
dist = D(n,m); 
end