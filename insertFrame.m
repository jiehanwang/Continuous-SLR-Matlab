function [data] = insertFrame(data_Ori, n)

frameN = size(data_Ori,1);
if frameN<n
    rate = frameN/n;
    for k=1:n
        oriFrame = floor(k*rate);
        if oriFrame < 1
            oriFrame = 1;
        end
        data(k,:) = data_Ori(oriFrame,:);
    end
else
    data = data_Ori;
end