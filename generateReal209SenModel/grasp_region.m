function grasp = grasp_region(t, t_, P, Q, n)
if t_>t 
    C = (1/(t_-t))*((Q{1,t_}-Q{1,t})-(1/(t_-t+1))...  
        *((P{1,t_}-P{1,t})*(P{1,t_}-P{1,t})'));
else   % 防止只有一帧的情况出现
    
    C = zeros(size(Q{1,t_},2), size(Q{1,t_},2));
end

lamda = 0.001 * trace(C);
C = C + lamda * eye(size(C, 1));   

    % SVD Cov，得到前5维
[u,~,~] = svd(C);  %[u,s,v]
grasp = u(:,1:n);
end