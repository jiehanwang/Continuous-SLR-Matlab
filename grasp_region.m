function grasp = grasp_region(t, t_, P, Q, n)

C = (1/(t_-t))*((Q{1,t_}-Q{1,t})-(1/(t_-t+1))...  
        *((P{1,t_}-P{1,t})*(P{1,t_}-P{1,t})'));
lamda = 0.001 * trace(C);
C = C + lamda * eye(size(C, 1));   

    % SVD Cov£¬µÃµ½Ç°5Î¬
[u,~,~] = svd(C);  %[u,s,v]
grasp = u(:,1:n);