function  cov_matrix = get_cov_matrix(data)

n = size(data, 2);
M = repmat(mean(data, 2), 1, n);
C = ((data - M) * (data - M)') ./ (n - 1);
lamda = 0.001 * trace(C);
C = C + lamda * eye(size(C, 1));
% C = logm(C);
% C = (C - diag(diag(C))) * sqrt(2) + diag(diag(C));

% C = sign(C).*(abs(C).^0.9); %  transformation

% [u,s,v] = svd(C);
% cov_matrix = u(:,1:10);
cov_matrix = C;

end