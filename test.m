clear all;
load b-61;
load score_61;

for i=1:216
    for j=1:29
        b_norm_fl(i,j) = (1-b_norm(i,j)^2)^0.5;
    end
end

x=[1:29];
se=[4,6,18,75,85,99];
b_norm_fl_se = b_norm_fl(se,:);
plot(x, b_norm_fl_se);
% plot(x, score_all);