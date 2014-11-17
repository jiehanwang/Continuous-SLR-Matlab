function outKernel = kernel(SY1,SY2,testID)

if (nargin<1)
    error('Not enough inputs');
end

number_sets1 = length(SY1);

if (isempty(SY2)~=1)
    number_sets2 = length(SY2);
    trainFlag = 0;    %1: traing. 0:test
else
    SY2 = SY1;
    number_sets2 = length(SY2);
     trainFlag = 1;
end

outKernel = zeros(number_sets1,number_sets2,1);
lamda = 0.5;
%%
for tmpC1 = 1:number_sets1
    if trainFlag == 0
        fprintf('Test kernel--%d------%d/%d\n', testID, tmpC1,number_sets1);
    else
        fprintf('Training kernel--%d------%d/%d\n', testID, tmpC1,number_sets1);
    end
    Y1 = SY1{tmpC1};
    for tmpC2 = 1:number_sets2
        Y2 = SY2{tmpC2};
        if(isempty(Y1)~=1 && isempty(Y2)~=1)
            %CA
%             CA1_GS = Gram_Schmidt(Y1.C*Y1.A);
%             CA2_GS = Gram_Schmidt(Y2.C*Y2.A);
%             tmpMatrix = CA1_GS'*CA2_GS;
%             outKernel(tmpC1,tmpC2) = sum(sum(tmpMatrix.^2));
            
            % B
%             [A1_orth,~,~] = svd(Y1.A);
%             [A2_orth,~,~] = svd(Y2.A);
%             tmpMatrix = Y1.B'*Y2.B;
%            
%             tmpMatrix = Y1.S'*Y2.S;
%             
%             outKernel(tmpC1,tmpC2) = sum(sum(tmpMatrix.^2));
            
            % CTCA
%             n = size(Y1.A,1);
%             O1 = Y1.C;
%             O2 = Y2.C;
%             for i = 1 : n
%                 O1 = [O1; Y1.C*Y1.A^i];
%                 O2 = [O2; Y2.C*Y2.A^i];
%             end
%             O1_GS = Gram_Schmidt(O1);
%             O2_GS = Gram_Schmidt(O2);
%             
%             tmpMatrix = O1_GS'*O2_GS;
%             outKernel(tmpC1,tmpC2) = sum(sum(tmpMatrix.^2));

            % CTC
            tmpMatrix = Y1.C'*Y2.C;
            outKernel(tmpC1,tmpC2) = sum(sum(tmpMatrix.^2));

            % Original
%             outKernel(tmpC1,tmpC2) = exp(-lamda*manifold_dis(Y1,Y2)^2);
        else
            outKernel(tmpC1,tmpC2) = 0;
        end
    end
end