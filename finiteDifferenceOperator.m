classdef finiteDifferenceOperator
    properties
        direction = [];
        adjoint = 0;
        sum_flag = 0;
    end
    
    methods
        function  A = finiteDifferenceOperator(direction)
            A.direction = direction;
            if nargin > 1 && ~isempty(sum_flag)
                A.sum_flag = sum_flag;
            end
        end
        
        function A = ctranspose(A)
            A.adjoint = ~A.adjoint;
        end
        %
        %         function s = size(A)
        %             s = A.direction;
        %         end
        
        function Q = mtimes(A,B)
            
            if isa(A, 'finiteDifferenceOperator')
                if A.adjoint==1
                    if isvector(B)
                        Q = B - circshift(B,-1);
                    else
                        s = size(B);
                        Q = zeros(s(2:end));
                        for id = 1:length(A.direction)
                            Q = Q + squeeze(B(id,:,:,:,:)) - circshift(squeeze(B(id,:,:,:,:)),-1,A.direction(id));
                        end
                        
%                         n = zeros(1,size(B,3),size(B,4));
%                         a = [n; squeeze(B(1,:,:,:))] - [squeeze(B(1,:,:,:)); n];
%                         Q = a(2:end,:,:,:);
%                         n = zeros(size(B,3),1,size(B,4));
%                         a = [n, squeeze(B(2,:,:,:))] - [squeeze(B(2,:,:,:)), n];
%                         Q = Q + a(:,2:end,:,:);
                    end
                    
                else
                    if isvector(B)
                        Q = B - circshift(B,1);
                    else
                        Q = [];
                        for id = 1:length(A.direction)
                            Q = cat(1, Q, reshape(B - circshift(B,1,A.direction(id)), [1 size(B)]));
                        end
%                         Q = zeros([2,size(B)]);
%                         n = zeros(1,size(B,2),size(B,3));
%                         a = [B; n] - [n; B];
%                         Q(1,:,:,:) = a(1:end-1,:,:);
%                         n = zeros(size(B,2),1,size(B,3));
%                         a = [B, n] - [n, B];
%                         Q(2,:,:,:) = a(:,1:end-1,:);
                    end
                end
                % now B is the operator and A is the vector
            else
                Q = mtimes(B',A')';
                
            end
        end
    end
end