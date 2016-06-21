classdef wavelet_operator
    properties
        space = 0;
        params = [];
        dim = [];
        sizes_all = [];
        N = [];
        wname = '';
        adjoint = 0;
    end
    
    methods
        function W = wavelet_operator(dim, N, wname)
            % function W = waveletDecompositionOperator(dim, N, wname)
            %
            % dim = dimension of image (e.g. [64 64 32])
            % N = order of decomposition
            % wname = name of wavelet
            
            if nargin==0
                W.space = 0;
            else
                if length(dim)==1
                    W.space = 1;
                elseif length(dim)==2
                    if dim(1)==1 || dim(2)==1
                        W.space = 1;
                    else
                        W.space = 2;
                    end
                elseif length(dim)==3
                    W.space = 3;
                end
            end
            
            if W.space==0
                W.params.sizes = [];
                W.params.sizeINI = [];
                W.params.level = [];
                W.params.mode = [];
                W.params.filters = [];
                W.dim = [];
                W.space = [];
                W.sizes_all = [];
            elseif W.space==1
                [~,sizes] = wavedec(zeros(dim,1),N,wname);
                W.params.sizes = sizes;
                W.params.sizeINI = [];
                W.params.level = [];
                W.params.mode = [];
                W.params.filters = [];
                W.dim = [dim 1];
                W.space = 1;
                W.sizes_all = [];
            elseif W.space==2
                [~,sizes] = wavedec2(zeros(dim),N,wname);
                W.params.sizes = sizes;
                W.params.sizeINI = [];
                W.params.level = [];
                W.params.mode = [];
                W.params.filters = [];
                W.dim = dim;
                W.space = 2;
                W.sizes_all = [];
            elseif W.space==3
                W.params = wavedec3(zeros(dim),N,wname);
                W.dim = dim;
                W.space = 3;
                sizes_all = zeros(length(W.params.dec),3);
                for k=1:length(W.params.dec)
                    sizes_all(k,:) = size(W.params.dec{k});
                end
                W.sizes_all = sizes_all;
            else
                error('dimension is not specified properly.');
            end
            
            if nargin==0
                W.N = [];
                W.wname = '';
                W.adjoint = 0;
            else
                W.N = N;
                W.wname = wname;
                W.adjoint = 0;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function B = ctranspose(A)
            B = A;
            if B.adjoint==0
                B.adjoint = 1;
            else
                B.adjoint = 0;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function Q = mtimes(A,B)
            
            if strcmp(class(A),'wavelet_operator')
                
                if A.adjoint==1
                    if A.space==1
                        Q = conj(waverec(B,A.params.sizes,A.wname));
                    elseif A.space==2
                        for i=size(B,2):-1:1
                            Q(:,:,i) = conj(waverec2(B(:,i),A.params.sizes,A.wname));
                        end
                    elseif A.space==3
                        X.sizeINI = A.params.sizeINI;
                        X.level = A.params.level;
                        X.mode = A.params.mode;
                        X.filters = A.params.filters;
                        X.sizes = A.params.sizes;
                        ps = prod(A.sizes_all,2);
                        Y = cell(1,length(ps));
                        for i=size(B,2):-1:1
                            startpt = 1;
                            for k=1:length(ps)
                                Y{k} = reshape(B(startpt:startpt+ps(k)-1,i),A.sizes_all(k,:));
                                startpt = startpt + ps(k);
                            end
                            X.dec = Y;
                            Q(:,:,:,i) = waverec3(X);
                        end
                    end
                else
                    if A.space==1
                        Q = wavedecX(B,A.N,A.wname);
                        
                    elseif A.space==2
                        for i=size(B,3):-1:1
                            Q(:,i) = wavedec2(B(:,:,i),A.N,A.wname).';
                        end
                    elseif A.space==3
                        for i=size(B,4):-1:1
                            X = wavedec3(B(:,:,:,i),A.params.level,A.wname);
                            X = X.dec;
                            if i == size(B,4)
                                ps = prod(A.sizes_all,2);
                                Q = zeros(sum(ps),size(B,4));
                            end
                            startpt = 1;
                            for k=1:length(X)
                                Q(startpt:startpt+ps(k)-1,i) = col(X{k});
                                startpt = startpt + ps(k);
                            end
                        end
                    end 
                end
            % now B is the operator and A is the vector
            else
                Q = conj(mtimes(B',conj(A)));
                
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function s = size(W,n)
            if nargin < 2
                s = W.dim;
            else
                s = W.dim(n);
            end
        end
    end
end