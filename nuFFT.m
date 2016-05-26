classdef nuFFT
    properties
        numCoils = [];
        imageDim = [];
        adjoint = 0;
        trajectory_length = [];
        nufftNeighbors = [];
        sensmaps = {};
        p = [];
        sn = [];
        dcomp = [];
    end
    
    methods
        function  A = nuFFT(trajectory, imageDim, sensmaps, os, neighbors, kernel, dcomp)
            
            %% Usage:
            %    A = nuFTOperator(trajectory, imageDim, sensmaps, os, neighbors, kernel)
            %
            % trajectory =  [N_samples x N_dimensions x Nt]:
            %               Arbitrary trajectory in 2D or 3D k-space (obligatory)
            %               k-space is defined in the range -pi - pi
            %
            % imageDim =    [Nx Ny Nz Nt]:
            %               Dimensions of the 4D image (obligatory)
            %
            % sensmaps =    [Nx Ny Nz Ncoils]:
            %               Coil sensitivity maps ([Nx Ny Ncoils] for 2D; optional)
            %
            % os =          Scalar: Oversampling (optional; default = 1)
            %
            % neighbors =   [x_neighbors y_neighbors] for 2D,
            %               [x_neighbors y_neighbors z_neighbors] for 3D
            %               Number of neighbors to include into interpolation;
            %               (optional; default 5 in each dimension);
            %
            % kernel =      'kaiser' for Kaiser-Bessel interpolation or
            %               'minmax:kb' for Fessler's Min-Max kernel with Kaiser-Bessel
            %               based scaling; See nufft_init for more options (optional;
            %               default = 'kaiser')
            %
            % 2010 - 2013 Jakob Asslaender: Minor changes + major documentation ;)
            
            
            % Without SENSE:
            if nargin<=2 || isempty(sensmaps)
                A.numCoils = 1;
                A.sensmaps = 1;
                % With SENSE:
            else
                % Get number of coils
                if size(trajectory,2) == 3 && length(size(sensmaps))== 4
                    A.numCoils = size(sensmaps, length(size(sensmaps)));
                end
                if size(trajectory,2) == 3 && length(size(sensmaps))== 3
                    A.numCoils = 1;
                end
                if size(trajectory,2) == 2 && length(size(sensmaps))== 3
                    A.numCoils = size(sensmaps, length(size(sensmaps)));
                end
                if size(trajectory,2) == 2 && length(size(sensmaps))== 2
                    A.numCoils = 1;
                end
                if length(imageDim) == 3
                   A.sensmaps = reshape(sensmaps, [size(sensmaps,1), size(sensmaps,2), 1, size(sensmaps,3)]);
                else
                   A.sensmaps = reshape(sensmaps, [size(sensmaps,1), size(sensmaps,2), size(sensmaps,3), 1, size(sensmaps,4)]);
                end
            end
            if nargin<=3 || isempty(os)
                os = 1;
            end
            
            A.imageDim = imageDim;
            A.trajectory_length = size(trajectory,1);
            
            % Size of neighborhood for gridding:
            if nargin < 5 || isempty(neighbors)
                if size(trajectory,2) == 3      % 3D
                    A.nufftNeighbors = [5 5 5];
                else                            % 2D
                    A.nufftNeighbors = [5 5];
                end
            else
                A.nufftNeighbors = neighbors;
            end
            
            if nargin < 6 || isempty(kernel)
                kernel = 'kaiser';
            end
            
            if nargin > 6 && ~isempty(dcomp)
                A.dcomp = sqrt(dcomp(:));
            end
            
            % Siemens dimensions 2 Fessler dimensions (always fun to shuffle)
            if size(trajectory,2) == 3
                trajectory = [trajectory(:,2,:), -trajectory(:,1,:) , trajectory(:,3,:)];
            else
                trajectory = [trajectory(:,2,:), -trajectory(:,1,:)];
            end
            
            nd = size(trajectory,1);
            np = prod(imageDim(1:end-1));
            A.p = sparse(nd*imageDim(end), prod(imageDim));
            
            % Now everything is in place and we can initialize the nuFFT. The
            % gridding kernel can be e.g. 'kaiser' or 'minmax:kb'
            mmall = [];
            kkall = [];
            uuall = [];
            for l = 1:imageDim(end)
                [init, mm, kk, uu] = nufft_init(trajectory(:,:,l), imageDim(1:end-1), A.nufftNeighbors, round(os*imageDim(1:end-1)), ceil(imageDim(1:end-1)/2), kernel);
                if l==1
                    A.sn = init.sn;
%                     A.n_shift = init.n_shift;
                end
                mmall = [mmall, mm+((l-1)*nd)];
                kkall = [kkall, kk+((l-1)*np)];
                uuall = [uuall, uu];
%                 A.p(((l-1)*nd+1):(l*nd), ((l-1)*npx+1):(l*npx)) = init.p;
            end
            
            % make sparse matrix, ensuring arguments are double for stupid matlab
            A.p = sparse(mmall, kkall, uuall, nd*imageDim(end), prod(imageDim));
            % sparse object, to better handle single precision operations!
%             A.p = Gsparse(A.p, 'odim', [nd*imageDim(end) 1], 'idim', [prod(imageDim) 1]);            
        end
        
        function s = size(A,n)
            
            t1 = [A.trajectory_length*A.imageDim(end), A.numCoils];
            t2 = A.imageDim;
            
            if A.adjoint
                tmp = t1;
                t1 = t2;
                t2 = tmp;
            end
            
            if nargin==1
                s = [prod(t1), prod(t2)];
            elseif nargin==2
                if n==1
                    s = t1;
                elseif n==2
                    s = t2;
                end
            end
        end
        
        function A = ctranspose(A)
            A.adjoint = ~A.adjoint;
        end
        
        function Q = mtimes(A,B)
            
            %% Usage:
            %    Q = mtimes(A,B)
            %
            % Either A or B must be a nuFFTOperator. If adjoint = 0, the other variable
            % must be an image of the size [Nx Ny] for 2D or [Nx Ny Nz] for 3D. If
            % adjoint = 1, the variable must be a vector of the length
            % N_timepoints x N_coils
            %
            % 2010 - 2013 Jakob Asslaender: Minor changes + major documentation ;)
            
            if isa(A,'nuFFT')
                % This is the case A'*B
                if A.adjoint
                    if ~isempty(A.dcomp)
                        B = B .* repmat(A.dcomp, [1 A.numCoils]);
                    end
                    
                    Q = complex(zeros(A.imageDim));
                    for c=1:A.numCoils
                        Xk = reshape(full(A.p' * B(:,c)), A.imageDim);
                        if length(A.imageDim) == 3
                            if A.numCoils>1
                                Q = Q +      ifft(ifft(Xk,[],1),[],2)       .* conj(A.sensmaps(  :,:,ones(A.imageDim(end),1),c));
                            else
                                Q = Q +      ifft(ifft(Xk,[],1),[],2);
                            end
                        else
                            if A.numCoils>1
                                Q = Q + ifft(ifft(ifft(Xk,[],1),[],2),[],3) .* conj(A.sensmaps(:,:,:,ones(A.imageDim(end),1),c));
                            else
                                Q = Q + ifft(ifft(ifft(Xk,[],1),[],2),[],3);
                            end
                        end
                    end    
                    snc = conj(A.sn);				% [*Nd,1]
                    if length(A.imageDim)==3
                        Q = Q .* snc(:,:,ones(1,A.imageDim(end))); % scaling factors
                    else
                        Q = Q .* snc(:,:,:,ones(1,A.imageDim(end))); % scaling factors
                    end
                                        
                % This is the case A*B, where B is an image that is multiplied with the
                % coil sensitivities. Thereafter the nuFFT is applied
                else
                    Q = complex(zeros(size(A.p,1), A.numCoils));
                    if length(A.imageDim)==3
                        B = B .* A.sn(:,:,ones(1,A.imageDim(end)));
                    else
                        B = B .* A.sn(:,:,:,ones(1,A.imageDim(end)));
                    end
                    for c=1:A.numCoils
                        if length(A.imageDim) == 3
                            if A.numCoils>1
                                tmp = fft(fft(B.*A.sensmaps(:,:,  ones(A.imageDim(end),1),c),[],1),[],2);
                            else
                                tmp = fft(fft(B,[],1),[],2);
                            end
                        else
                            if A.numCoils>1
                                tmp = fft(fft(fft(B.*A.sensmaps(:,:,:,ones(A.imageDim(end),1),c),[],1),[],2),[],3);
                            else
                                tmp = fft(fft(fft(B,[],1),[],2),[],3);
                            end
                        end
                        tmp = tmp(:);
                        Q(:,c) = A.p * tmp;
                    end
                    if ~isempty(A.dcomp)
                        Q = Q .* repmat(A.dcomp, [1 A.numCoils]);
                    end
                end
                
            % now B is the operator and A is the vector
            elseif isa(B,'nuFFT')
                Q = mtimes(B',A')';
            else
                error('nuFTOperator:mtimes', 'Neither A nor B is of class nuFTOperator');
            end
            
        end
        
        function A = subsref(A,S)
            switch S.type
                case '()'
                    A = A(S.subs{:});
                case '.'
                    A = eval(['A.', S.subs]);
                otherwise
                    error('Subref can only be called with the types ''()'' and ''.''');
            end
        end
        
        function A = change_trajectory(A, n)
            A.nufftStruct.p = A.nufftStruct.p_all{n};
        end
        
    end
end
