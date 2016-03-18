classdef nuFFT
    properties
        numCoils = [];
        imageDim = [];
        adjoint = 0;
        trajectory_length = [];
        nufftNeighbors = [];
        sensmaps = {};
        nufftStruct = [];
        %         sparseMat = {};
    end
    
    methods
        function  A = nuFFT(trajectory, imageDim, sensmaps, os, neighbors, kernel)
            
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
                A.sensmaps{1} = 1;
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
                
                % Write coils sensitivities in the struct
                for k=1:A.numCoils
                    if size(trajectory,2) == 3      % 3D
                        A.sensmaps{k} = sensmaps(:,:,:,k);
                    else                            % 2D
                        A.sensmaps{k} = sensmaps(:,:,k);
                    end
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
            
            
            % Siemens dimensions 2 Fessler dimensions (always fun to shuffle)
            if size(trajectory,2) == 3
                trajectory = [trajectory(:,2,:), -trajectory(:,1,:) , trajectory(:,3,:)];
            else
                trajectory = [trajectory(:,2,:), -trajectory(:,1,:)];
            end
            
            % Now everything is in place and we can initialize the nuFFT. The
            % gridding kernel can be e.g. 'kaiser' or 'minmax:kb'
            for l = 1:size(trajectory, 3)                
                init = nufft_init(trajectory(:,:,l), imageDim(1:end-1), A.nufftNeighbors, round(os*imageDim(1:end-1)), ceil(imageDim(1:end-1)/2), kernel);
                if l==1
                    A.nufftStruct = init;
                end
                A.nufftStruct.p_all{l} = init.p;
            end
            
        end
        
        function s = size(A,n)
            
            t1 = [A.trajectory_length, A.numCoils, A.imageDim(end)];
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
                
                % This is the case A'*B: For each Coil nufft_adj is called with the
                % right part of the signal vector. Thereafter it is multiplied with the
                % complex conjugated sensitivity maps and all images of all coils are
                % sumed up.
                if A.adjoint
                    Q = complex(zeros(A.imageDim));
                    for j=1:A.imageDim(end)
                        A.nufftStruct.p = A.nufftStruct.p_all{j};
                        for c=1:A.numCoils
                            if length(A.imageDim) == 3
                                Q(  :,:,j) = Q(  :,:,j) + nufft_adj(B(:,c,j), A.nufftStruct) .* conj(A.sensmaps{c});
                            else
                                Q(:,:,:,j) = Q(:,:,:,j) + nufft_adj(B(:,c,j), A.nufftStruct) .* conj(A.sensmaps{c});
                            end
                        end
                    end
                    % Normalization
                    Q = Q / sqrt(prod(A.imageDim(1:end-1)));
                    
                    
                    % This is the case A*B, where B is an image that is multiplied with the
                    % coil sensitivitieA. Thereafter the nuFFT is applied
                else
                    Q = zeros(A.trajectory_length, A.numCoils, A.imageDim(end));
                    for j=1:A.imageDim(end)
                        A.nufftStruct.p = A.nufftStruct.p_all{j};
                        for c=1:A.numCoils
                            if length(A.imageDim) == 3
                                Q(:,c,j) = nufft((B(  :,:,j).*A.sensmaps{c}), A.nufftStruct);
                            else
                                Q(:,c,j) = nufft((B(:,:,:,j).*A.sensmaps{c}), A.nufftStruct);
                            end
                        end
                    end
                    Q = Q / sqrt(prod(A.imageDim(1:end-1)));
                    
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
                case '{}'
                    A.nufftStruct.p = A.nufftStruct.p_all{S.subs{:}};
                    A.nufftStruct.p_all = {};
                otherwise
                    error('Subref can only be called with the types ''()'' and ''{}''');
            end
        end
        
        function A = change_trajectory(A, n)
            A.nufftStruct.p = A.nufftStruct.p_all{n};
        end
        
    end
end
