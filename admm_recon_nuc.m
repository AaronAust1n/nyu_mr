function [qMaps, PD, x, Dz, opt] = admm_recon_ddhx_const(E, recon_dim, data, Dic, ADMM_iter, cg_iter, mu1, mu2, lambda, P, verbose)
% Reconstructs quantitative maps from k-space data by alternately
% solving the inverse imaging problem, constraint to be close to the latest
% dictionary fit, and fitting the series of images to the dictionary.
%
% [qMaps, PD, x]    = admm_recon(E, recon_dim, data, Dic)
% [qMaps, PD, x]    = admm_recon(E, recon_dim, data, Dic, ADMM_iter)
% [qMaps, PD, x]    = admm_recon(E, recon_dim, data, Dic, ADMM_iter, cg_iter)
% [qMaps, PD, x]    = admm_recon(E, recon_dim, data, Dic, ADMM_iter, cg_iter, mu1)
% [qMaps, PD, x]    = admm_recon(E, recon_dim, data, Dic, ADMM_iter, cg_iter, mu1, mu2, lambda)
% [qMaps, PD, x]    = admm_recon(E, recon_dim, data, Dic, ADMM_iter, cg_iter, mu1, mu2, lambda, nuc_flag)
% [qMaps, PD, x]    = admm_recon(E, recon_dim, data, Dic, ADMM_iter, cg_iter, mu1, mu2, lambda, nuc_flag, P)
% [qMaps, PD, x]    = admm_recon(E, recon_dim, data, Dic, ADMM_iter, cg_iter, mu1, mu2, lambda, nuc_flag, P, verbose)
% [qMaps, PD, x, r] = admm_recon(___)
%
% Input:
%   E         =  Imaging operator (use LR_nuFFT_operator provided by this
%                toolbox. It can be used for a low rank approximation of the
%                time series, but also for a time frame by time frame
%                reconstruction.
%   recon_dim =  [nx ny (nz) R(nt)]
%                Dimensions of the series of images to be reconstructed. The
%                first 2 or 3 dimensions are the spatial ones and the last
%                is either the rank of the approximation or the number of
%                time frames, is no low rank approximation is employed
%   data      in [n_samples*nt ncoils]
%                k-space data to be reconstructed. The first dimension
%                represents the readout of all time frames concatted and
%                the second dimension is allows multi-coil data, if E
%                includes the corresponding sensitivity maps.
%   Dic       =  Dictionary struct (see MRF_dictionary.m for details)
%   ADMM_iter =  number of ADMM iterations (default = 10)
%   cg_iter   =  number of conjugate gradient iterations in each ADMM
%                iteration (default = 20)
%   mu1       =  ADMM coupling parameter (dictionary) (default = 1.26e-6,
%                but needs to be changed very likely)
%   mu2       =  ADMM coupling parameter to the spatial regularization
%                term. Has only an effect, if lambda>0 (default = .25)
%   lambda    =  Regularization parameter (default = 0, which results in no
%                spatial regularization)
%   nuc_flag  =  Swichtes between a spatial l21-penalty (=0, default) or
%                nuclear norm penalty (=1). Has only an effect if lambda>0
%   P         =  'nuclear_norm' for a nuclear norm penalty of the gradient
%                or an operator that transforms the images into the space,
%                in which an l21-norm penalty is applied. Has only an
%                effect if lambda>0. Default = 1 (penalty in the image
%                space).
%                Examples:
%                P = 'nuclear_norm';
%                P = wavelet_operator([nx ny nz], 3, 'db2');
%                P = finite_difference_operator([1 2 3]);
%   verbose   =  0 for no output, 1 for plotting the images and in each
%                iteration and give only one output per ADMM iteration in
%                the commandline and 2 for also print the residal of the
%                CG in each CG iteration.
%
%
%
% Output:
%   qMaps = Maps of quantities contained in D.lookup_table
%   PD    = Proton density retrived from the correlation
%   x     = Low rank - or time-series of images
%   r     = residual after all ADMM steps. Use only when you really want to
%           know it since it requires and additional nuFFT operation
%
% For more details, please refer to
%   J. Asslaender, M.A. Cloos, F. Knoll, D.K. Sodickson, J.Hennig and
%   R. Lattanzi, Low Rank Alternating Direction Method of Multipliers
%   Reconstruction for MR Fingerprinting  Magn. Reson. Med., epub
%   ahead of print, 2016.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Jakob Asslaender, August 2016
% New York University School of Medicine, Center for Biomedical Imaging
% University Medical Center Freiburg, Medical Physics
% jakob.asslaender@nyumc.org
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

qMaps=0;
PD=0;

if nargin < 5 || isempty(ADMM_iter)
    ADMM_iter = 10;
end
if nargin < 6 || isempty(cg_iter)
    cg_iter = 20;
end
if nargin < 7 || isempty(mu1)
    mu1 = 1.26e-3;
end
if nargin < 8 || isempty(mu2)
    mu2 = .25;
end
if nargin < 9 || isempty(lambda)
    lambda = 0;
end
if nargin < 10 || isempty(P)
    nuc_flag = 0;
    P = 1;
elseif ischar(P)
    if ~strcmp(P, 'nuclear_norm') && lambda > 0
        warning('P is a string, but not ''nuclear_norm'', we assume that is what you wanted to call');
    end
    nuc_flag = 1;
    if length(recon_dim) == 3
        P = finite_difference_operator([1 2]);
    else
        P = finite_difference_operator([1 2 3]);
    end
else
    nuc_flag = 0;
end
if nargin < 11 || isempty(verbose)
    verbose = 1;
end

% ploting stuff
persistent h1 h2 h3 h4 h5
for param = 1:size(Dic.lookup_table,2)
    eval(['persistent h', num2str(param+5)]);
end


%Algorithmic variables


    %Storage of backprojected data
    backprojection = E' * data;

    %Primal variable initialization
    f = @(x) E'*(E*x);
    x = conjugate_gradient(f,backprojection,1e-6,cg_iter,[],verbose);

    %Splitting variable: gx = \nabla x
    gx = zeros(size(P*x));

    %Splitting variable Dz = x
    Dz = zeros(size(x));

    %Dual splitting for Dz=x
    y = zeros(size(x));
    %Dual splitting für \nabla x = g
    s = zeros(size(gx));

%Tracking for objective functional and Splitting constraint
    opt.data = zeros(ADMM_iter+1,1); %Data term
    opt.reg = zeros(ADMM_iter+1,1);   %Regularization
    opt.gx_split = zeros(ADMM_iter+1,1); %Constraint gx = P*X
    opt.dz_split = zeros(ADMM_iter+1,1);  %Constraint Dz = x



%Evaluate Optimality
    if nargout > 4
        EDz = E*Dz;
        opt.data(1) = 0.5*sum( abs(EDz(:) - data(:)).^2 );
        [~,opt.reg(1)] = nuc_norm_prox_2d(P*Dz,1,1);
        
        Px = P*x;
        opt.gx_split(1) = sum(abs(gx(:)-Px(:)).^2);
        opt.dz_split(1) = sum(abs(Dz(:)-x(:)).^2);
    end

%Main Loop
for j=0:ADMM_iter

    
    %Dictionary Projection: Dz = P_D(x-y) = P_D (xmy)
        xmy = reshape(x-y, [], recon_dim(end));
        for q=size(xmy,1):-1:1
            [c(q,1),idx(q,1)] = max(xmy(q,:) * conj(Dic.magnetization), [], 2);
        end
        D = double(Dic.magnetization(:,idx)).';
        Dz = D .* repmat(sum(conj(D) .* xmy ,2), [1 recon_dim(end)]);
        Dz = reshape(Dz, recon_dim);

    %Nuc Norm prox: gx = N_prox( \nabla x + s )
        gx = nuc_norm_prox_2d(P*x+s,lambda,mu2);

    %CG Solution for x
        %Right hand side
        rhs = backprojection + mu1 * (Dz + y) + mu2*(P'*(gx - s));
        %f(x)
        f = @(x) E'*(E*x) + mu1*x + mu2*(P' * (P * x));
        %Solve
        x = conjugate_gradient(f,rhs,1e-6,cg_iter,x,verbose);
    
    %Update of dual variables
    y = y + (Dz-x);
    s = s + (P*x-gx);        
    
    if 0
        % Dynamic update of mu2 according to Boyd et al. 2011
        ss = l2_norm(1 * (P' * (G - G_old)));
        rs = l2_norm(Px - G);
        if rs > 10 * ss
            mu2 = 2*mu2;
            z = z/2;
        elseif ss > 10 * rs
            mu2 = mu2/2;
            z = 2*z;
        end
    end
    
    %Evaluate Optimality
    if nargout > 4
        EDz = E*Dz;
        opt.data(j+1) = 0.5*sum( abs(EDz(:) - data(:)).^2 );
        [~,opt.reg(j+1)] = nuc_norm_prox_2d(P*Dz,1,1);
        
        Px = P*x;
        opt.gx_split(j+1) = sum(abs(gx(:)-Px(:)).^2);
        opt.dz_split(j+1) = sum(abs(Dz(:)-x(:)).^2);
    end
    
    %Show iterations  
    if rem(j,10)==0%verbose > 0
        display(['Iteration ', num2str(j)]);
    end

end

opt.objective = opt.data + lambda*opt.reg;

