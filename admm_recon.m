function [qMaps, PD, x, r] = admm_recon(E, recon_dim, data, Dic, ADMM_iter, cg_iter, mu1, mu2, lambda, P, verbose)
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
%   J. Asslšnder, M.A. Cloos, F. Knoll, D.K. Sodickson, J.Hennig and
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
    if ~strcmp(P, 'nuclear_norm')
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


r = zeros(1,ADMM_iter+1);
for j=0:ADMM_iter
    tic;
    
    if j == 0
        backprojection = E' * data;
        f = @(x) E'*(E*x);
        x = conjugate_gradient(f,backprojection,1e-6,cg_iter,[],verbose);
        
        y = zeros(size(x));
        if lambda > 0
            Px = P * x;
            z = zeros(size(Px));
        end
    else
        b = backprojection - mu1 * y + mu1 * D .* repmat(sum(conj(D) .* y, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)]);
        
        if lambda > 0
            if nuc_flag
                G = nuc_norm_prox_2d(Px-z,lambda,mu2);
            else
                G = Px - z;
                Tl2 = l2_norm(G, length(size(G)));
                G = G - G ./ repmat(Tl2, [ones(1, length(size(G))-1) recon_dim(end)]) * lambda/mu2;
                G(isnan(G)) = 0;
                G = G .* repmat(Tl2 > lambda/mu2, [ones(1, length(size(G))-1) recon_dim(end)]);
            end
            b = b + mu2 * (P' * (G + z));
            
            f = @(x) E'*(E*x) + mu1 * (x - D .* repmat(sum(conj(D) .* x, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)])) + mu2 * (P' * (P * x));
        else
            f = @(x) E'*(E*x) + mu1 * (x - D .* repmat(sum(conj(D) .* x, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)]));
        end
        x = conjugate_gradient(f,b,1e-6,cg_iter,x,verbose);
        
        if lambda > 0
            Px = P * x;
            z = z + G - Px;
        end
    end
    x  = reshape(x, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    y  = reshape(y, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    
    clear idx
    for q=size(x,1):-1:1
        Dx = x(q,:) * Dic.magnetization;
        Dy = y(q,:) * Dic.magnetization;
        [~,idx(q,1)] = max(2*real(Dx.*conj(Dy)) + abs(Dx).^2, [], 2);
    end
    
    D = double(Dic.magnetization(:,idx)).';
    Dhx = sum(conj(D) .* x ,2);
    PD = Dhx ./ Dic.normalization(idx).';
    qMaps = Dic.lookup_table(idx,:);
    
    x   = reshape(x,    recon_dim);
    y   = reshape(y,    recon_dim);
    D   = reshape(D,    recon_dim);
    Dhx = reshape(Dhx,  recon_dim(1:end-1));
    PD  = reshape(PD,   recon_dim(1:end-1));
    qMaps = reshape(qMaps, [recon_dim(1:end-1), size(Dic.lookup_table,2)]);
    
    DDhx = D .* repmat(Dhx, [ones(1,length(recon_dim)-1) recon_dim(end)]);
    y = y + x - DDhx;
    
    % Below here is just plotting stuff...
    if verbose == 1
        % display D*Dh*x and (x-D*Dh*x)
        figure(234); imagesc34d(abs(    DDhx),0); title([      'D*Dh*x - iteration = ', num2str(j)]); colorbar; colormap gray; axis off;
        figure(235); imagesc34d(abs(x - DDhx),0); title(['(x - D*Dh*x) - iteration = ', num2str(j)]); colorbar; colormap gray; axis off;
        
        % display PD and T1
        figure(236); imagesc34d(abs(PD)); colorbar; axis off; title('PD (a.u.)');
        if isfield(Dic, 'plot_details') && length(Dic.plot_details)>size(Dic.lookup_table,2) && ~isempty(Dic.plot_details{size(Dic.lookup_table,2)+1})
            eval(Dic.plot_details{size(Dic.lookup_table,2)+1});
        end
        
        if length(size(PD))==2
            for param = 1:size(qMaps,3)
                figure(236+param); imagesc34d(qMaps(:,:  ,param)); colorbar; axis off;
                if isfield(Dic, 'plot_details') && length(Dic.plot_details)>=param && ~isempty(Dic.plot_details{param})
                    eval(Dic.plot_details{param});
                end
            end
        else
            for param = 1:size(qMaps,4)
                figure(236+param); imagesc34d(qMaps(:,:,:,param)); colorbar; axis off;
                if isfield(Dic, 'plot_details') && length(Dic.plot_details)>=param && ~isempty(Dic.plot_details{param})
                    eval(Dic.plot_details{param});
                end
            end
        end
        
        if nargout > 3
            r(1,j+1) = sum(col(abs(E*DDhx - data)).^2)/sum(col(abs(data)).^2);
            if lambda > 0
                if nuc_flag
                    [~, r_spatial] = nuc_norm_prox_2d(P * DDhx,1,1);
                else
                    r_spatial = P * DDhx;
                    r_spatial = sum(col(l2_norm(r_spatial, length(size(r_spatial)))));
                end
                r(1,j+1) = r(1,j+1) + lambda*abs(r_spatial);
            end
            figure(8346); plot(0:j, log(r(1,1:j+1)), 'o');
            xlabel('iteration'); ylabel('log(r)');
            
            
            drawnow;
        end
        if verbose > 0
            display(['Iteration ', num2str(j)]);
            toc
        end
    end
end