function [qMaps, PD, x] = admm_recon_bart(k, imageDim, data, Dic, bart_path, n_iter, lambda, smaps)
% Reconstructs quantitative maps from k-space data by alternately
% solving the inverse imaging problem, constraint to be close to the latest
% dictionary fit, and fitting the series of images to the dictionary. 
%
% NOTE: You must have write access in the directory where BART is
% installed.
%
% [qMaps, PD, x]    = admm_recon(k, imageDim, data, Dic)
% [qMaps, PD, x]    = admm_recon(k, imageDim, data, Dic, bart_path)
% [qMaps, PD, x]    = admm_recon(k, imageDim, data, Dic, bart_path, n_iter)
% [qMaps, PD, x]    = admm_recon(k, imageDim, data, Dic, bart_path, n_iter, lambda)
% [qMaps, PD, x]    = admm_recon(k, imageDim, data, Dic, bart_path, n_iter, lambda, smaps)
% [qMaps, PD, x]    = admm_recon(k, imageDim, data, Dic, bart_path, n_iter, lambda, smaps)
%
% Input:
%   k         =  [N_samples 2(3) Nt] (obligatory)
%                Arbitrary trajectory in 2D k-space N_samples is
%                the lenth of the trajectory for each time frame, 2
%                depends on 2D imaging and Nt is the number of
%                time frames to be transformed.
%                k-space is defined in the range -pi - pi   
%   data      =  [n_samples*nt ncoils] (obligatory)
%                k-space data to be reconstructed. The first dimension
%                represents the readout of all time frames concatted and
%                the second dimension is allows multi-coil data.
%   Dic       =  Dictionary struct (see MRF_dictionary.m for details)
%                (obligatory)
%   bart_path =  Directory of BART toolbox installation. (optional)
%   n_iter    =  Number of ADMM iterations (default = 100) (optional)
%   lambda    =  Regularization parameter (default = 0, which results in no
%                spatial regularization). Range ~ 0 - 0.005 (optional)
%   smaps     =  [Nx Ny (Nz) Ncoils] (optional)
%                Coil sensitivity maps.
%
%
%
% Output:
%   qMaps = Maps of quantities contained in D.lookup_table
%   PD    = Proton density retrived from the correlation
%   x     = Low rank - or time-series of images
%
% For more details, please refer to
%   J. Asslaender, M.A. Cloos, F. Knoll, D.K. Sodickson, J.Hennig and
%   R. Lattanzi, Low Rank Alternating Direction Method of Multipliers
%   Reconstruction for MR Fingerprinting  Magn. Reson. Med., epub
%   ahead of print, 2016.
%
%   Martin Uecker, Frank Ong, Jonathan I Tamir, Dara Bahri,
%   Patrick Virtue, Joseph Y Cheng, Tao Zhang, and Michael Lustig.
%   Berkeley Advanced Reconstruction Toolbox. Annual Meeting ISMRM,
%   Toronto 2015, In Proc. Intl. Soc. Mag. Reson. Med. 23:2486 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% T. Bruijnen, July 2019
% Department of Radiotherapy, University Medical Center Utrecht, 
%   the Netherlands
% Computational Imaging Group for MRI diagnostics and therapy, Centre for
%   Image Sciences, University Medical Center Utrecht, The Netherlands
% T.Bruijnen@umcutrecht.nl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Manage the input and set paths
if nargin < 8  || isempty(smaps)
    smaps = ones([imageDim 1 size(k,4)],'single');
end
if nargin < 7 || isempty(lambda)
    lambda = 0;
end
if nargin < 6 || isempty(n_iter)
    n_iter = 1000;
end 
if nargin < 5 || isempty(bart_path)
    try
        bart_path = which('bart');
        bart_path = bart_path(1:end-13); % Remove matlab folder
    catch
        disp(['Cant find bart directory.'])
    end
end

% Environmental variables
setenv([bart_path,'bart']);
addpath(genpath(bart_path));

%% Preprocess the data and setup correct dimensions
% Apply the sqrt(2) filter to get corner of Cartesian k-space
% All points outside the square are set to zero
% This effectively reduces the oversampling ratio with sqrt(2)
for ns = 1 : size(k,3)
    % Edge 1
    idx = [];
    nx = 1;
    while abs(k(nx,1,ns)) > pi || abs(k(nx,2,ns)) > pi
        idx = [idx nx];
        nx = nx + 1;
    end
    if ~isempty(idx)
        k(idx,:,ns) = repmat(k(idx(end),:,ns), [numel(idx) 1]);
        data((ns-1) * size(k,1) + idx) = 0; 
    end
    
    % Edge 2
    idx = [];
    nx = 1;
    while abs(k(end-nx+1,1,ns)) > pi || abs(k(end-nx+1,2,ns)) > pi
        idx = [idx size(k,1) - nx + 1];
        nx = nx + 1;
    end
    if ~isempty(idx)
        k(idx,:,ns) = repmat(k(idx(end),:,ns), [numel(idx) 1]);
        data((ns-1) * size(k,1) + idx) = 0;
    end    
end

% Set dimensions for BART
k        = permute(0.5 * imageDim(1) * k / max(abs(k(:))), [2 1 4 5 6 3]);
k(1:2,:) = k([2 1],:);
k(2,:)   = -k(2,:);
k(3,:)   = 0; % For 3D you need to change this
data     = reshape(data,1,[],1,1,1,size(Dic.u,1));
u        = permute(Dic.u, [3 4 5 6 7 1 2]);

%% Write temporary files and do BART reconstruction for singular images
% Write data-files for BART
writecfl([bart_path,'smaps'],smaps);
writecfl([bart_path,'u'],u);
writecfl([bart_path,'k'],k)
writecfl([bart_path,'data'],data);

% Define BART call
pics_call = [bart_path,'bart pics -d5 -l1 -R T:3:0:',num2str(lambda),' -i',num2str(n_iter),' -t ',...
        bart_path,'k',' ','-B ',bart_path,'u',' ',bart_path,'data',' ',bart_path,'smaps',' ',bart_path,'x'];

% System call to BART
system(pics_call);
x = readcfl([bart_path,'x']);
x = reshape(x, [], size(Dic.u,2));

%% Dictionary matching:
clear c idx
for q=size(x,1):-1:1
    [c(q,1),idx(q,1)] = max(x(q,:) * conj(Dic.magnetization), [], 2);
end
PD    = c ./ Dic.normalization(idx).';
PD    = reshape(PD, [imageDim(1) imageDim(2)]);
qMaps = Dic.lookup_table(idx,:);
qMaps = reshape(qMaps, [imageDim(1), imageDim(2), size(Dic.lookup_table,2)]);

%% Filter background components based on singular images
mask = sum(abs(reshape(x,size(PD,1),size(PD,2),size(Dic.u,2))),3);
mask(mask > .1 * max(mask(:))) = 1;
mask(mask < .1 * max(mask(:))) = 0;
qMaps = bsxfun(@times,qMaps,mask);
PD = bsxfun(@times,PD,mask);

% END
end