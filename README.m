%% Set parameters
mu1 = .01;
mu2 = 1e-3;
lambda = 0;
n_iter = 5;
n_cg_iter = 10;
nsvd = 5;

nx = 256;
ny = nx;
nt = 850;

%%
T1 = phantom(nx,ny);
T2 = T1;
T1(round(T1*100)==10) = .25; %s
T1(round(T1*100)==20) = .35; %s
T1(round(T1*100)==30) = .5; %s
T1(round(T1*100)==40) = .5; %s
T1(round(T1*100)==100) =  5; %s

T2(round(T2*100)==10) = .1; %s
T2(round(T2*100)==20) = .2; %s
T2(round(T2*100)==30) = .1; %s
T2(round(T2*100)==40) = .3; %s
T2(round(T2*100) ==100) =  2; %s

figure(1); imagesc(T1); colorbar; title('T1 (s)');
figure(2); imagesc(T2); colorbar; title('T2 (s)');


%% Build nominal Trajectory
GoldenAngle = pi/((sqrt(5.)+1)/2);
kr = pi * (-1+1/nx:1/nx:1).';
phi = (0:nt-1)*GoldenAngle;

k = [];
k(:,2,:) = kr * sin(phi);
k(:,1,:) = kr * cos(phi);

%% Load Data
mid = 602;

cd ~/mygs/20160809_IR_SE_pSSFP_MT
twx = mapVBVD(mid);
data = twx{end}.image();
data = squeeze(data);
data = data(:,:,:);
data = permute(data, [1 3 2]);

sdata = size(data);
data = reshape(data, prod(sdata(1:2)), []);
[~,~,v] = svd(data, 'econ');
data = data * v(:,1:8);
data = reshape(data, sdata(1), sdata(2), []);

% data = data(:,1:850,:);
%% Calibrate timing errors
% phi_calib = mod(phi, 2*pi);
% phi_calib = phi_calib(end/2:end);
% data_calib = data(:,end/2:end,:);
% 
% phi_calib = phi_calib(makesos(makesos(data_calib,3),1)>0);
% data_calib = data_calib(:,makesos(makesos(data_calib,3),1)>0,:);
% 
% if mod(length(phi_calib), 2) > 0
%     phi_calib = phi_calib(2:end);
% end
% [phi_calib, idx] = sort(phi_calib);
% data_calib = data_calib(:,idx,:);
% 
% idx180=size(data_calib,2)/2;
% 
% for c=size(data_calib,3):-1:1
%     dat1=fft_1d(                 squeeze((data_calib(:,1:idx180    ,c)))    ,1);
%     dat2=fft_1d(circshift(flipud(squeeze((data_calib(:,idx180+1:end,c)))),0),1);
%     for ik=idx180:-1:1
%         G_conj(:,ik,c)=squeeze(dat1(:,ik)).*conj(squeeze(dat2(:,ik)));
%     end
% end
% 
% for ik=idx180:-1:1
%     data_loc=sum(squeeze(G_conj(:,ik,:)),2);
%     G_slope(ik)=angle(sum(data_loc.*conj(circshift(data_loc,1))));
% end
% G_slope = G_slope/2;
% 
% F = @(param, phi) ((param(1) + param(2) * cos(phi)));
% param = lsqcurvefit(F, [0 0], 2*phi_calib(1:idx180), double(G_slope));
% 
% figure(5); hold all; plot(2*phi_calib(1:idx180), G_slope)
% figure(5); hold all; plot(2*phi_calib(1:idx180), F(param,2*phi_calib(1:idx180)));
% drawnow;
% 
% clear data_calib G_conj G_slope dat1 dat2 data_loc;
% 
% k = [];
% k(2,:,:) = (kr + F(param, 0)) * sin(phi);
% k(1,:,:) = (kr + F(param,pi)) * cos(phi);


%%
% data = data .* repmat(sqrt(makesos(k)), [1 1 size(data,3)]);


%% Make Dictionary
load('~/mygs/20150114_MRF_pattern_Ma_et_al/MRF_pattern.mat')
alpha = fa;
theta = alpha/2;
theta = theta(theta>.01);
alpha = theta + [pi; theta(1:end-1)];
alpha = [alpha; alpha(end:-1:2); alpha(2:end); alpha(end:-1:2)];
alpha = alpha(1:size(data,2));


TR0 = twx{end}.hdr.MeasYaps.alTR{1}*1e-6;
pSSFP = 1;

% Get rid of not acquired data
idx = find(l2_norm(data, [1, 3]) > 0);
k = k(:,:,idx);
data = double(data(:,idx,:));
data = reshape(data, [size(data,1)*size(data,2) size(data,3)]);

T1 = .3; while T1(end)<6, T1 = [T1, T1(end)*1.02]; end
T2 = .05; while T2(end)<3, T2 = [T2, T2(end)*1.02]; end
w  = 0;

D = MRF_dictionary(T1, T2, w, alpha, TR0, pSSFP, idx, nsvd);
D.plot_details{1} = 'title(''T1 (s)''); caxis([0  2]); colormap hot';
D.plot_details{2} = 'title(''T2 (s)''); caxis([0 .5]); colormap hot';
D.plot_details{3} = 'title(''PD (a.u.)''); colormap hot';

% fa_scale = [1., 0.987207, 0.949667, 0.889817, 0.811461, 0.719419, 0.619089, 0.515981, 0.415261, 0.321371, 0.237735, 0.166607, 0.10903, 0.0649297, 0.0332976, 0.0124423];
% m_dict = 0;
% for iscale = 1:length(fa_scale)
%     alpha_tmp = alpha * fa_scale(iscale);
%     alpha_tmp(1) = (alpha(1) - pi) * fa_scale(iscale) + pi;
%     [m_dict_tmp, sos_dict, T12_dict] = MRF_dictionary(T1, T2, w, alpha_tmp, TR0, pSSFP, idx);
%     m_dict = m_dict + m_dict_tmp; % .* repmat(sos_dict, [size(m_dict_tmp,1) 1]);
% end
% clear m_dict_tmp;


%% Calculate B1 maps
A = LR_nuFFT_operator(k, [nx ny 1], D.u(:,1), [], 2, [5 5]);

recon = zeros(nx,ny,size(data,2));
for c=1:size(data,2)
%     recon(:,:,c) = regularizedReconstruction(A, data(:,c), 'maxit', 20, 'verbose_flag',0);
    recon(:,:,c) = A'*(data(:,c) .* col(l2_norm(k,2)));
%     disp(c);
end
% figure; imagesc34d(l2_norm(recon,3),1); drawnow;

[calib, emaps] = bart('ecalib -r 20 -c .98', fft_2d(permute(recon, [1 2 4 3])));
smaps = squeeze(calib(:,:,:,:,1));
save(['mid', num2str(mid), '_smaps'], 'smaps');


%% Load B1
% load(['mid', num2str(mid), '_smaps']);

%% Construct nuFFT
A = LR_nuFFT_operator(k, [nx ny nsvd], D.u, smaps, 2);
% A = LR_nuFFT_operator(permute(k, [2 1 3]), [nx ny nsvd], D.u, smaps, [], [], [], makesos(k));



%% Finally Reconstruct the maps...
% WL = wavelet_operator([nx ny], 3, 'db2');
% WL = finite_difference_operator([1 2]);
[T12, PD, x] = admm_recon(A, [nx ny nsvd], data, D, n_iter, n_cg_iter, mu1, mu2, lambda, 'nuclear_norm', 1);
return;


%% Cleavland SVD Recon
x = A' * reshape(double(data), [size(data,1)*size(data,2) size(data,3)]);
x = reshape(x, [size(x,1)*size(x,2) size(x,3)]);
clear c idx
for q=size(x,1):-1:1
    [c(q,1),idx(q,1)] = max(x(q,:) * conj(m_dict), [], 2);
end
% [c,idx] = max(x * conj(m_dict), [], 2);

PD = c ./ sos_dict(idx).';
T12 = T12_dict(idx,:);
PD  = reshape(PD, [nx ny]);
T12 = reshape(T12, [nx, ny, size(T12_dict,2)]);

load('~/mygs/20151023_MRF/mask.mat')
PD = PD ./ mean(PD(:)) .* mean(mask(:));


PDint = abs(PD .* repmat(sqrt(256:-1:1).', [1 256])) .* mask;
T1int = T12(:,:,1) .* mask;
T2int = T12(:,:,2) .* mask;

figure(634);
subplot(4,2,[1,3]); imagesc(PDint(21:220,end/2-99:end/2+100), [0 30]); colorbar; axis off; axis equal; title('PD [a.u.]');
subplot(4,2,[2,4]); imagesc(T1int(21:220,end/2-99:end/2+100), [0 2.5]); colorbar; axis off; axis equal; title('T1 [s]');
subplot(4,2,[5,7]); imagesc(T2int(21:220,end/2-99:end/2+100), [0 .2]); colormap(morgenstemning(256)); colorbar; axis off; axis equal; title('T2 [s]');
