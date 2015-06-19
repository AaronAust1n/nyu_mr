function [T1, PD, x] = BLIP(nuFFT, sparseMatrices, recon_dim, data, T1_dict, m_dict, n_iter, kappa, mask, x, lambda, mid, verbose)

% for display
% slices = 7:2:13;
slices = 9:12;
t = [1:2:20, size(data,2)];

if nargin < 11 || isempty(lambda)
    lambda = .25;
end 
if nargin < 10 || isempty(x)
    x = zeros(recon_dim);
end
if nargin < 9 || isempty(mask)
    mask = 1;
end
if nargin < 8 || isempty(kappa)
    kappa = 1;
end
if nargin < 7 || isempty(n_iter)
    n_iter = 100;
end

Px = x;
r = x;
Ar = r;
alpha = 0;

for j = 1:n_iter
    tic;
    % match images to data for each timeframe

%     if j>1
%         r = r - alpha * Ar;
%     end
    for l=1:recon_dim(end)
        nuFFT = change_trajectory(nuFFT, sparseMatrices{l});
%         if j==1
            r(:,:,:,l) = nuFFT' * (data(:,l) - nuFFT * x(:,:,:,l)) + lambda.^2 * (Px(:,:,:,l) - x(:,:,:,l));
%         end
        Ar(:,:,:,l) = nuFFT' * (nuFFT * r(:,:,:,l));
    end
    alpha = real(r(:)' * r(:) ./ (r(:)' * Ar(:)));
    x = x + alpha * r;
    
    display(['sos(r) = ', num2str(makesos(r(:)))]);
    if verbose
    if length(recon_dim) == 3
        sfig(4234); subplot(2,1,1); imagesc(array2mosaic(abs(x(:,:,t))));
        sfig(4234); subplot(2,1,2); imagesc(array2mosaic(angle(x(:,:,t)))); colormap jet;
    elseif length(recon_dim) == 4
        tmp = x(:,:,slices,t);
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(09568756); imagesc(angle(tmp)); colormap jet
        sfig(4234); imagesc(abs(tmp));
    else
        error('Images must be either 2D or 3D');
    end
    title(['iteration = ', num2str(j-.5), '; alpha = ', num2str(alpha)]); drawnow;
    end
    
    
    x  = reshape(x    , [prod(recon_dim(1:end-1)), recon_dim(end)]);
    Px = reshape(Px, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    
    
    phi = angle(mean(x(:,end/2+1:end), 2));
    tmp = real(x .* exp(-1i * repmat(phi, [1 size(x,2)]))) * m_dict;
%     tmp = abs(z .* exp(-1i * repmat(phi, [1 size(z,2)]))) * m_dict;
    tmp(tmp<0) = 0;
    tmp = tmp ./ repmat(makesos(m_dict,1),    [size(tmp,1) 1]);
    [~, idx] = max(tmp, [], 2);
    tmp = tmp ./ repmat(makesos(m_dict,1),    [size(tmp,1) 1]);
    for n=prod(recon_dim(1:end-1)):-1:1
        PD(n) = tmp(n,idx(n));
        Px(n,:) = PD(n) .* m_dict(:,idx(n)).' .* exp(1i * repmat(phi(n), [1 size(x,2)]));
        T1(n) = T1_dict(idx(n));
    end
    x  = reshape(x,  recon_dim);
    Px = reshape(Px, recon_dim);
    PD = reshape(PD, recon_dim(1:end-1));
    T1 = reshape(T1, recon_dim(1:end-1));
    
    if verbose
    if length(recon_dim) == 3
        sfig(234); subplot(2,1,1); imagesc(array2mosaic(real(x(:,:,t)))); title(['iteration = ', num2str(j)]);colorbar
        sfig(234); subplot(2,1,2); imagesc(array2mosaic(imag(x(:,:,t)))); title(['imag']); colorbar
        sfig(12342); subplot(2,1,1); imagesc(PD); colormap jet; colorbar
        title('PD [a.u.]');
        sfig(12342); subplot(2,1,2); imagesc(T1); colormap jet; colorbar
        title('T1 [s]');
    elseif length(recon_dim) == 4
        tmp = abs(Px(:,:,slices,t));
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(234); imagesc(tmp); title(['z_fit - iteration = ', num2str(j)]); colorbar
        tmp = abs(x(:,:,slices,t) - Px(:,:,slices,t));
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(235); imagesc(tmp); title(['(z - z_fit) - iteration = ', num2str(j)]); colorbar
        sfig(12342); subplot(2,1,1); imagesc(array2mosaic(PD(:,:,slices))); colormap jet; colorbar
        title('PD [a.u.]');
        sfig(12342); subplot(2,1,2); imagesc(array2mosaic(T1(:,:,slices)), [0 3]); colormap jet; colorbar
        title('T1 [s]');
    else
        error('Images must be either 2D or 3D');
    end
    drawnow;
    else
    save(['mid', num2str(mid), '_recon_lambda_p', num2str(lambda*100)], 'T1', 'PD', 'z', 'j');
    end
    toc
end

end