function [T1, PD, x] = BLIP(nuFFT, sparseMatrices, recon_dim, data, T1_dict, m_dict, n_iter, kappa, mask, x, lambda)


% for display
slices = 7:2:13;
t = [1:2:20, size(data,2)];

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
if nargin < 6 || isempty(m_dict)
    maps = ones(prod(recon_dim(1:end-1)),3);
end


r = x;
Ar = r;
alpha = 0;

for j = 1:n_iter
    tic;
    % match images to data for each timeframe

    for l=1:recon_dim(end)
        nuFFT = change_trajectory(nuFFT, sparseMatrices{l});
        r(:,:,:,l) = nuFFT' * (data(:,l) - nuFFT * x(:,:,:,l)) + lambda.^2 * x(:,:,:,l);
        Ar(:,:,:,l) = nuFFT' * (nuFFT * r(:,:,:,l));
    end
    r2 = r(:)' * r(:);
    alpha = real(r2 ./ (lambda.^2 * r2 + r(:)' * Ar(:)));
    x = x + alpha * r;
    
    
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
    
    
    x = reshape(x, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    
    % fit data to model for each voxel
    if isempty(m_dict)
        t = TR * (1:recon_dim(end));
        F = @(param, t) (param(1) - param(2) * exp(- t/param(3)));
        for n=1:prod(recon_dim(1:end-1))
            if x(n,1) ~= 0 % = mask
                param = lsqcurvefit(F, maps(n,:).', t, real(x(n,:)));
                x(n,:) = F(param, t);
                maps(n,:) = param.';
                
                
            end
        end
        
        x     = reshape(x, recon_dim);
        maps  = reshape(maps, [recon_dim(1:end-1), 3]);
        
        PD = maps(:,:,2)-maps(:,:,1);
        T1 = maps(:,:,3) .* (maps(:,:,2)./maps(:,:,1) - 1);
    else
        phi = angle(mean(x(:,end/2+1:end), 2));
        tmp = real(x .* exp(-1i * repmat(phi, [1 size(x,2)]))) * m_dict;
        tmp(tmp<0) = 0;
        
        tmp = tmp ./ repmat(makesos(m_dict,1),    [size(tmp,1) 1]);
        [~, idx] = max(tmp, [], 2);
        tmp = tmp ./ repmat(makesos(m_dict,1),    [size(tmp,1) 1]);
        
        for n=prod(recon_dim(1:end-1)):-1:1
            PD(n) = tmp(n,idx(n));
            x(n,:) = PD(n) .* m_dict(:,idx(n)).' .* exp(1i * repmat(phi(n), [1 size(x,2)]));
            T1(n) = T1_dict(idx(n));
        end
        
        x  = reshape(x,  recon_dim);
        PD = reshape(PD, recon_dim(1:end-1));
        T1 = reshape(T1, recon_dim(1:end-1));
    end
    
    if length(recon_dim) == 3
        sfig(234); subplot(2,1,1); imagesc(array2mosaic(real(x(:,:,t)))); title(['iteration = ', num2str(j)]);colorbar
        sfig(234); subplot(2,1,2); imagesc(array2mosaic(imag(x(:,:,t)))); title(['imag']); colorbar
        sfig(12342); subplot(2,1,1); imagesc(PD); colormap jet; colorbar
        title('PD [a.u.]');
        sfig(12342); subplot(2,1,2); imagesc(T1); colormap jet; colorbar
        title('T1 [s]');
    elseif length(recon_dim) == 4
        tmp = abs(x(:,:,slices,t));
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(234); imagesc(tmp); title(['iteration = ', num2str(j)]);
        sfig(12342); subplot(2,1,1); imagesc(array2mosaic(PD(:,:,slices))); colormap jet; colorbar
        title('PD [a.u.]');
        sfig(12342); subplot(2,1,2); imagesc(array2mosaic(T1(:,:,slices)), [0 3]); colormap jet; colorbar
        title('T1 [s]');
    else
        error('Images must be either 2D or 3D');
    end
    drawnow;
    toc
end

end