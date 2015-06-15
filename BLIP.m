function [T1, PD, z] = BLIP(nuFFT, sparseMatrices, recon_dim, data, T1_dict, m_dict, n_iter, kappa, mask, z, lambda, mid, verbose)

% for display
% slices = 7:2:13;
slices = 9:12;
t = [1:2:20, size(data,2)];

if nargin < 11 || isempty(lambda)
    lambda = .25;
end 
if nargin < 10 || isempty(z)
    z = zeros(recon_dim);
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

z_data = 0*data;
z_old  = z;
z_fit = z;

for j = 1:n_iter
    tic;
    % match images to data for each timeframe
    
    z_data_old = z_data;
    
    if max(abs(z(:))) > 0 % saves comp time if the initial z is 0
        for l=1:recon_dim(end)
            nuFFT = change_trajectory(nuFFT, sparseMatrices{l});
            if length(recon_dim) == 3
                z_data(:,l) = nuFFT * z(:,:,l);
            elseif length(recon_dim) == 4
                z_data(:,l) = nuFFT * z(:,:,:,l);
            else
                error('Images must be either 2D or 3D');
            end
        end
    end
    
    omega = makesos(z(:) - z_old(:)) ./ makesos(z_data(:) - z_data_old(:));
    display(['omega = ', num2str(omega)]);
%     mu = max(omega*kappa, 1);
    if isnan(omega)
        mu = 1;
    else
        mu = omega*kappa;
    end
    z_old = z;
    
    data_error = 0;
    for l=1:recon_dim(end)
        tmp = data(:,l) - z_data(:,l);
        data_error = data_error + makesos(tmp).^2;
        
        nuFFT = change_trajectory(nuFFT, sparseMatrices{l});
        if length(recon_dim) == 3
            z(  :,:,l) = mask .* (z(  :,:,l) .* (1 - mu .* lambda.^2) + mu * (nuFFT' * tmp + lambda.^2 .* z_fit(  :,:,l)));
        elseif length(recon_dim) == 4
            z(:,:,:,l) = mask .* (z(:,:,:,l) .* (1 - mu .* lambda.^2) + mu * (nuFFT' * tmp + lambda.^2 .* z_fit(:,:,:,l)));
        else
            error('Images must be either 2D or 3D');
        end
    end
    display(['Data error = ', num2str(sqrt(data_error)./makesos(data(:)))]);
    
    if verbose
    if length(recon_dim) == 3
        sfig(4234); subplot(2,1,1); imagesc(array2mosaic(abs(z(:,:,t))));
        sfig(4234); subplot(2,1,2); imagesc(array2mosaic(angle(z(:,:,t)))); colormap jet;
    elseif length(recon_dim) == 4
        tmp = z(:,:,slices,t);
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(09568756); imagesc(angle(tmp)); colormap jet
        sfig(4234); imagesc(abs(tmp));
    else
        error('Images must be either 2D or 3D');
    end
    title(['iteration = ', num2str(j-.5), '; mu = ', num2str(mu)]); drawnow;
    end
    
    
    z     = reshape(z    , [prod(recon_dim(1:end-1)), recon_dim(end)]);
    z_fit = reshape(z_fit, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    
    
    phi = angle(mean(z(:,end/2+1:end), 2));
    tmp = real(z .* exp(-1i * repmat(phi, [1 size(z,2)]))) * m_dict;
%     tmp = abs(z .* exp(-1i * repmat(phi, [1 size(z,2)]))) * m_dict;
    tmp(tmp<0) = 0;
    tmp = tmp ./ repmat(makesos(m_dict,1),    [size(tmp,1) 1]);
    [~, idx] = max(tmp, [], 2);
    tmp = tmp ./ repmat(makesos(m_dict,1),    [size(tmp,1) 1]);
    for n=prod(recon_dim(1:end-1)):-1:1
        PD(n) = tmp(n,idx(n));
        z_fit(n,:) = PD(n) .* m_dict(:,idx(n)).' .* exp(1i * repmat(phi(n), [1 size(z,2)]));
        T1(n) = T1_dict(idx(n));
    end
    z     = reshape(z,     recon_dim);
    z_fit = reshape(z_fit, recon_dim);
    PD = reshape(PD, recon_dim(1:end-1));
    T1 = reshape(T1, recon_dim(1:end-1));
    
    if verbose
    if length(recon_dim) == 3
        sfig(234); subplot(2,1,1); imagesc(array2mosaic(real(z(:,:,t)))); title(['iteration = ', num2str(j)]);colorbar
        sfig(234); subplot(2,1,2); imagesc(array2mosaic(imag(z(:,:,t)))); title(['imag']); colorbar
        sfig(12342); subplot(2,1,1); imagesc(PD); colormap jet; colorbar
        title('PD [a.u.]');
        sfig(12342); subplot(2,1,2); imagesc(T1); colormap jet; colorbar
        title('T1 [s]');
    elseif length(recon_dim) == 4
        tmp = abs(z_fit(:,:,slices,t));
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(234); imagesc(tmp); title(['z_fit - iteration = ', num2str(j)]); colorbar
        tmp = abs(z(:,:,slices,t) - z_fit(:,:,slices,t));
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