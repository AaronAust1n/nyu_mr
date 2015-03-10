function [T1, PD, z] = BLIP(nuFFT, sparseMatrices, recon_dim, data, T1_dict, m_dict, n_iter, kappa, mask, z)


% for display
slices = 7:2:13;
t = [1:2:20, size(data,2)];

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
if nargin < 6 || isempty(m_dict)
    maps = ones(prod(recon_dim(1:end-1)),3);
end

z_data = 0*data;
z_old  = z;

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
    mu = max(omega*kappa, 1);
    z_old = z;
    
    data_error = 0;
    for l=1:recon_dim(end)
        tmp = data(:,l) - z_data(:,l);
        data_error = data_error + makesos(tmp).^2;
        
        nuFFT = change_trajectory(nuFFT, sparseMatrices{l});
        if length(recon_dim) == 3
            z(:,:,l) = mask .* (z(:,:,l) + mu * (nuFFT' * tmp));
        elseif length(recon_dim) == 4
            z(:,:,:,l) = mask .* (z(:,:,:,l) + mu * (nuFFT' * tmp));
        else
            error('Images must be either 2D or 3D');
        end
    end
    display(['Data error = ', num2str(sqrt(data_error)./makesos(data(:)))]);
    
    
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
    
    
    z = reshape(z, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    
    % fit data to model for each voxel
    if isempty(m_dict)
        t = TR * (1:recon_dim(end));
        F = @(param, t) (param(1) - param(2) * exp(- t/param(3)));
        for n=1:prod(recon_dim(1:end-1))
            if z(n,1) ~= 0 % = mask
                param = lsqcurvefit(F, maps(n,:).', t, real(z(n,:)));
                z(n,:) = F(param, t);
                maps(n,:) = param.';
                
                
            end
        end
        
        z     = reshape(z, recon_dim);
        maps  = reshape(maps, [recon_dim(1:end-1), 3]);
        
        PD = maps(:,:,2)-maps(:,:,1);
        T1 = maps(:,:,3) .* (maps(:,:,2)./maps(:,:,1) - 1);
    else
%         tmp = abs(z);
%         tmp = tmp .* sign(real(z));
%         tmp = tmp * m_dict;
        tmp = real(z) * m_dict;
        tmp(tmp<0) = 0;
        tmp = tmp ./ repmat(makesos(m_dict,1),    [size(tmp,1) 1]);
        [~, idx] = max(tmp, [], 2);
        tmp = tmp ./ repmat(makesos(m_dict,1),    [size(tmp,1) 1]);
        for n=prod(recon_dim(1:end-1)):-1:1
            PD(n) = tmp(n,idx(n));
            z(n,:) = PD(n) .* m_dict(:,idx(n)).'; % + 1i * imag(z(n,:));
%             z(n,:) = abs(PD(n) .* m_dict(:,idx(n)).') .* exp(1i * angle(z(n,:)));
            T1(n) = T1_dict(idx(n));
        end
        z  = reshape(z,  recon_dim);
        PD = reshape(PD, recon_dim(1:end-1));
        T1 = reshape(T1, recon_dim(1:end-1));
    end
    
    if length(recon_dim) == 3
        sfig(234); subplot(2,1,1); imagesc(array2mosaic(real(z(:,:,t)))); title(['iteration = ', num2str(j)]);colorbar
        sfig(234); subplot(2,1,2); imagesc(array2mosaic(imag(z(:,:,t)))); title(['imag']); colorbar
        sfig(12342); subplot(2,1,1); imagesc(PD); colormap jet; colorbar
        title('PD [a.u.]');
        sfig(12342); subplot(2,1,2); imagesc(T1); colormap jet; colorbar
        title('T1 [s]');
    elseif length(recon_dim) == 4
        tmp = abs(z(:,:,slices,t));
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