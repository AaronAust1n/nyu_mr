function [T1, PD, z] = BLIP(nuFFT, recon_dim, data, TR, T1_dict, m_dict, mask, z)


n_iter = 100;
kappa = 2;

% for display
if recon_dim(3) < 51
    slices = 5:2:11;
else
    slices = 11:10:51;
end

if nargin < 8 || isempty(z)
    z = zeros(recon_dim);
end

if isempty(m_dict)
    maps = ones(prod(recon_dim(1:end-1)),3);
end

z_data = 0*data;
z_old  = z;

for j = 1:n_iter
    % match images to data for each timeframe
    
    z_data_old = z_data;
    
    parfor l=1:recon_dim(end)
        nuFFT_l = change_trajectory(nuFFT, l);
%         if length(recon_dim) == 3
%             z_data(:,l) = nuFFT_l * z(:,:,l);
        if length(recon_dim) == 4
            z_data(:,l) = nuFFT_l * z(:,:,:,l);
        else
            error('Images must be either 2D or 3D');
        end
    end
    
    omega = makesos(z(:) - z_old(:)) ./ makesos(z_data(:) - z_data_old(:));
    display(['omega = ', num2str(omega)]);
    mu = max(omega*kappa, 1);
    z_old = z;
    
    data_error = 0;
    parfor l=1:recon_dim(end)
        tmp = data(:,l) - z_data(:,l);
        data_error = data_error + makesos(tmp);
        
        nuFFT_l = change_trajectory(nuFFT, l);
%         if length(recon_dim) == 3
%             z(:,:,l) = mask .* (z(:,:,l) + mu * (nuFFT_l' * tmp));
        if length(recon_dim) == 4
%             tmp = z(:,:,:,l);
            z(:,:,:,l) = mask .* (z(:,:,:,l) + mu * (nuFFT_l' * tmp));
        else
            error('Images must be either 2D or 3D');
        end
    end
    display(['Data error = ', num2str(data_error)]);
    
    
    if length(recon_dim) == 3
        figure(4234); imagesc(array2mosaic(abs(z(:,:,1:100:720))))
    elseif length(recon_dim) == 4
        tmp = real(z(:,:,slices,1:100:720));
        tmp = array2mosaic(tmp(:,:,:), [8 length(slices)]);
        figure(4234); imagesc(tmp); colorbar;
        title('real');
        tmp = imag(z(:,:,slices,1:100:720));
        tmp = array2mosaic(tmp(:,:,:), [8 length(slices)]);
        figure(4234234); imagesc(tmp); colorbar;
        title('imag');
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
        tmp = real(z) * m_dict;
        tmp = tmp ./ repmat(makesos(m_dict,1),    [size(tmp,1) 1]);
        [~, idx] = max(tmp, [], 2);
        tmp = tmp ./ repmat(makesos(m_dict,1),    [size(tmp,1) 1]);
        for n=prod(recon_dim(1:end-1)):-1:1
            PD(n) = real(tmp(n,idx(n)));
            z(n,:) = PD(n) .* m_dict(:,idx(n)).';
            T1(n) = T1_dict(idx(n));
        end
        z  = reshape(z,  recon_dim);
        PD = reshape(PD, recon_dim(1:end-1));
        T1 = reshape(T1, recon_dim(1:end-1));
        
        %         figure(2); hold all; plot(squeeze(z(40,27,:)));
    end
    
    if length(recon_dim) == 3
        figure(234); imagesc(array2mosaic(abs(z(:,:,1:100:720)))); title(['iteration = ', num2str(j)]);
        figure(12342); subplot(2,1,1); imagesc(PD); colormap jet; colorbar
        title('PD [a.u.]');
        figure(12342); subplot(2,1,2); imagesc(T1, [0 5]); colormap jet; colorbar
        title('T1 [s]');
    elseif length(recon_dim) == 4
        tmp = abs(z(:,:,slices,1:100:720));
        tmp = array2mosaic(tmp(:,:,:), [8 length(slices)]);
        figure(234); imagesc(tmp); title(['iteration = ', num2str(j)]);
        figure(12342); subplot(2,1,1); imagesc(array2mosaic(PD(:,:,slices))); colormap jet; colorbar
        title('PD [a.u.]');
        figure(12342); subplot(2,1,2); imagesc(array2mosaic(T1(:,:,slices)), [0 5]); colormap jet; colorbar
        title('T1 [s]');
    else
        error('Images must be either 2D or 3D');
    end
    drawnow;
end

end