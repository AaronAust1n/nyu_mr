function [T1, PD, x] = BLIP(nuFFT, recon_dim, data, T1_dict, m_dict, n_iter, x, lambda, verbose, mid)

% for display
% slices = 7:2:13;
slices = 9:12;
t = [1:2:20, size(data,3)];

if nargin < 10 || isempty(mid)
    mid = 000;
end
if nargin < 9 || isempty(verbose)
    verbose = 1;
end
if nargin < 8 || isempty(lambda)
    lambda = .25;
end
if nargin < 78 || isempty(x)
    x = complex(zeros(recon_dim));
end
if nargin < 6 || isempty(n_iter)
    n_iter = 100;
end

sos_dict = makesos(m_dict,1);

for j=1:n_iter
    tic;
    
    if j == 1
        r = nuFFT' * data;
        Er = nuFFT * r;
        Ex = 0 * Er;
    else
        Ex = nuFFT * x;
        r = nuFFT' * (data - Ex) + lambda.^2 * (Px - x);
        Er = nuFFT * r;
    end
    
    
    x  = reshape(x, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    r  = reshape(r, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    
    %     phi = angle(mean(x(:,end/2+1:end), 2));
    %     xm = real(x .* exp(-1i * repmat(phi, [1 size(x,2)]))) * m_dict;
    %     xm = x * m_dict;
    %     rm = r * m_dict;
    
    kappa_all = .5:.2:2;
    for j_kappa = length(kappa_all):-1:1
        kappa = kappa_all(j_kappa);
        
        %         PD = xm + kappa * rm;
        %         phi = angle(mean(PD(:,end/2+1:end), 2));
        %         PD = real(PD .* exp(-1i * repmat(phi, [1 recon_dim(end)])));
        %         PD(PD<0) = 0;
        %         PD = PD ./ repmat(sos_dict, [size(PD,1) 1]);
        %         [PD, idx] = max(PD, [], 2);
        %         PD = PD ./ sos_dict(idx).';
        %         Px = repmat(PD .* exp(1i * phi), [1 recon_dim(2)]) .* m_dict(:,idx).';
        
        xkr = x + kappa * r;
        
        PD = (x * m_dict);
        PD = PD ./ repmat(sos_dict, [size(PD,1) 1]);
        [PD, idx] = max(PD, [], 2);
        PD = PD ./ sos_dict(idx).';
        Px = repmat(PD, [1 recon_dim(end)]) .* m_dict(:,idx).';
        
        
        xkrmPx = xkr - Px;
        cost(j_kappa) = real(Ex(:)'*Ex(:) + 2 * kappa * Ex(:)' * Er(:) + kappa^2 * Er(:)' * Er(:) - 2 * data(:)' * (Ex(:) + kappa * Er(:)) + data(:)' * data(:) + lambda^2 * xkrmPx(:)' * xkrmPx(:));
    end
    [min_cost, cost_idx] = min(cost);
    kappa = kappa_all(cost_idx);
    display(['cost = ', num2str(min_cost)]);
    sosr = makesos(r(:));
    display(['sos(r) = ', num2str(sosr)]);
    display(['kappa = ', num2str(kappa)]);
    
    x  = reshape(x, recon_dim);
    r  = reshape(r, recon_dim);
    
    
    figure(94234); subplot(2,1,1); hold all; plot(j, min_cost, '.'); xlabel('Iteration'); ylabel('cost');
    figure(94234); subplot(2,1,2); hold all; plot(j, sosr, '.'); xlabel('Iteration'); ylabel('sos(r)');
       
    x = x + kappa * r;
    
    if verbose
        tmp = x(:,:,slices,t);
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(09568756); imagesc(angle(tmp)); colormap jet
        sfig(4234); imagesc(abs(tmp));
        title(['iteration = ', num2str(j-.5), '; kappa = ', num2str(kappa)]); drawnow;
    end
    
    
    %     [Px, PD, T1] = P(x, m_dict, recon_dim);
    
    x  = reshape(x, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    
    PD = (x * m_dict);
    PD = PD ./ repmat(sos_dict, [size(PD,1) 1]);
    [PD, idx] = max(PD, [], 2);
    PD = PD ./ sos_dict(idx).';
    Px = repmat(PD, [1 recon_dim(end)]) .* m_dict(:,idx).';
    T1 = T1_dict(idx);
    
    x  = reshape(x,  recon_dim);
    Px = reshape(Px, recon_dim);
    PD = reshape(PD, recon_dim(1:end-1));
    T1 = reshape(T1, recon_dim(1:end-1));
    
    if verbose
        tmp = abs(Px(:,:,slices,t));
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(234); imagesc(tmp); title(['P(x) - iteration = ', num2str(j)]); colorbar
        tmp = abs(x(:,:,slices,t) - Px(:,:,slices,t));
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(235); imagesc(tmp); title(['(x - P(x)) - iteration = ', num2str(j)]); colorbar
        sfig(12342); subplot(2,1,1); imagesc(array2mosaic(abs(PD(:,:,slices))), [0 7e-4]); colormap jet; colorbar
        title('PD [a.u.]');
        sfig(12342); subplot(2,1,2); imagesc(array2mosaic(T1(:,:,slices)), [0 2]); colormap jet; colorbar
        title('T1 [s]');
        drawnow;
    else
        save(['mid', num2str(mid), '_recon_lambda_p', num2str(lambda*100)], 'T1', 'PD', 'z', 'j');
    end
    toc
end

end