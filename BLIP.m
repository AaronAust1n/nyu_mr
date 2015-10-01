function [T1, PD, x] = BLIP(nuFFT, recon_dim, data, T1_dict, m_dict, sos_dict, n_iter, x, lambda, verbose, mid)

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
if nargin < 7 || isempty(x)
    x = complex(zeros(recon_dim));
end
if nargin < 6 || isempty(n_iter)
    n_iter = 100;
end

kappa = 50;
SsS = data(:)' * data(:);
cost_last_iter = SsS;
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
    
    
    % linesearch
    rr = (r(:)'*r(:));
    linesearch = 1;
    backtrack  = 1;
    cost_last_backtrack = 0;
    while linesearch
        xkr = x + kappa * r;
        PD = (xkr * m_dict);
        [PD, idx] = max(PD, [], 2);
        Px = repmat(PD, [1 recon_dim(end)]) .* m_dict(:,idx).';
        xkrmPx = xkr - Px;
        cost = real(Ex(:)'*Ex(:) + 2 * kappa * Ex(:)' * Er(:) + kappa^2 * Er(:)' * Er(:) - 2 * data(:)' * (Ex(:) + kappa * Er(:)) + SsS + lambda^2 * xkrmPx(:)' * xkrmPx(:));
        
        if cost_last_backtrack > 0 && cost > cost_last_backtrack
            backtrack = ~backtrack;
        end
        if (cost_last_iter - cost) > (kappa * rr)
            cost_last_iter = cost;
            linesearch = 0;
        elseif backtrack
            cost_last_backtrack = cost;
            kappa = 0.8 * kappa;
        else
            cost_last_backtrack = cost;
            kappa = 2 * kappa;
        end
    end
    
    
    PD = PD ./ sos_dict(idx).';
    T1 = T1_dict(idx);
    
    x  = reshape(xkr,recon_dim);
    Px = reshape(Px, recon_dim);
    PD = reshape(PD, recon_dim(1:end-1));
    T1 = reshape(T1, recon_dim(1:end-1));
        
    if verbose
        % display convergence
        display(['cost = ',  num2str(cost)]);
        display(['kappa = ', num2str(kappa)]);
        sfig(94234); subplot(2,1,1); hold all; plot(j, cost, '.'); xlabel('Iteration'); ylabel('cost');
        sfig(94234); subplot(2,1,2); hold all; plot(j,   rr, '.'); xlabel('Iteration'); ylabel('rr');
        
        % display x
        tmp = x(:,:,slices,t);
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(09568756); imagesc(angle(tmp)); colormap jet
        sfig(4234); imagesc(abs(tmp));
        title(['iteration = ', num2str(j-.5), '; kappa = ', num2str(kappa)]); drawnow;
        
        % display P(x) and (x-P(x))
        tmp = abs(Px(:,:,slices,t));
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(234); imagesc(tmp); title(['P(x) - iteration = ', num2str(j)]); colorbar
        tmp = abs(x(:,:,slices,t) - Px(:,:,slices,t));
        tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        sfig(235); imagesc(tmp); title(['(x - P(x)) - iteration = ', num2str(j)]); colorbar
        
        % display PD and T1
        sfig(12342); subplot(2,1,1); imagesc(array2mosaic(abs(PD(:,:,slices))), [0 7e-4]); colormap jet; colorbar
        title('PD [a.u.]');
        sfig(12342); subplot(2,1,2); imagesc(array2mosaic(T1(:,:,slices)), [0 2]); colormap jet; colorbar
        title('T1 [s]');
        drawnow;
    else
        save(['mid', num2str(mid), '_recon_lambda_p', num2str(lambda*100)], 'T1', 'PD', 'z', 'j');
    end
    display(['Iteration ', num2str(j)]);
    toc
end

end