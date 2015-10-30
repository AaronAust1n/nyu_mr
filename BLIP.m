function [T12, PD, x] = BLIP(nuFFT, recon_dim, data, T1_dict, m_dict, sos_dict, n_iter, x, lambda, verbose, mid)

% for display
% slices = 7:2:13;
slices = 2*(9:12);
% t = [1:2:20, size(data,3)];
t = 1:ceil(size(m_dict,1)/9):size(m_dict,1);

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

% lambda_tv = 1e-8;
lambda_tv = 0;
if lambda_tv > 0
    [hf, hdf] = L1Norm(lambda_tv,finiteDifferenceOperator(1), finiteDifferenceOperator(2), finiteDifferenceOperator(3));
    % [hf, hdf] = L1Norm(lambda_tv,waveletDecompositionOperator(recon_dim(1:end-1), 3, 'db2'));
end

kappa = 20;
SsS = data(:)' * data(:);
cost_last_iter = SsS;
for j=1:n_iter
    tic;
    
    if j == 1
        r = nuFFT' * data;
        Er = nuFFT * r;
        Ex = 0 * Er;
    else
%         Ex = nuFFT * x;
        Ex = Ex + kappa * Er;
        if lambda_tv > 0
            r = nuFFT' * (data - Ex) + lambda.^2 * (Px - x) - hdf(x);
        else
            r = nuFFT' * (data - Ex) + lambda.^2 * (Px - x);
        end
        Er = nuFFT * r;
    end
    
    
    x  = reshape(x, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    r  = reshape(r, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    
    
    % linesearch
    rr = (r(:)'*r(:));
    linesearch = 1;
    backtrack  = 1;
    cost_last_backtrack = 0;
    change_direction = 0;
    while linesearch
        xkr = x + kappa * r;
        PD = (xkr * m_dict);
        [PD, idx] = max(PD, [], 2);
        Px = repmat(PD, [1 recon_dim(end)]) .* m_dict(:,idx).';
        xkrmPx = xkr - Px;
        
        if lambda_tv > 0
            cost = real(Ex(:)'*Ex(:) + 2 * kappa * Ex(:)' * Er(:) + kappa^2 * Er(:)' * Er(:) - 2 * data(:)' * (Ex(:) + kappa * Er(:)) + SsS + lambda^2 * xkrmPx(:)' * xkrmPx(:)) + hf(x, kappa, r, cost_last_backtrack == 0);
        else
            cost = real(Ex(:)'*Ex(:) + 2 * kappa * Ex(:)' * Er(:) + kappa^2 * Er(:)' * Er(:) - 2 * data(:)' * (Ex(:) + kappa * Er(:)) + SsS + lambda^2 * xkrmPx(:)' * xkrmPx(:));
        end
        
        if cost_last_backtrack > 0 && cost > cost_last_backtrack
            if change_direction == 0
                backtrack = ~backtrack;
                change_direction = 1;
            else
                warning('Changed search direction twice. Stopping linesearch here');
                break;
            end
        end
        if (cost_last_iter - cost) > (0.1 * kappa * rr)
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
    T12 = T1_dict(idx,:);
    
    x  = reshape(xkr,recon_dim);
    Px = reshape(Px, recon_dim);
    PD = reshape(PD, recon_dim(1:end-1));
    T12 = reshape(T12, [recon_dim(1:end-1), size(T1_dict,2)]);
        
    if verbose        
        % display P(x) and (x-P(x))
        if length(recon_dim) == 4
            tmp = abs(Px(:,:,slices,t));
            tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        else
            tmp = abs(Px(:,:,t));
            tmp = array2mosaic(tmp);
        end
        sfig(234); imagesc(tmp); title(['P(x) - iteration = ', num2str(j)]); colorbar
        if length(recon_dim) == 4
            tmp = abs(x(:,:,slices,t) - Px(:,:,slices,t));
            tmp = array2mosaic(tmp(:,:,:), [length(t) length(slices)]);
        else
            tmp = abs(x(:,:,t) - Px(:,:,t));
            tmp = array2mosaic(tmp);
        end
        sfig(235); imagesc(tmp); title(['(x - P(x)) - iteration = ', num2str(j)]); colorbar
        
        % display PD and T1
        if length(recon_dim) == 4
            sfig(12342); subplot(2,1,1); imagesc(array2mosaic(abs(PD(:,:,slices))), [0 7e-4]); colormap jet; colorbar
            title('PD [a.u.]');
            sfig(12342); subplot(2,1,2); imagesc(array2mosaic(T12(:,:,slices)), [0 2]); colormap jet; colorbar
            title('T1 [s]');
        else
            sfig(12342); subplot(4,2,[1,3]); imagesc(abs(PD)); colormap hot; colorbar; axis off; axis equal;
            title('PD [a.u.]');
            sfig(12342); subplot(4,2,[2,4]); imagesc(T12(:,:,1), [0 4]); colormap hot; colorbar; axis off; axis equal;
            title('T1 [s]');
            sfig(12342); subplot(4,2,[5,7]); imagesc(T12(:,:,2), [0 .3]); colormap hot; colorbar; axis off; axis equal;
            title('T2 [s]');
            
            % display convergence
            display(['Iteration ', num2str(j)]);
            display(['cost = ',  num2str(cost)]);
            display(['kappa = ', num2str(kappa)]);
            sfig(12342); subplot(4,2,6); hold all; plot(j, cost, '.'); xlabel('Iteration'); ylabel('cost');
            sfig(12342); subplot(4,2,8); hold all; plot(j,   rr, '.'); xlabel('Iteration'); ylabel('rr');
        end
        drawnow;
    else
        save(['mid', num2str(mid), '_recon_lambda_p', num2str(lambda*100)], 'T1', 'PD', 'x', 'j');
    end
    toc
end

end