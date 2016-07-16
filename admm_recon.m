function [T12, PD, x] = admm_recon(E, recon_dim, data, T12_dict, m_dict, sos_dict, n_iter, cg_iter, mu1, mu2, lambda, nuc_flag, WL, verbose)

% for display
if length(recon_dim) == 4
    slices = 7:2:13;
end

if nargin < 10 || isempty(verbose)
    verbose = 1;
end
if nargin < 9 || isempty(lambda)
    lambda = .25;
end
if nargin < 8 || isempty(mu1)
    mu1 = .25;
end
if nargin < 7 || isempty(cg_iter)
    cg_iter = 50;
end
if nargin < 6 || isempty(n_iter)
    n_iter = 10;
end

verbose_cg = verbose;
if length(recon_dim) == 4 && verbose == 1
    verbose_cg = 2;
end


for j=0:n_iter
    tic;
    
    if j == 0
        backprojection = E' * data;
        f = @(x) E'*(E*x);
        
        S = virtualMatrix(f,size(E,2));
        x = conjugateGradient(S,backprojection,1e-6,cg_iter,[],verbose_cg,0);
        
        y = zeros(size(x));
        if lambda > 0
            WLx = WL * x;
            z = zeros(size(WLx));
        end
    else
        b = backprojection - mu1 * y + mu1 * D .* repmat(sum(conj(D) .* y, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)]);
%         b = backprojection + mu1 * (DDhx - y);

        if lambda > 0
            if nuc_flag
                G = nuc_norm_prox_2d(WLx-z,lambda,mu2);
            else
                G = WLx - z;
                Tl2 = makesos(G, length(size(G)));
                G = G - G ./ repmat(Tl2, [ones(1, length(size(G))-1) recon_dim(end)]) * lambda/mu2;
                G(isnan(G)) = 0;
                G = G .* repmat(Tl2 > lambda/mu2, [ones(1, length(size(G))-1) recon_dim(end)]);
            end
            b = b + mu2 * (WL' * (G + z));
            
%             f = @(x) E'*(E*x) + mu1 * x + mu2 * (WL' * (WL * x));
            f = @(x) E'*(E*x) + mu1 * (x - D .* repmat(sum(conj(D) .* x, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)])) + mu2 * (WL' * (WL * x));
        else
%             f = @(x) E'*(E*x) + mu1 * x;
            f = @(x) E'*(E*x) + mu1 * (x - D .* repmat(sum(conj(D) .* x, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)]));
        end
        
        S = virtualMatrix(f,size(E,2));
        x = conjugateGradient(S,b,1e-6,cg_iter,x,verbose_cg,0);
        %         x = pcg(f,b,1e-6,cg_iter,[],[],x);
        
        if lambda > 0
            WLx = WL * x;
            z = z + G - WLx;
        end
    end
    x  = reshape(x, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    y  = reshape(y, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    
    clear idx
    for q=size(x,1):-1:1
%         [~,idx(q,1)] = max((x(q,:)+y(q,:)) * m_dict, [], 2);
        
        Dx = x(q,:) * m_dict;
        Dy = y(q,:) * m_dict;
        [~,idx(q,1)] = max(2*real(Dx.*conj(Dy)) + abs(Dx).^2, [], 2);
        
%         a = sum(abs(repmat(x(q,:).' + y(q,:).', [1 size(m_dict,2)]) - repmat(x(q,:) * m_dict, [size(m_dict,1) 1]) .* m_dict).^2, 1);
%         [~,idx(q,1)] = min(a, [], 2);
    end
    
    D = double(m_dict(:,idx)).';
    Dhx = sum(conj(D) .* x ,2);
    PD = Dhx ./ sos_dict(idx).';
    T12 = T12_dict(idx,:);
    
    x   = reshape(x,    recon_dim);
    y   = reshape(y,    recon_dim);
    D   = reshape(D,    recon_dim);
    Dhx = reshape(Dhx,  recon_dim(1:end-1));
    PD  = reshape(PD,   recon_dim(1:end-1));
    T12 = reshape(T12, [recon_dim(1:end-1), size(T12_dict,2)]);
    
    DDhx = D .* repmat(Dhx, [ones(1,length(recon_dim)-1) recon_dim(end)]);
    y = y + x - DDhx;
    
    if verbose == 1
        % display D*Dh*x and (x-D*Dh*x)
        if length(recon_dim) == 4
            tmp = abs(repmat(Dhx(:,:,slices), [1 1 1 size(D,4)]) .* D(:,:,slices,:));
            tmp = array2mosaic(tmp(:,:,:), [size(D,4) length(slices)]);
        else
            tmp = abs(repmat(Dhx, [1 1 size(D,3)]) .* D);
            tmp = array2mosaic(tmp);
        end
        sfig(234); imagesc(tmp); title(['D*Dh*x - iteration = ', num2str(j)]); colorbar
        
        if length(recon_dim) == 4
            tmp = abs(x(:,:,slices,:) - repmat(Dhx(:,:,slices), [1 1 1 size(D,4)]) .* D(:,:,slices,:));
            tmp = array2mosaic(tmp(:,:,:), [size(D,4) length(slices)]);
        else
            tmp = abs(x - repmat(Dhx, [1 1 size(D,3)]) .* D);
            tmp = array2mosaic(tmp);
        end
        sfig(235); imagesc(tmp); title(['(x - D*Dh*x) - iteration = ', num2str(j)]); colorbar
        
        % display PD and T1
        if length(recon_dim) == 4
            sfig(12342); subplot(2,1,1); imagesc(array2mosaic(abs(PD(:,:,slices)))); colormap hot; colorbar; title('PD [a.u.]');
            subplot(2,1,2); imagesc(array2mosaic(T12(:,:,slices)), [0 2.5]); colormap hot; colorbar; title('T1 [s]');
        else
            sfig(12342);
            subplot(4,2,[1,3]); imagesc(abs(PD)); colorbar; axis off; axis equal; title('PD [a.u.]');
            subplot(4,2,[2,4]); imagesc(T12(:,:,1), [0 2.5]); colorbar; axis off; axis equal; title('T1 [s]');
            subplot(4,2,[5,7]); imagesc(T12(:,:,2), [0  .2]); colormap(morgenstemning(256)); colorbar; axis off; axis equal; title('T2 [s]');
            
            
            if size(T12,3)>2
                subplot(4,2,[6,8]); imagesc(T12(:,:,3)); colormap hot; colorbar; axis off; axis equal; title('w [rad/s]');
            end
        end
        
        if nuc_flag
            r = sum(col(abs(E*DDhx - data)).^2);
            [~, nuc_norm] = nuc_norm_prox_2d(WL * DDhx,1,1);
            if j==0
                sfig(9563); hold off;
            else
                sfig(9563); hold all;
            end
            plot(j, r + lambda*abs(nuc_norm), 'o');
        end
        
        drawnow;
    end
    if verbose > 0
        display(['Iteration ', num2str(j)]);
        toc
    end
end
end