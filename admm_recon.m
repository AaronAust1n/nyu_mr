function [T12, PD, x] = admm_recon(E, recon_dim, data, T12_dict, m_dict, sos_dict, n_iter, cg_iter, mu1, mu2, lambda_l21, verbose, savestr)

% for display
if length(recon_dim) == 4
    slices = 7:2:13;
end

if nargin < 10 || isempty(verbose)
    verbose = 1;
end
if nargin < 9 || isempty(lambda_l21)
    lambda_l21 = .25;
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

if lambda_l21 > 0
    WL = wavelet_operator(recon_dim(1:end-1), 3, 'db2');
end

for j=0:n_iter
    tic;
    
    if j == 0
        backprojection = E' * data;
%         x = backprojection;
        
%         if lambda_l21 > 0    
%             f = @(x) E'*(E*x) + mu1/2 * x + mu2/2 * (WL' * (WL * x));
%         else
%             f = @(x) E'*(E*x) + mu1/2 * x;
%         end
        f = @(x) E'*(E*x);
            
        S = virtualMatrix(f,size(E,2));
        x = conjugateGradient(S,backprojection,1e-6,cg_iter,[],verbose_cg,0);
%         x = reshape(pcg(f,backprojection(:),1e-6,cg_iter), E.imageDim);
        
        y = zeros(size(x));
        if lambda_l21 > 0
            WLx = WL * x;
            z = zeros(size(WLx));
        end
    else
%         if j==1
%             x = repmat(c, [1 1 size(D,3)]) .* D;
%         end
        b = backprojection - y/2 + D .* repmat(sum(conj(D) .* y/2, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)]);
        
        if lambda_l21 > 0
            G = WLx - z/mu2;
            Tl2 = makesos(G, 2);
            G = G - G ./ repmat(Tl2, [1 recon_dim(end)]) * lambda_l21/mu2;
            G(isnan(G)) = 0;
            G = G .* repmat(Tl2 > lambda_l21/mu2, [1 recon_dim(end)]);            
            b = b + WL' * (mu2 * G + z)/2;
            
            f = @(x) E'*(E*x) + mu1/2 * (x - D .* repmat(sum(conj(D) .* x, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)])) + mu2/2 * (WL' * (WL * x));
        else
            f = @(x) E'*(E*x) + mu1/2 * (x - D .* repmat(sum(conj(D) .* x, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)]));
        end
        
        S = virtualMatrix(f,size(E,2));
        x = conjugateGradient(S,b,1e-6,cg_iter,x,verbose_cg,0);
%         x = pcg(f,b,1e-6,cg_iter,[],[],x);
        
        if lambda_l21 > 0
            WLx = WL * x;
            z = z + mu2 * (G - WLx);
        end
    end
    x  = reshape(x, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    y  = reshape(y, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    
    % calculate best fit: argmin of least squares = argmax correlation
%     try
%         c = (x * conj(m_dict));
%         [c, idx] = max(c, [], 2);
%     catch
%         clear c idx
%         for q=size(x,1):-1:1
%             [c(q,1),idx(q,1)] = max(x(q,:) * conj(m_dict), [], 2);
%         end
%     end
        
    clear idx
    for q=size(x,1):-1:1
        Dx = x(q,:) * m_dict;
        Dy = y(q,:) * m_dict;
        [~,idx(q,1)] = max(2*real(Dx.*Dy) + abs(Dx).^2, [], 2);
    end
    c = sum(x .* m_dict(:,idx)',2);
    
    D = m_dict(:,idx).';
    PD = c ./ sos_dict(idx).';
    T12 = T12_dict(idx,:);
    
    x   = reshape(x,    recon_dim);
    y   = reshape(y,    recon_dim);
    D   = reshape(D,    recon_dim);
    c   = reshape(c,    recon_dim(1:end-1));
    PD  = reshape(PD,   recon_dim(1:end-1));
    T12 = reshape(T12, [recon_dim(1:end-1), size(T12_dict,2)]);
    
    y = y + mu1 * (x - D .* repmat(sum(conj(D) .* x, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)]));
        
    if verbose == 1
        % display P(x) and (x-P(x))
        if length(recon_dim) == 4
            tmp = abs(repmat(c(:,:,slices), [1 1 1 size(D,4)]) .* D(:,:,slices,:));
            tmp = array2mosaic(tmp(:,:,:), [size(D,4) length(slices)]);
        else
            tmp = abs(repmat(c, [1 1 size(D,3)]) .* D);
            tmp = array2mosaic(tmp);
        end
        sfig(234); imagesc(tmp); title(['P(x) - iteration = ', num2str(j)]); colorbar
        
        if length(recon_dim) == 4
            tmp = abs(x(:,:,slices,:) - repmat(c(:,:,slices), [1 1 1 size(D,4)]) .* D(:,:,slices,:));
            tmp = array2mosaic(tmp(:,:,:), [size(D,4) length(slices)]);
        else
            tmp = abs(x - repmat(c, [1 1 size(D,3)]) .* D);
            tmp = array2mosaic(tmp);
        end
        sfig(235); imagesc(tmp); title(['(x - P(x)) - iteration = ', num2str(j)]); colorbar
        
        % display PD and T1
        if length(recon_dim) == 4
            sfig(12342); subplot(2,1,1); imagesc(array2mosaic(abs(PD(:,:,slices)))); colormap hot; colorbar; title('PD [a.u.]');
            subplot(2,1,2); imagesc(array2mosaic(T12(:,:,slices)), [0 2.5]); colormap hot; colorbar; title('T1 [s]');
        else
%             sfig(12342); 
%             subplot(4,2,[1,3]); imagesc(abs(PD)); colormap hot; colorbar; axis off; axis equal; title('PD [a.u.]');
%             subplot(4,2,[2,4]); imagesc(T12(:,:,1), [0 4]); colormap hot; colorbar; axis off; axis equal; title('T1 [s]');
%             subplot(4,2,[5,7]); imagesc(T12(:,:,2), [0 .3]); colormap hot; colorbar; axis off; axis equal; title('T2 [s]');
            
            sfig(12342); 
            subplot(4,2,[1,3]); imagesc(abs(PD)); colorbar; axis off; axis equal; title('PD [a.u.]');
            subplot(4,2,[2,4]); imagesc(T12(:,:,1), [0 2.5]); colorbar; axis off; axis equal; title('T1 [s]');
            subplot(4,2,[5,7]); imagesc(T12(:,:,2), [0  .2]); colormap(morgenstemning(256)); colorbar; axis off; axis equal; title('T2 [s]');
   
            
            if size(T12,3)>2
                subplot(4,2,[6,8]); imagesc(T12(:,:,3)); colormap hot; colorbar; axis off; axis equal; title('w [rad/s]');
            end
        end
        drawnow;
    end
    if verbose > 0
        display(['Iteration ', num2str(j)]);
        if nargin > 12 && ~isempty(savestr)
            save(savestr, 'T12', 'PD', 'x', 'j');
%             if j==5
%                 save(['j5_', savestr], 'T12', 'PD', 'x', 'j');
%             end
        end
        toc
    end
end
end