function [y, norm] = nuc_norm_prox_2d(x, lambda,mu)
%Note: The function has been adapted to deal with complex valued input, see handwritten notes and testsvd.m for a test

px = mu*squeeze(x(1,:,:,:));
py = mu*squeeze(x(2,:,:,:));


%Spectral norm prox:

P = size(px,3);

% build A'*A %NOTE: Adapted for nuclear-norm prox
a = sum(abs(px).^2, 3);
b = sum(conj(px).*py, 3);
c = sum(abs(py).^2, 3);


% compute eigenvalues
d = a + c;
e = a - c;
f = sqrt(4*abs(b).^2 + e.^2);


sigma1 = sqrt((d+f)/2);
sigma2 = sqrt((d-f)/2);

norm = sum(col(sigma1 + sigma2))/mu;

g = (e + f)/2;
h = (e - f)/2;

k = g.^2 + abs(b).^2;
%k(k == 0) = 1;
l = h.^2 + abs(b).^2;
%l(l == 0) = 1;

% modify singular values
fac1 = 1./max(1, sigma1/lambda);
fac2 = 1./max(1, sigma2/lambda);

v1 = repmat(g, [1 1 P]).*px + repmat(conj(b), [1 1 P]).*py;
v2 = repmat(h, [1 1 P]).*px + repmat(conj(b), [1 1 P]).*py;


qx = repmat(fac1.*g, [1 1 P]).*v1./repmat(k, [1 1 P]) ...
    + repmat(fac2.*h, [1 1 P]).*v2./repmat(l, [1 1 P]);
qy = repmat(fac1.*b, [1 1 P]).*v1./repmat(k, [1 1 P]) ...
    + repmat(fac2.*b, [1 1 P]).*v2./repmat(l, [1 1 P]);



%Correction for special case of orthogonal columns in A (see handwritten notes)
idb = abs(b) == 0; %Case of orthogonal columns in original matrix
idbr =  repmat( idb,[1 1 P]); %Vector extension of above
idpx = sum(abs(px),3)==0; %Subcase of first vector being zero

%Only if px=0, fac need to be modified
fac2(idpx) = fac1(idpx);
fac1(idpx) = 1;


%Prox in case of orthogonal columns of A
if size(px,1)*size(px,2)>1
    qx(idbr) = px(idbr).*repmat(fac1(idb),[P 1]);
    qy(idbr) = py(idbr).*repmat(fac2(idb),[P 1]);
elseif ~isempty(px(idbr)) %special case for 1x1 array
    qx(idbr) = px(idbr).*repmat(fac1(idb),[1 1 P]);
    qy(idbr) = py(idbr).*repmat(fac2(idb),[1 1 P]);
end

%Postprocessing: Conversion to spectral norm prox
qx = (px - qx)./mu;
qy = (py - qy)./mu;

%Expand singelton dimensions (faster than with permute)
y(2,:,:,:) = qy;
y(1,:,:,:) = qx;


end
