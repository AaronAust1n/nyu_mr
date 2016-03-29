function [m_dict, sos_dict, T12_dict] = MRF_dictionary(T1, T2, w, alpha, TR0, pSSFP, idx)

if nargin < 6 || isempty(pSSFP)
    pSSFP = 0;
end


%% setup TR and TD/TE
if pSSFP
    TD = zeros(length(alpha), 1);
    TE = TD;
    sin_theta = abs(sin(cumsum(alpha .* round(cos((1:length(alpha))' * pi)))));
    for ip=2:length(alpha)
        TD(ip-1) = TR0/2;
        TE(ip)   = TD(ip-1) * sin_theta(ip-1) / sin_theta(ip);
        
        if TE(ip) > TR0/2
            TE(ip)   = TR0/2;
            TD(ip-1) = TE(ip) * sin_theta(ip) / sin_theta(ip-1);
        end
    end
else
    TR = repmat(TR0, [length(alpha) 1]);
    TD = TR/2;
    TE = TD;
    TD(end) = 0;
    TE(1) = 0;
end


%% Make Vector with T1 and T2

if length(w) > 1
    T1 = repmat(T1,  [length(T2), 1, length(w)]);
    T2 = repmat(T2', [1, size(T1,2), length(w)]);
    w  = repmat(reshape(w, [1 1 length(w)]), [size(T2,1), size(T1,2), 1]);
    w = w(:); T1 = T1(:); T2 = T2(:);
    T12_dict = [T1, T2, w];
else
    T1 = repmat(T1, [length(T2), 1]);
    T2 = repmat(T2', [1 size(T1,2)]);
    T1 = T1(:); T2 = T2(:);
    T12_dict = [T1, T2];
end
T1 = T1.';
T2 = T2.';
w = w.';
    


%% SO(2) - complex
R = @(alpha) [cos(alpha) -sin(alpha);
    sin(alpha)  cos(alpha)];


M = complex(zeros(2, length(T1)));
M(2,:) = 1;
D = complex(ones(2, length(T1)));

m_dict = zeros(length(alpha),length(T1));

% dist = (1/T2d)./((1/T2d)^2 + (w).^2);
% dist = dist./sum(dist);

for ip=1:length(alpha)
    % Apply RF-pulse (that now acts only on the real part)
    Mtmp = R(alpha(ip)*round(cos(ip*pi))) * real(M);
    M = Mtmp + 1i*imag(M);
    
    % Free precession for TE
    if any(w~= 0)
        D(1,:) = exp(-1i*w*TE(ip));
        M = D .* M;
    end
    
    % Relaxation
    M(1,:) = M(1,:) .* exp(-TE(ip)./T2);
    M(2,:) = ones(1,size(M,2)) + (M(2,:)-ones(1,size(M,2))) .* exp(-TE(ip)./T1);
    
    % Calculate Signal at the echo time
    for iT1 = 1:length(T1);
        m_dict(ip,iT1) = M(1,iT1)*cos(pi*ip);
    end
    
    % Relaxation
    M(1,:) = M(1,:) .* exp(-TD(ip)./T2);
    M(2,:) = ones(1,size(M,2)) + (M(2,:)-ones(1,size(M,2))) .* exp(-TD(ip)./T1);
    
    % Free precession for TD
    if any(w~= 0)
        D(1,:) = exp(-1i*w*TD(ip));
        M = D .* M;
    end
    
end

if nargin > 6 && ~isempty(idx)
    m_dict = m_dict(idx,:);
end
sos_dict = makesos(m_dict, 1);
% m_dict = (m_dict./repmat(sos_dict, [size(m_dict,1) 1]));

end