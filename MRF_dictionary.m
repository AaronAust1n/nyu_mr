function D = MRF_dictionary(T1, T2, w, alpha, TR0, pSSFP, idx, R)
% Calculates a dictionary for a fingerprinting sequences with the flip
% angle alpha. TR0 is either constant (pSSFP = 0) or follows the pSSFP
% pattern (pSSFP = 1)
%
% D = MRF_dictionary(T1, T2,[], alpha)
% D = MRF_dictionary(T1, T2, w, alpha)
% D = MRF_dictionary(T1, T2, w, alpha, TR0)
% D = MRF_dictionary(T1, T2, w, alpha, TR0, pSSFP)
% D = MRF_dictionary(T1, T2, w, alpha, TR0, pSSFP, idx)
% D = MRF_dictionary(T1, T2, w, alpha, TR0, pSSFP, idx, R)
%
% Input:
%   T1    = Array with T1 values to be sampled (obligatory)
%   T2    = Array with T2 values to be sampled (obligatory)
%   w     = Array with frequencies (radians/time_unit; optional)
%   alpha = Array with flip angles (radians; obligatory)
%   TR0   = Scalar: TR (pSSFP = 0) or the basis TR (pSSFP = 1) (obligatory)
%   pSSFP = Boolean: 0 -> contant TR, 1 -> pSSFP Pattern (optional; default = 0)
%   idx   = Array of boolean indicating whether the particular
%           repetition was sampled (optional; default: ones)
%   R     = Rank to which the dictionary should compressed (optional;
%           default = no compression)
%
% Output: Dictionary Struct with the fields:
%   D.magnetization in [NT1*NT2PNw R(Nt)]
%                      Matrix containing the magnetization evolution over
%                      time or SVD-component. All columns have been
%                      normalized to unit l2-norm.
%   D.normalization in [NT1*NT2*Nw]
%                      Vector containing the factor by which magnetization
%                      evolution has been normalized. Devide the correlation
%                      by this factor to get the proton density.
%   D.lookup_table  in [Nparameter NT1*NT2*Nw]
%                      Table containing the parameter that correspond to the
%                      magnetization evolutions in D.magnetization
%   D.parameter{:}  =  Strings specifying the parameters in the order of the
%                      look-up table
%   D.u             in [Nt R]
%                      Compression matrix resulting from the SVD of the 
%                      dictionary; empty if R is empty.
%
% For the pSSFP pattern, please refer to 
%   J. Assländer, S. J. Glaser, and J. Hennig, Pseudo Steady-State 
%   Free Precession for MR-Fingerprinting, Magn. Reson. Med., epub 
%   ahead of print, 2016.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Jakob Asslaender, August 2016
% New York University School of Medicine, Center for Biomedical Imaging
% University Medical Center Freiburg, Medical Physics
% jakob.asslaender@nyumc.org
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3 || isempty(w)
    w = 0;
end
if nargin < 5 || isempty(pSSFP)
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


%% Create a Vector with T1 and T2
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



%% Bloch Simulation in the complex SO(2) group
Rot = @(alpha) [cos(alpha) -sin(alpha);
    sin(alpha)  cos(alpha)];

M = complex(zeros(2, length(T1)));
M(2,:) = 1;
d = complex(ones(2, length(T1)));
m_dict = zeros(length(alpha),length(T1));

for ip=1:length(alpha)
    % Apply RF-pulse (that now acts only on the real part)
    Mtmp = Rot(alpha(ip)*round(cos(ip*pi))) * real(M);
    M = Mtmp + 1i*imag(M);
    
    % Free precession for TE
    if any(w~= 0)
        d(1,:) = exp(-1i*w*TE(ip));
        M = d .* M;
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
        d(1,:) = exp(-1i*w*TD(ip));
        M = d .* M;
    end
    
end

% Remove unaquired time frames
if nargin > 6 && ~isempty(idx)
    m_dict = m_dict(idx,:);
end

if nargin > 7 && ~isempty(R)
    [u,~,~]=svd(m_dict, 'econ');
    u = u(:,1:R);
    m_dict = u'*m_dict;
    D.u    = u;
else
    D.u = [];
end

sos_dict = makesos(m_dict, 1);
m_dict = (m_dict./repmat(sos_dict, [size(m_dict,1) 1]));

D.magnetization = m_dict;
D.normalization = sos_dict;
D.lookup_table  = T12_dict;
D.parameter{1}  = 'T1';
D.parameter{2}  = 'T2';
if length(w) > 1
    D.parameter{3}  = 'w';
end

end