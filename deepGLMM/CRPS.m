function f = CRPS(y,p)
% compute the continuous ranked probability score CRPS: the smaller the
% better
%f = sum(-p.^2.*(1-y)-(1-p).^2.*y);
p(p==0) = 10e-10;
p(p==1) = 1-(10e-10);
f = -sum(y.*log(p)+(1-y).*log(1-p));
end