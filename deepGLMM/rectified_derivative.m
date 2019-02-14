function f = rectified_derivative(z)
f = ones(size(z));
f(z<=0) = 0;
end