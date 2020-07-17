function [twiddle] = Wnp(N,p)
%Wnp Generates a single twiddle factor given input arguments N and p
twiddle = exp(-1i * (2 * p * pi)/N);
end

