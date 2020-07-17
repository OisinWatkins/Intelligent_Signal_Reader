function [fft_freq, fft_actual, fft_mag] = full_fft(input_sig, signal_length, Fs)
%full_fft performs all necessary computations to calculate a signal's fft
%in a manner that is easy to interpret visually.
Y = fft(input_sig);
fft_actual = Y;

P2 = abs(Y/signal_length);
P1 = P2(1:signal_length/2+1);
P1(2:end-1) = 2*P1(2:end-1);

fft_freq = Fs*(0:(signal_length/2))/signal_length;
fft_mag = P1;

end

