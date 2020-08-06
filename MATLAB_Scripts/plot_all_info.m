function [] = plot_all_info(figure_handle, time, freq, clean_signal, clean_sig_fft_mag, clean_sig_title, noisy_signal, noisy_sig_fft_mag, noisy_sig_title, predicted_fft_mag)
%plot_all_info
figure(figure_handle);
clf;

subplot(2, 3, 1)
plot(time, clean_signal)
xlabel("Time (s)")
ylabel("Signal Amplitude")
title(strcat(clean_sig_title, " Time Domain"))

subplot(2, 3, 2)
plot(freq, clean_sig_fft_mag)
xlabel("Frequency (Hz)")
ylabel("Signal Magnitude")
title(strcat(clean_sig_title, " Frequency Domain"))
axis([0, 2.5e7, 0, 1])

subplot(2, 3, 3)
plot(freq, (clean_sig_fft_mag - noisy_sig_fft_mag), 'x', freq, (clean_sig_fft_mag - predicted_fft_mag), 'o')
xlabel("Frequency (Hz)")
ylabel("Difference in Signal Magnitudes")
title("Differences between Clean Signal FFT and Noisy Signal FFT / Predicted Signal FFT")
legend("Clean FFT - Noisy FFT", "Clean FFT - Predicted FFT")
axis([0, 2.5e7, -0.1, 1])

subplot(2, 3, 4)
plot(time, noisy_signal)
xlabel("Time (s)")
ylabel("Signal Amplitude")
title(strcat(noisy_sig_title, " Time Domain"))

subplot(2, 3, 5)
plot(freq, noisy_sig_fft_mag)
xlabel("Frequency (Hz)")
ylabel("Signal Magnitude")
title(strcat(noisy_sig_title, " Frequency Domain"))
axis([0, 2.5e7, 0, 0.2])

subplot(2, 3, 6)
plot(freq, predicted_fft_mag)
xlabel("Frequency (Hz)")
ylabel("Predicted Signal Magnitude")
title("DFT Computed using ML Tactics")
axis([0, 2.5e7, 0, 0.2])

end

