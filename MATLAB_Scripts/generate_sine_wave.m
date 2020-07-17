function [t, g] = generate_sine_wave(frequency, Fs, signal_length, amplitude_noise_profile, phase_noise_profile)
%generate_sine_wave makes a sine wave 

T = 1/Fs;
t = (0 : signal_length-1) * T;
g = amplitude_noise_profile .* sin((2 * pi * frequency * t) + phase_noise_profile);
end

