clc
clear

% Set the signal length to be constant and load the corresponding Twiddle
% Array file.
signal_length = 2^12;
load('C:\Users\owatkins\OneDrive - Analog Devices, Inc\Documents\Project Folder\Project 3\Code\Python\MATLAB_Scripts\default_twiddle_arrays\TA_2^12.mat');

% Copy the Twiddle Array into a new variable. This new variable is the one 
% which will be trained.
W_alt = W;

% Open a new figure and initialise a counter.
fig1 = figure(1);
cntr = 0;

while true
    % Set the frequency of the signal and the sampling frequency to be at
    % least twice the frequency.
    Frequency = 10000000 + 10000000*rand(1);
    Fs = 50000000;

    % Define both the clean and the noisy profiles to apply to the
    % generated signal.
    clean_amp_profile = ones(1, signal_length);
    clean_phase_profile = zeros(1, signal_length);
    noisy_amp_profile = 1.1 * rand(1, signal_length);
    noisy_phase_profile = 10 * rand(1, signal_length);
    
    % Generate the signals.
    [clean_time, clean_sig] = generate_sine_wave(Frequency, Fs, signal_length, clean_amp_profile, clean_phase_profile);
    [noisy_time, noisy_sig] = generate_sine_wave(Frequency, Fs, signal_length, noisy_amp_profile, noisy_phase_profile);
    
    % Compute the FFT's
    [clean_freq, clean_fft, clean_fft_mag] = full_fft(clean_sig, signal_length, Fs);
    [noisy_freq, noisy_fft, noisy_fft_mag] = full_fft(noisy_sig, signal_length, Fs);
    
    % Compute the DFT using the custom Twiddle Array, then also compute its
    % one-sided magnitude.
    predicted_freq = mtimes(noisy_sig, W_alt);
    P2 = abs(predicted_freq / signal_length);
    P1 = P2(1 : signal_length/2+1);
    P1(2 : end-1) = 2 * P1(2 : end-1);
    
    prediction = P1;
    
    % Plot all information.
    plot_all_info(fig1, clean_time, clean_freq, clean_sig, clean_fft_mag, "Clean Signal", noisy_sig, noisy_fft_mag, "Noisy Signal", prediction);
    
    % Perform the training with L2 Regularisation.
    delta = ((predicted_freq.^2) - (clean_fft.^2));
    dEdW = mtimes(noisy_sig.', delta);
    
    complexity_score = (W - W_alt).^2;
    
    LR = sqrt(sqrt(sqrt(sum(abs(delta)))));
    
    % L2 = ;
    
    W_alt = W_alt - (LR * (dEdW ./ max(dEdW)));% + (L2 * (complexity_score./max(complexity_score)));
    
    % Control the running of this loop.
    if cntr == 5
        choice = input('Perform another iteration? [y]/n ', 's');
        if strcmp(choice, 'n')
            break
        end
        cntr = -1;
    end
    cntr = cntr + 1;
    clc    
end
clc
