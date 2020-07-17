function [] = generate_twiddle_arrays(save_directory, lowest_power_of_2, highest_power_of_2)
%generate_twiddle_arrays generates and saves twiddle arrays used in dft's
%and saves them into .mat files in the given directory.

for power = lowest_power_of_2 : 1 : highest_power_of_2
    num_samples = 2 ^ power;
    W = zeros(num_samples, num_samples);
    for a = 1 : (num_samples)
        for b = 1 : (num_samples)
            W(a, b) = Wnp(num_samples, ((a-1) * (b-1)));
        end
    end
    file_dir = strcat(save_directory, "\TA_2^", num2str(power), ".mat");
    save(file_dir, 'W');
end
disp("Twiddle Array Generation Complete")
end

