function [output] = next_power_of_2(input)
%next_power_of_2 takes any input value and computes the next power of two
%that is larger than the input.
if input == 0
    output = 1;
else
    currentPow = log2(input);
    if mod(currentPow, 1) == 0
        output = 2^(currentPow + 1);
    else
        output = 2^(ceil(currentPow));
    end
end
end

