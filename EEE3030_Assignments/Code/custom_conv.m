function y = custom_conv(x, h)
% CUSTOM_CONV Apply FIR filter using custom convolution implementation
%
% Usage:
%   y = custom_conv(x, h)
%
% Inputs:
%   x - Input signal (column or row vector)
%   h - FIR filter coefficients (column or row vector)
%
% Output:
%   y - Filtered signal (same length and orientation as x)
%
% Description:
%   Implements FIR filtering by direct convolution:
%       y[n] = sum_{k=0}^{N-1} h[k] * x[n-k]
%
%   Zero-padding is used for samples where n-k < 0.
%
% EEE3030 DSP Assignment - Task 2

    % Store original orientation
    is_column = iscolumn(x);
    
    % Convert to row vectors for processing
    x = x(:)';
    h = h(:)';
    
    % Get lengths
    L = length(x);  % Input signal length
    N = length(h);  % Filter length
    
    % Zero-pad the input signal (N-1 zeros at beginning)
    x_padded = [zeros(1, N-1), x];
    
    % Preallocate output
    y = zeros(1, L);
    
    % Perform convolution
    for n = 1:L
        accumulator = 0;
        for k = 1:N
            x_index = n + N - k;
            accumulator = accumulator + h(k) * x_padded(x_index);
        end
        y(n) = accumulator;
    end
    
    % Restore original orientation
    if is_column
        y = y(:);
    end
end