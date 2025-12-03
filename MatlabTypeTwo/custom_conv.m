function y = custom_conv(x, h)
% CUSTOM_CONV Performs linear convolution of signal x with filter h
%
% Purpose:
%   Implements FIR filtering through discrete-time convolution
%   Convolution is the fundamental operation for linear time-invariant systems
%
% Mathematical Definition:
%   y[n] = sum(h[k] * x[n-k]) for k = 0 to Nh-1
%   This is the discrete convolution sum
%
% Inputs:
%   x - Input signal (column vector)
%       This is the signal to be filtered (e.g., noisy AM signal)
%   h - Filter impulse response (row vector)
%       These are the FIR filter coefficients
%
% Output:
%   y - Filtered signal (same length as x)
%       Uses 'same' mode to maintain input signal length
%
% How Convolution Works:
%   1. Flip the filter impulse response h[k]
%   2. Slide it across the input signal x[n]
%   3. At each position n, multiply overlapping samples
%   4. Sum the products to get output y[n]
%
% Why Convolution for Filtering:
%   - Convolution in time domain = multiplication in frequency domain
%   - Filter frequency response H(f) multiplies signal spectrum X(f)
%   - Result: Y(f) = H(f) * X(f)
%   - Inverse FFT gives filtered time signal y[n]

% Ensure correct dimensions for processing
% x must be column vector for consistent indexing
% h must be row vector for loop processing
x = x(:);  % Force column vector
h = h(:)';  % Force row vector

% Get signal and filter lengths
Nx = length(x);  % Number of input samples
Nh = length(h);  % Number of filter coefficients (taps)

% Calculate full convolution output length
% Full convolution produces Nx + Nh - 1 samples
% This is because the filter "slides" beyond signal boundaries
Ny_full = Nx + Nh - 1;

% Initialize full output vector with zeros
% Pre-allocation improves performance
y_full = zeros(Ny_full, 1);

% Perform discrete convolution using double loop
% Outer loop: iterate through each output sample position
for n = 1:Ny_full
    % Initialize accumulator for this output sample
    % This will store sum of products
    
    % Inner loop: iterate through filter coefficients
    for k = 1:Nh
        % Calculate input index for this coefficient
        % idx = n - k + 1 implements the time-reversal and shift
        idx = n - k + 1;
        
        % Check if input index is valid
        % Only process if index is within signal bounds
        if idx >= 1 && idx <= Nx
            % Multiply-accumulate operation
            % This is the core of convolution: h[k] * x[n-k]
            y_full(n) = y_full(n) + h(k) * x(idx);
        end
        % If idx is out of bounds, treat x[idx] as zero (zero-padding)
    end
end

% Trim output to same length as input (MATLAB 'same' mode)
% This removes the transient effects at start and end
%
% The delay compensates for filter group delay
% For linear phase FIR: group delay = (Nh-1)/2 samples
delay = floor(Nh / 2);

% Extract central portion of convolution result
% This aligns the filtered signal with the input signal
y = y_full(delay + 1 : delay + Nx);

% Note on computational complexity:
%   Time complexity: O(Nx * Nh)
%   For long signals or filters, FFT-based convolution is faster: O(Nx*log(Nx))
%   FFT convolution uses: y = ifft(fft(x) .* fft(h))

end