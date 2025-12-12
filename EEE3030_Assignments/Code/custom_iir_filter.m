function y = custom_iir_filter(b, a, x)
% CUSTOM_IIR_FILTER - Implements IIR filtering using difference equation
%
% This function filters the input signal x using the IIR filter defined
% by numerator coefficients b and denominator coefficients a.
%
% USAGE:
%   y = custom_iir_filter(b, a, x)
%
% INPUTS:
%   b - Numerator coefficients (feedforward), row or column vector
%   a - Denominator coefficients (feedback), row or column vector
%       Note: a(1) should be 1 (normalised form)
%   x - Input signal, row or column vector
%
% OUTPUT:
%   y - Filtered output signal, same size as x

%% Input validation and preparation

% Store original orientation
is_column = iscolumn(x);

% Convert all inputs to row vectors for consistent processing
b = b(:).';  % Ensure row vector
a = a(:).';  % Ensure row vector
x = x(:).';  % Ensure row vector

% Get lengths
L = length(x);      % Signal length
M = length(b) - 1;  % Numerator order (number of b coefficients minus 1)
N = length(a) - 1;  % Denominator order (number of a coefficients minus 1)

%% Normalise coefficients if a(1) is not 1

if a(1) ~= 1
    b = b / a(1);
    a = a / a(1);
end

%% Initialise buffers

% Input buffer: stores past M input samples
% x_buffer(1) = x[n], x_buffer(2) = x[n-1], ..., x_buffer(M+1) = x[n-M]
x_buffer = zeros(1, M + 1);

% Output buffer: stores past N output samples
% y_buffer(1) = y[n-1], y_buffer(2) = y[n-2], ..., y_buffer(N) = y[n-N]
y_buffer = zeros(1, N);

%% Preallocate output array

y = zeros(1, L);

%% Main filtering loop

for n = 1:L
    
    % Shift input buffer to the right (make room for new sample)
    % x_buffer = [x[n], x[n-1], x[n-2], ..., x[n-M]]
    for k = M+1:-1:2
        x_buffer(k) = x_buffer(k-1);
    end
    
    % Insert current input sample at the beginning
    x_buffer(1) = x(n);
    
    % Calculate feedforward (FIR) part: sum of b(k) * x[n-k+1]
    % This computes: b(1)*x[n] + b(2)*x[n-1] + ... + b(M+1)*x[n-M]
    feedforward_sum = 0;
    for k = 1:M+1
        feedforward_sum = feedforward_sum + b(k) * x_buffer(k);
    end
    
    % Calculate feedback (recursive) part: sum of a(k) * y[n-k+1]
    % This computes: a(2)*y[n-1] + a(3)*y[n-2] + ... + a(N+1)*y[n-N]
    % Note: we start from a(2) because a(1) = 1 and multiplies y[n]
    feedback_sum = 0;
    for k = 1:N
        feedback_sum = feedback_sum + a(k+1) * y_buffer(k);
    end
    
    % Compute current output: y[n] = feedforward - feedback
    y(n) = feedforward_sum - feedback_sum;
    
    % Shift output buffer to the right (make room for new output)
    % y_buffer = [y[n-1], y[n-2], ..., y[n-N]]
    for k = N:-1:2
        y_buffer(k) = y_buffer(k-1);
    end
    
    % Insert current output at the beginning (becomes y[n-1] for next iteration)
    if N >= 1
        y_buffer(1) = y(n);
    end
    
end

%% Restore original orientation if input was column vector

if is_column
    y = y(:);
end

end