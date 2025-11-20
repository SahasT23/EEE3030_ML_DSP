function y = custom_iir_filter(b, a, x)
% CUSTOM_IIR_FILTER Implements IIR filter using Direct Form II structure
%   y = custom_iir_filter(b, a, x)
%
% Inputs:
%   b - Numerator coefficients (feedforward)
%   a - Denominator coefficients (feedback), a(1) should be 1
%   x - Input signal (column vector)
%
% Output:
%   y - Filtered signal (same length as x)
%
% Implements difference equation:
%   a(1)*y[n] = b(1)*x[n] + b(2)*x[n-1] + ... + b(nb)*x[n-nb+1]
%             - a(2)*y[n-1] - ... - a(na)*y[n-na+1]

% Ensure correct dimensions
x = x(:);  % Column vector
b = b(:)';  % Row vector
a = a(:)';  % Row vector

% Normalize by a(1)
if a(1) ~= 1
    b = b / a(1);
    a = a / a(1);
end

% Get filter orders
nb = length(b);  % Number of feedforward coefficients
na = length(a);  % Number of feedback coefficients
nx = length(x);  % Signal length

% Initialize output
y = zeros(nx, 1);

% Initialize state buffers (past inputs and outputs)
x_buffer = zeros(nb, 1);  % Past inputs
y_buffer = zeros(na, 1);  % Past outputs

% Process each sample
for n = 1:nx
    % Shift buffers (most recent at index 1)
    x_buffer = [x(n); x_buffer(1:end-1)];
    
    % Feedforward: compute output from inputs (numerator)
    y_current = sum(b .* x_buffer');
    
    % Feedback: subtract past outputs (denominator, skip a(1))
    if na > 1
        y_current = y_current - sum(a(2:end) .* y_buffer(1:na-1)');
    end
    
    % Store output
    y(n) = y_current;
    
    % Update output buffer
    y_buffer = [y_current; y_buffer(1:end-1)];
end

fprintf('Custom IIR filter: Processed %d samples\n', nx);

end