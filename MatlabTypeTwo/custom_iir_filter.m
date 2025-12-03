function y = custom_iir_filter(b, a, x)
% CUSTOM_IIR_FILTER Implements IIR filter using Direct Form II structure
%
% Purpose:
%   Performs infinite impulse response (IIR) filtering
%   Uses recursive difference equation with feedback
%
% Difference Equation:
%   a(1)*y[n] = b(1)*x[n] + b(2)*x[n-1] + ... + b(nb)*x[n-nb+1]
%             - a(2)*y[n-1] - ... - a(na)*y[n-na+1]
%
% Inputs:
%   b - Numerator coefficients (feedforward path)
%       These coefficients process the input signal
%       Equivalent to zeros in z-domain
%   a - Denominator coefficients (feedback path)
%       These coefficients use previous outputs
%       Equivalent to poles in z-domain
%       Note: a(1) should be 1 (normalized form)
%   x - Input signal (column vector)
%       Signal to be filtered
%
% Output:
%   y - Filtered signal (same length as x)
%       Result of applying IIR filter to input
%
% IIR vs FIR:
%   FIR: y[n] = sum(b[k]*x[n-k])           (no feedback, finite response)
%   IIR: y[n] = sum(b[k]*x[n-k]) - sum(a[k]*y[n-k])  (feedback, infinite response)
%
% Advantages of IIR:
%   - Fewer coefficients needed for sharp cutoff
%   - More efficient (less computation per sample)
%   - Can match analog filter responses (Butterworth, Chebyshev)
%
% Disadvantages of IIR:
%   - Nonlinear phase response (introduces phase distortion)
%   - Can become unstable if poles outside unit circle
%   - More sensitive to coefficient quantization
%
% Direct Form II Structure:
%   Most memory-efficient IIR structure
%   Uses single delay line for both input and output samples
%   Minimizes number of delay elements needed

% Ensure correct dimensions for processing
x = x(:);   % Force input to column vector
b = b(:)';  % Force numerator coefficients to row vector
a = a(:)';  % Force denominator coefficients to row vector

% Normalize coefficients by a(1)
% Standard form requires a(1) = 1
if a(1) ~= 1
    % If a(1) is not 1, divide all coefficients by a(1)
    b = b / a(1);
    a = a / a(1);
end

% Get filter orders
nb = length(b);  % Number of feedforward coefficients
na = length(a);  % Number of feedback coefficients
nx = length(x);  % Number of input samples

% Initialize output vector
% Pre-allocation improves performance
y = zeros(nx, 1);

% Initialize state buffers for past samples
% These buffers store previous values needed for recursion
x_buffer = zeros(nb, 1);  % Past input samples
y_buffer = zeros(na, 1);  % Past output samples

% Process each input sample sequentially
% IIR filtering must be done sample-by-sample due to feedback
for n = 1:nx
    % Update input buffer with new sample
    % Shift buffer right (oldest sample falls off)
    % Insert new sample at position 1 (most recent)
    x_buffer = [x(n); x_buffer(1:end-1)];
    
    % Compute feedforward (FIR) part
    % This processes current and past inputs
    % Implements: sum(b[k] * x[n-k]) for k = 0 to nb-1
    y_current = sum(b .* x_buffer');
    
    % Compute feedback (recursive) part
    % This uses previous outputs to influence current output
    % Implements: -sum(a[k] * y[n-k]) for k = 1 to na-1
    % Note: Skip a(1) because it equals 1 (normalized)
    if na > 1
        % Multiply feedback coefficients (a(2:end)) with past outputs
        % Subtract this from current output
        y_current = y_current - sum(a(2:end) .* y_buffer(1:na-1)');
    end
    
    % Store computed output sample
    y(n) = y_current;
    
    % Update output buffer with new output sample
    % Shift buffer right (oldest output falls off)
    % Insert new output at position 1 (most recent)
    y_buffer = [y_current; y_buffer(1:end-1)];
end

% Display filter information for debugging
fprintf('Custom IIR filter: Processed %d samples\n', nx);

% Note on stability:
%   IIR filter is stable if all poles are inside unit circle: |z| < 1
%   Poles are roots of denominator polynomial A(z)
%   Check stability: max(abs(roots(a))) < 1

% Note on computational complexity:
%   Time complexity per sample: O(nb + na)
%   Total complexity: O(nx * (nb + na))
%   Much more efficient than FIR for equivalent frequency response

end