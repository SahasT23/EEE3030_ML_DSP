function y = custom_conv(x, h)
% CUSTOM_CONV Performs linear convolution of signal x with filter h
%   y = custom_conv(x, h)
%
% Inputs:
%   x - Input signal (column vector)
%   h - Filter impulse response (row vector)
%
% Output:
%   y - Filtered signal (same length as x, using 'same' mode)

% Ensure correct dimensions
x = x(:);  % Column vector
h = h(:)';  % Row vector

Nx = length(x);
Nh = length(h);

% Full convolution length
Ny_full = Nx + Nh - 1;

% Initialize output
y_full = zeros(Ny_full, 1);

% Perform convolution
for n = 1:Ny_full
    for k = 1:Nh
        idx = n - k + 1;
        if idx >= 1 && idx <= Nx
            y_full(n) = y_full(n) + h(k) * x(idx);
        end
    end
end

% Trim to same length as input (equivalent to 'same' mode)
delay = floor(Nh / 2);
y = y_full(delay + 1 : delay + Nx);

end