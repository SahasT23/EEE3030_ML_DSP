function [x_demodulated, iir_params] = task4_iir_filter(x_mixed, params)
% TASK4_IIR_FILTER - Design and apply IIR lowpass filter
%
% Usage:
%   [x_demodulated, iir_params] = task4_iir_filter(x_mixed, params)
%
% Inputs:
%   x_mixed - Mixed signal from Task 3
%   params  - Parameters structure from Task 1
%
% Outputs:
%   x_demodulated - Lowpass filtered (demodulated) signal
%   iir_params    - Structure containing:
%                   .b_iir     - Numerator coefficients
%                   .a_iir     - Denominator coefficients
%                   .fc_lowpass - Cutoff frequency
%                   .order     - Filter order
%
% Filter specifications:
%   - Type: Butterworth (maximally flat)
%   - Order: 4
%   - Cutoff frequency: 4 kHz
%
% EEE3030 DSP Assignment - Task 4

%% Extract parameters
fs = params.fs;

%% Filter specifications
filter_order = 4;
fc_lowpass = 4000;  % Cutoff frequency in Hz

% Normalised cutoff (MATLAB uses Nyquist-normalised frequency)
Wn = fc_lowpass / (fs/2);

fprintf('IIR LOWPASS FILTER SPECIFICATIONS:\n');
fprintf('  Filter type:             Butterworth\n');
fprintf('  Filter order:            %d\n', filter_order);
fprintf('  Cutoff frequency:        %d Hz\n', fc_lowpass);
fprintf('  Normalised cutoff (Wn):  %.6f\n', Wn);

%% Design Butterworth filter using bilinear transform
[b_iir, a_iir] = butter(filter_order, Wn, 'low');

fprintf('\nFILTER COEFFICIENTS:\n');
fprintf('Numerator (b):\n');
for i = 1:length(b_iir)
    fprintf('  b[%d] = %.10f\n', i-1, b_iir(i));
end
fprintf('Denominator (a):\n');
for i = 1:length(a_iir)
    fprintf('  a[%d] = %.10f\n', i-1, a_iir(i));
end

fprintf('\nCOEFFICIENT PROPERTIES:\n');
fprintf('  Sum of b coefficients:   %.10f\n', sum(b_iir));
fprintf('  Sum of a coefficients:   %.10f\n', sum(a_iir));
fprintf('  DC gain H(z=1):          %.6f\n', sum(b_iir)/sum(a_iir));

%% Plot coefficients
figure('Name', 'Task 4 - IIR Coefficients', 'Position', [100, 100, 1000, 400]);

subplot(1,2,1);
stem(0:length(b_iir)-1, b_iir, 'b', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Coefficient Index');
ylabel('Value');
title('Numerator Coefficients (b)');
grid on;

subplot(1,2,2);
stem(0:length(a_iir)-1, a_iir, 'r', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Coefficient Index');
ylabel('Value');
title('Denominator Coefficients (a)');
grid on;

%% Verify frequency response
N_freq = 8192;
[H_iir, f_iir] = freqz(b_iir, a_iir, N_freq, fs);

H_iir_magnitude = abs(H_iir);
H_iir_dB = 20 * log10(H_iir_magnitude + eps);
H_iir_phase = unwrap(angle(H_iir));

figure('Name', 'Task 4 - IIR Frequency Response', 'Position', [100, 100, 1200, 700]);

subplot(2,2,1);
plot(f_iir/1000, H_iir_dB, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Magnitude Response (Full Range)');
grid on;
xlim([0, fs/2000]);
ylim([-100, 5]);

hold on;
xline(fc_lowpass/1000, 'r--', 'LineWidth', 1.5);
yline(-3, 'g--', 'LineWidth', 1.5);
legend('Response', 'f_c = 4 kHz', '-3 dB', 'Location', 'southwest');
hold off;

subplot(2,2,2);
plot(f_iir/1000, H_iir_dB, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Passband Detail');
grid on;
xlim([0, 8]);
ylim([-10, 2]);

hold on;
xline(fc_lowpass/1000, 'r--', 'LineWidth', 1.5);
yline(-3, 'g--', 'LineWidth', 1.5);
hold off;

subplot(2,2,3);
plot(f_iir/1000, H_iir_phase * 180/pi, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)');
ylabel('Phase (degrees)');
title('Phase Response');
grid on;
xlim([0, 10]);

subplot(2,2,4);
plot(f_iir/1000, H_iir_magnitude, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)');
ylabel('Magnitude (linear)');
title('Linear Magnitude Response');
grid on;
xlim([0, 20]);
ylim([0, 1.1]);

hold on;
xline(fc_lowpass/1000, 'r--', 'LineWidth', 1.5);
yline(0.707, 'g--', 'LineWidth', 1.5);
legend('Response', 'f_c', '0.707 (-3 dB)', 'Location', 'northeast');
hold off;

%% Verify -3 dB point
idx_3dB = find(H_iir_dB <= -3, 1, 'first');
if ~isempty(idx_3dB)
    f_3dB_actual = f_iir(idx_3dB);
else
    f_3dB_actual = NaN;
end

fprintf('\nFREQUENCY RESPONSE VERIFICATION:\n');
fprintf('  Specified cutoff:        %d Hz\n', fc_lowpass);
fprintf('  Measured -3 dB point:    %.2f Hz\n', f_3dB_actual);

%% Stability analysis (pole-zero plot)
zeros_iir = roots(b_iir);
poles_iir = roots(a_iir);
pole_magnitudes = abs(poles_iir);
max_pole_magnitude = max(pole_magnitudes);

figure('Name', 'Task 4 - Pole-Zero Plot', 'Position', [100, 100, 600, 600]);

theta_circle = linspace(0, 2*pi, 100);
plot(cos(theta_circle), sin(theta_circle), 'k--', 'LineWidth', 1);
hold on;
plot(real(zeros_iir), imag(zeros_iir), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
plot(real(poles_iir), imag(poles_iir), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

xlabel('Real Part');
ylabel('Imaginary Part');
title('Pole-Zero Plot');
legend('Unit Circle', 'Zeros', 'Poles', 'Location', 'best');
grid on;
axis equal;
xlim([-1.5, 1.5]);
ylim([-1.5, 1.5]);

fprintf('\nSTABILITY ANALYSIS:\n');
fprintf('  Max pole magnitude:      %.6f\n', max_pole_magnitude);
if max_pole_magnitude < 1
    fprintf('  Status:                  STABLE (all poles inside unit circle)\n');
else
    fprintf('  Status:                  UNSTABLE\n');
end

%% Verify custom IIR filter implementation
fprintf('\nCUSTOM IIR FILTER VERIFICATION:\n');

test_length = 10000;
test_signal = x_mixed(1:test_length);

y_matlab = filter(b_iir, a_iir, test_signal);
y_custom = custom_iir_filter(b_iir, a_iir, test_signal);

error_iir = y_matlab - y_custom;
max_error = max(abs(error_iir));

fprintf('  Test length:             %d samples\n', test_length);
fprintf('  Max error vs MATLAB:     %.2e\n', max_error);

if max_error < 1e-10
    fprintf('  Status:                  PASS\n');
else
    fprintf('  Status:                  CHECK IMPLEMENTATION\n');
end

%% Apply lowpass filter using custom implementation
fprintf('\nAPPLYING IIR LOWPASS FILTER:\n');

tic;
x_demodulated = custom_iir_filter(b_iir, a_iir, x_mixed);
time_filter = toc;

fprintf('  Time taken:              %.2f seconds\n', time_filter);
fprintf('  Input length:            %d samples\n', length(x_mixed));
fprintf('  Output length:           %d samples\n', length(x_demodulated));

%% Time domain comparison
t = (0:length(x_mixed)-1) / fs;

figure('Name', 'Task 4 - Filtering Results (Time)', 'Position', [100, 100, 1200, 800]);

subplot(3,1,1);
plot(t, x_mixed, 'b', 'LineWidth', 0.5);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Mixed Signal (Input to Lowpass Filter)');
grid on;

subplot(3,1,2);
plot(t, x_demodulated, 'r', 'LineWidth', 0.5);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Demodulated Signal (Output of Lowpass Filter)');
grid on;

subplot(3,1,3);
plot_duration = 0.05;
plot_samples = round(plot_duration * fs);
plot_start = round(length(t) / 2);
plot_end = min(plot_start + plot_samples, length(t));
t_portion = t(plot_start:plot_end);

plot(t_portion, x_mixed(plot_start:plot_end), 'b', 'LineWidth', 0.5);
hold on;
plot(t_portion, x_demodulated(plot_start:plot_end), 'r', 'LineWidth', 1.5);
hold off;
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Comparison (50 ms Portion)');
legend('Mixed', 'Demodulated', 'Location', 'best');
grid on;

%% Frequency domain analysis
hamming_window = hamming(length(x_demodulated))';
x_demod_windowed = x_demodulated .* hamming_window;
X_demod = fft(x_demod_windowed);
X_demod_magnitude = abs(X_demod) / length(X_demod);

N_demod = length(X_demod);
X_demod_single = X_demod_magnitude(1:floor(N_demod/2)+1);
X_demod_single(2:end-1) = 2 * X_demod_single(2:end-1);
f_demod = (0:floor(N_demod/2)) * fs / N_demod;
X_demod_dB = 20 * log10(X_demod_single + eps);

figure('Name', 'Task 4 - Filtering Results (Frequency)', 'Position', [100, 100, 1200, 500]);

plot(f_demod/1000, X_demod_dB, 'r', 'LineWidth', 1);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Demodulated Signal Spectrum');
grid on;
xlim([0, 10]);

hold on;
xline(fc_lowpass/1000, 'g--', 'LineWidth', 1.5);
legend('Spectrum', 'Lowpass cutoff (4 kHz)', 'Location', 'northeast');
hold off;

%% Calculate filtering effectiveness
baseband_idx = find(f_demod <= fc_lowpass);
stopband_idx = find(f_demod > fc_lowpass);

baseband_power = sum(X_demod_single(baseband_idx).^2);
stopband_power = sum(X_demod_single(stopband_idx).^2);

fprintf('\nFILTERING EFFECTIVENESS:\n');
fprintf('  Baseband power (0-4 kHz):   %.6e\n', baseband_power);
fprintf('  Stopband power (>4 kHz):    %.6e\n', stopband_power);
fprintf('  Ratio (dB):                 %.2f dB\n', 10*log10(baseband_power/stopband_power));

%% Prepare output structure
iir_params.b_iir = b_iir;
iir_params.a_iir = a_iir;
iir_params.fc_lowpass = fc_lowpass;
iir_params.order = filter_order;
iir_params.max_pole_magnitude = max_pole_magnitude;
iir_params.f_3dB = f_3dB_actual;
iir_params.X_demod_dB = X_demod_dB;
iir_params.f_demod = f_demod;

fprintf('\nTask 4 complete. Demodulated signal ready for Task 5.\n');

end