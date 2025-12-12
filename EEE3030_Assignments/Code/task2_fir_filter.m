function [x_filtered, h_bp, fir_params] = task2_fir_filter(signal, params)
% TASK2_FIR_FILTER - Design and apply FIR bandpass filter using IRT method
%
% Usage:
%   [x_filtered, h_bp, fir_params] = task2_fir_filter(signal, params)
%
% Inputs:
%   signal - Structure from task1_signal_analysis containing signal data
%   params - Structure from task1_signal_analysis containing parameters
%
% Outputs:
%   x_filtered - Bandpass filtered signal
%   h_bp       - FIR filter coefficients
%   fir_params - Structure containing filter design parameters
%
% Filter specifications:
%   - Passband: fmin to fmax (from Task 1)
%   - Stopband: fmin-2kHz and fmax+2kHz
%   - Max passband ripple: 0.1 dB
%   - Min stopband attenuation: 50 dB
%   - Design method: Impulse Response Truncation with Hamming window
%
% EEE3030 DSP Assignment - Task 2

%% Extract parameters
fs = params.fs;
fmin = params.fmin;
fmax = params.fmax;
x = signal.x;

%% Define filter specifications
fstop_lower = fmin - 2000;   % Lower stopband edge (Hz)
fstop_upper = fmax + 2000;   % Upper stopband edge (Hz)
transition_bandwidth = 2000;  % Hz

% Cutoff frequencies at centre of transition bands
Fc1 = (fmin - 1000) / fs;    % Lower cutoff (normalised)
Fc2 = (fmax + 1000) / fs;    % Upper cutoff (normalised)

% Normalised transition width
delta_F = transition_bandwidth / fs;

fprintf('FIR BANDPASS FILTER SPECIFICATIONS:\n');
fprintf('  Lower stopband edge:     %d Hz\n', fstop_lower);
fprintf('  Lower passband edge:     %d Hz\n', fmin);
fprintf('  Upper passband edge:     %d Hz\n', fmax);
fprintf('  Upper stopband edge:     %d Hz\n', fstop_upper);
fprintf('  Transition bandwidth:    %d Hz\n', transition_bandwidth);
fprintf('  Normalised Fc1:          %.6f\n', Fc1);
fprintf('  Normalised Fc2:          %.6f\n', Fc2);

%% Calculate required filter length
% Using Hamming window: transition width = 3.3/N
N_calculated = 3.3 / delta_F;
N_fir = ceil(N_calculated);

% Ensure N is odd for symmetric filter
if mod(N_fir, 2) == 0
    N_fir = N_fir + 1;
end

M = (N_fir - 1) / 2;

fprintf('\nFILTER LENGTH CALCULATION:\n');
fprintf('  Window function:         Hamming\n');
fprintf('  Calculated N:            %.2f\n', N_calculated);
fprintf('  Rounded N (odd):         %d\n', N_fir);
fprintf('  M (half-length):         %d\n', M);

%% Design the ideal bandpass impulse response
% h_BP[n] = 2*Fc2*sinc(2*Fc2*n) - 2*Fc1*sinc(2*Fc1*n)

n_ideal = -M:M;
h_ideal = zeros(1, N_fir);

for i = 1:N_fir
    n = n_ideal(i);
    if n == 0
        % For n = 0: h[0] = 2*Fc2 - 2*Fc1
        h_ideal(i) = 2*Fc2 - 2*Fc1;
    else
        % For n != 0
        term1 = 2*Fc2 * sin(n * 2*pi*Fc2) / (n * 2*pi*Fc2);
        term2 = 2*Fc1 * sin(n * 2*pi*Fc1) / (n * 2*pi*Fc1);
        h_ideal(i) = term1 - term2;
    end
end

%% Generate Hamming window
n_window = 0:N_fir-1;
hamming_win = 0.54 - 0.46 * cos(2 * pi * n_window / (N_fir - 1));

%% Apply window to ideal impulse response
h_bp = h_ideal .* hamming_win;

fprintf('\nFILTER COEFFICIENTS:\n');
fprintf('  Centre coefficient:      %.6f\n', h_bp(M+1));
fprintf('  Sum of coefficients:     %.6f\n', sum(h_bp));

%% Plot filter design
figure('Name', 'Task 2 - FIR Filter Design', 'Position', [100, 100, 1200, 800]);

subplot(3,1,1);
stem(n_ideal, h_ideal, 'b', 'LineWidth', 0.5, 'MarkerSize', 3);
xlabel('Sample Index (n)');
ylabel('Amplitude');
title('Ideal Bandpass Impulse Response (Unwindowed)');
grid on;
xlim([-M-5, M+5]);

subplot(3,1,2);
stem(n_ideal, hamming_win, 'r', 'LineWidth', 0.5, 'MarkerSize', 3);
xlabel('Sample Index (n)');
ylabel('Amplitude');
title('Hamming Window Function');
grid on;
xlim([-M-5, M+5]);
ylim([0, 1.1]);

subplot(3,1,3);
stem(n_ideal, h_bp, 'g', 'LineWidth', 0.5, 'MarkerSize', 3);
xlabel('Sample Index (n)');
ylabel('Amplitude');
title('Final FIR Filter Coefficients (Windowed)');
grid on;
xlim([-M-5, M+5]);

%% Verify frequency response
N_fft = 8192;
H = fft(h_bp, N_fft);
H_magnitude = abs(H);
H_dB = 20 * log10(H_magnitude + eps);
H_phase = unwrap(angle(H));

f_response = (0:N_fft/2) * fs / N_fft;
H_dB_single = H_dB(1:N_fft/2+1);

figure('Name', 'Task 2 - Filter Frequency Response', 'Position', [100, 100, 1200, 600]);

subplot(2,1,1);
plot(f_response/1000, H_dB_single, 'b', 'LineWidth', 1);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('FIR Bandpass Filter - Magnitude Response');
grid on;
xlim([0, fs/2000]);
ylim([-100, 10]);

hold on;
xline(fmin/1000, 'g--', 'LineWidth', 1.5);
xline(fmax/1000, 'g--', 'LineWidth', 1.5);
xline(fstop_lower/1000, 'r--', 'LineWidth', 1.5);
xline(fstop_upper/1000, 'r--', 'LineWidth', 1.5);
yline(-50, 'm--', 'LineWidth', 1.5);
legend('Response', 'f_{min}', 'f_{max}', 'f_{stop,lower}', 'f_{stop,upper}', '-50 dB spec', 'Location', 'southwest');
hold off;

subplot(2,1,2);
plot(f_response/1000, H_dB_single, 'b', 'LineWidth', 1);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Passband Detail');
grid on;
xlim([(fmin-3000)/1000, (fmax+3000)/1000]);
ylim([-80, 5]);

%% Measure filter performance
passband_indices = find(f_response >= fmin & f_response <= fmax);
stopband_lower_indices = find(f_response <= fstop_lower);
stopband_upper_indices = find(f_response >= fstop_upper & f_response <= fs/2);

passband_gain_dB = H_dB_single(passband_indices);
passband_ripple = max(passband_gain_dB) - min(passband_gain_dB);

stopband_lower_max = max(H_dB_single(stopband_lower_indices));
stopband_upper_max = max(H_dB_single(stopband_upper_indices));
stopband_max = max(stopband_lower_max, stopband_upper_max);

fprintf('\nFILTER VERIFICATION:\n');
fprintf('  Passband ripple:         %.4f dB (spec: < 0.1 dB)\n', passband_ripple);
fprintf('  Stopband attenuation:    %.2f dB (spec: > 50 dB)\n', -stopband_max);

if passband_ripple < 0.1
    fprintf('  Passband:                PASS\n');
else
    fprintf('  Passband:                FAIL\n');
end

if stopband_max < -50
    fprintf('  Stopband:                PASS\n');
else
    fprintf('  Stopband:                FAIL\n');
end

%% Verify custom convolution
fprintf('\nCUSTOM CONVOLUTION VERIFICATION:\n');
test_signal = randn(1, 1000);
test_filter = ones(1, 10) / 10;

y_custom = custom_conv(test_signal, test_filter);
y_matlab = conv(test_signal, test_filter, 'same');

max_error = max(abs(y_custom - y_matlab));
fprintf('  Max error vs MATLAB:     %.2e\n', max_error);

if max_error < 1e-10
    fprintf('  Status:                  PASS\n');
else
    fprintf('  Status:                  CHECK IMPLEMENTATION\n');
end

%% Apply bandpass filter to AM signal
fprintf('\nAPPLYING BANDPASS FILTER:\n');

tic;
x_filtered = custom_conv(x, h_bp);
filter_time = toc;

fprintf('  Time taken:              %.4f seconds\n', filter_time);
fprintf('  Input length:            %d samples\n', length(x));
fprintf('  Output length:           %d samples\n', length(x_filtered));

%% Plot filtering results
figure('Name', 'Task 2 - Filtering Results', 'Position', [100, 100, 1200, 700]);

subplot(2,1,1);
plot(signal.t, x, 'b', 'LineWidth', 0.3);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Original Signal (Before Bandpass Filtering)');
grid on;
xlim([0, params.duration]);

subplot(2,1,2);
plot(signal.t, x_filtered, 'r', 'LineWidth', 0.3);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Filtered Signal (After Bandpass Filtering)');
grid on;
xlim([0, params.duration]);

%% Frequency domain comparison
x_filtered_windowed = x_filtered .* signal.hamming_window;
X_filtered = fft(x_filtered_windowed);
X_filtered_magnitude = abs(X_filtered);
X_filtered_normalised = X_filtered_magnitude / params.N / signal.CG_hamming;
X_filtered_single = X_filtered_normalised(1:params.num_bins);
X_filtered_single(2:end-1) = 2 * X_filtered_single(2:end-1);
X_filtered_dB = 20 * log10(X_filtered_single + eps);

figure('Name', 'Task 2 - Frequency Domain Comparison', 'Position', [100, 100, 1200, 700]);

subplot(2,1,1);
plot(signal.f/1000, signal.X_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Original Signal Spectrum');
grid on;
xlim([0, fs/2000]);
ylim([-120, max(signal.X_dB) + 10]);

subplot(2,1,2);
plot(signal.f/1000, X_filtered_dB, 'r', 'LineWidth', 0.3);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Filtered Signal Spectrum');
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_filtered_dB) + 10]);

%% Calculate SNR improvement
passband_idx = find(signal.f >= fmin & signal.f <= fmax);
stopband_idx = [find(signal.f <= fstop_lower), find(signal.f >= fstop_upper & signal.f <= fs/2)];

passband_power_before = mean(signal.X_single(passband_idx).^2);
passband_power_after = mean(X_filtered_single(passband_idx).^2);
stopband_power_before = mean(signal.X_single(stopband_idx).^2);
stopband_power_after = mean(X_filtered_single(stopband_idx).^2);

snr_before = 10 * log10(passband_power_before / stopband_power_before);
snr_after = 10 * log10(passband_power_after / stopband_power_after);
snr_improvement = snr_after - snr_before;

fprintf('\nSNR IMPROVEMENT:\n');
fprintf('  SNR before filtering:    %.2f dB\n', snr_before);
fprintf('  SNR after filtering:     %.2f dB\n', snr_after);
fprintf('  Improvement:             %.2f dB\n', snr_improvement);

%% Prepare output structure
fir_params.N_fir = N_fir;
fir_params.M = M;
fir_params.Fc1 = Fc1;
fir_params.Fc2 = Fc2;
fir_params.passband_ripple = passband_ripple;
fir_params.stopband_attenuation = -stopband_max;
fir_params.snr_improvement = snr_improvement;
fir_params.X_filtered_dB = X_filtered_dB;
fir_params.X_filtered_single = X_filtered_single;

fprintf('\nTask 2 complete. Filtered signal ready for Task 3.\n');

end