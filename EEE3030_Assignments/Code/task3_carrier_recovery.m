function [x_mixed, carrier_params] = task3_carrier_recovery(x_filtered, params)
% TASK3_CARRIER_RECOVERY - Recover carrier frequency and mix to baseband
%
% Usage:
%   [x_mixed, carrier_params] = task3_carrier_recovery(x_filtered, params)
%
% Inputs:
%   x_filtered - Bandpass filtered signal from Task 2
%   params     - Parameters structure from Task 1
%
% Outputs:
%   x_mixed       - Mixed signal (filtered AM × carrier)
%   carrier_params - Structure containing:
%                    .fc       - Recovered carrier frequency
%                    .phi      - Initial phase (0, to be optimised in Task 5)
%
% Method:
%   1. Apply square law to generate 2fc component
%   2. Find peak at 2fc using FFT
%   3. Generate local carrier and mix with filtered signal
%
% EEE3030 DSP Assignment - Task 3

%% Extract parameters
fs = params.fs;
N = params.N;
t = (0:N-1) / fs;
fc_initial = params.fc;  % Initial estimate from Task 1

%% Apply square law
fprintf('Applying square law to recover carrier...\n');
x_squared = x_filtered .^ 2;

%% Compute spectrum of squared signal
hamming_window = 0.54 - 0.46 * cos(2 * pi * (0:N-1) / (N - 1));
CG_hamming = sum(hamming_window) / N;

x_squared_windowed = x_squared .* hamming_window;
X_squared = fft(x_squared_windowed);
X_squared_magnitude = abs(X_squared);
X_squared_normalised = X_squared_magnitude / N / CG_hamming;

num_bins = floor(N/2) + 1;
X_squared_single = X_squared_normalised(1:num_bins);
X_squared_single(2:end-1) = 2 * X_squared_single(2:end-1);
X_squared_dB = 20 * log10(X_squared_single + eps);

f = (0:num_bins-1) * fs / N;

%% Find peak at 2fc
search_range_low = 2*fc_initial - 5000;
search_range_high = 2*fc_initial + 5000;
search_indices = find(f >= search_range_low & f <= search_range_high);

X_search = X_squared_single(search_indices);
f_search = f(search_indices);

[peaks, locs] = findpeaks(X_search, f_search);
[peak_value, max_idx] = max(peaks);
f_2fc_measured = locs(max_idx);

% Calculate carrier frequency
fc_measured = f_2fc_measured / 2;
fc_final = round(fc_measured / 1000) * 1000;

fprintf('\nCARRIER FREQUENCY DETECTION:\n');
fprintf('  Search range:            %d - %d Hz\n', search_range_low, search_range_high);
fprintf('  Peak found at 2fc:       %.2f Hz\n', f_2fc_measured);
fprintf('  Calculated fc:           %.2f Hz\n', fc_measured);
fprintf('  Rounded fc:              %d Hz\n', fc_final);

if fc_final == fc_initial
    fprintf('  Status:                  CONFIRMED (matches Task 1)\n');
else
    fprintf('  Status:                  UPDATED (differs from Task 1)\n');
end

%% Plot squared signal spectrum
figure('Name', 'Task 3 - Carrier Recovery', 'Position', [100, 100, 1200, 600]);

subplot(2,1,1);
plot(f/1000, X_squared_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Spectrum of Squared Signal (Full Range)');
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_squared_dB) + 10]);

hold on;
xline(2*fc_final/1000, 'r--', 'LineWidth', 2);
plot(f_2fc_measured/1000, 20*log10(peak_value), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
legend('Spectrum', 'Expected 2f_c', 'Detected 2f_c Peak', 'Location', 'northeast');
hold off;

subplot(2,1,2);
plot(f/1000, X_squared_dB, 'b', 'LineWidth', 0.5);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Spectrum of Squared Signal (Zoomed Around 2f_c)');
grid on;
xlim([(2*fc_final - 15000)/1000, (2*fc_final + 15000)/1000]);
ylim([-80, max(X_squared_dB) + 10]);

hold on;
xline(2*fc_final/1000, 'r--', 'LineWidth', 2);
plot(f_2fc_measured/1000, 20*log10(peak_value), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
legend('Spectrum', 'Expected 2f_c', 'Detected 2f_c Peak', 'Location', 'northeast');
hold off;

%% Generate carrier and mix
phi = 0;  % Initial phase (will be optimised in Task 5)

carrier = cos(2 * pi * fc_final * t + phi);
x_mixed = x_filtered .* carrier;

fprintf('\nCARRIER GENERATION AND MIXING:\n');
fprintf('  Carrier frequency:       %d Hz\n', fc_final);
fprintf('  Initial phase:           %.4f rad (%.2f deg)\n', phi, phi*180/pi);

%% Time domain plots
figure('Name', 'Task 3 - Mixing Process', 'Position', [100, 100, 1200, 800]);

subplot(3,1,1);
plot(t, x_filtered, 'b', 'LineWidth', 0.3);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Bandpass Filtered AM Signal (Input to Mixer)');
grid on;
xlim([0, params.duration]);

subplot(3,1,2);
plot(t, carrier, 'g', 'LineWidth', 0.3);
xlabel('Time (seconds)');
ylabel('Amplitude');
title(sprintf('Local Carrier Signal: cos(2\\pi \\cdot %d \\cdot t)', fc_final));
grid on;
xlim([0, params.duration]);

subplot(3,1,3);
plot(t, x_mixed, 'r', 'LineWidth', 0.3);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Mixed Signal (Filtered AM × Carrier)');
grid on;
xlim([0, params.duration]);

%% Frequency domain analysis of mixed signal
x_mixed_windowed = x_mixed .* hamming_window;
X_mixed = fft(x_mixed_windowed);
X_mixed_magnitude = abs(X_mixed);
X_mixed_normalised = X_mixed_magnitude / N / CG_hamming;
X_mixed_single = X_mixed_normalised(1:num_bins);
X_mixed_single(2:end-1) = 2 * X_mixed_single(2:end-1);
X_mixed_dB = 20 * log10(X_mixed_single + eps);

figure('Name', 'Task 3 - Mixed Signal Spectrum', 'Position', [100, 100, 1200, 500]);

plot(f/1000, X_mixed_dB, 'r', 'LineWidth', 0.3);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Spectrum After Mixing (Shows Baseband and 2f_c Components)');
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_mixed_dB) + 10]);

hold on;
xline(0, 'g--', 'LineWidth', 1.5);
xline(2*fc_final/1000, 'm--', 'LineWidth', 1.5);
legend('Spectrum', 'Baseband (0 Hz)', '2f_c', 'Location', 'northeast');
hold off;

%% Analyse frequency components
baseband_indices = find(f >= 0 & f <= 4000);
baseband_power = mean(X_mixed_single(baseband_indices).^2);

double_fc_indices = find(f >= (2*fc_final - 4000) & f <= (2*fc_final + 4000));
double_fc_power = mean(X_mixed_single(double_fc_indices).^2);

fprintf('\nMIXED SIGNAL ANALYSIS:\n');
fprintf('  Baseband power (0-4 kHz):   %.6e\n', baseband_power);
fprintf('  2f_c power:                 %.6e\n', double_fc_power);
fprintf('  Baseband contains the message signal.\n');
fprintf('  2f_c component will be removed by lowpass filter.\n');

%% Prepare output structure
carrier_params.fc = fc_final;
carrier_params.phi = phi;
carrier_params.f_2fc_measured = f_2fc_measured;
carrier_params.X_mixed_dB = X_mixed_dB;
carrier_params.X_mixed_single = X_mixed_single;

fprintf('\nTask 3 complete. Mixed signal ready for Task 4.\n');

end