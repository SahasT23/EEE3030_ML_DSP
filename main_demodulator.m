%% EEE3030 DSB-SC AM Demodulator
% Task 1: Time and Frequency Domain Analysis
clear; close all; clc;

% Load the AM signal
[signal, fs] = audioread('Sahas_Talasila.wav');  % Replace with your actual filename
signal = signal(:);  % Ensure column vector

% Display basic information
fprintf('Sampling frequency: %d Hz\n', fs);
fprintf('Signal length: %d samples\n', length(signal));
fprintf('Duration: %.2f seconds\n', length(signal)/fs);

% %% Time Domain Plot
% t = (0:length(signal)-1) / fs;  % Time vector in seconds

% figure('Position', [100 100 1200 400]);
% plot(t, signal, 'b', 'LineWidth', 0.5);
% xlabel('Time (s)', 'FontSize', 12);
% ylabel('Amplitude', 'FontSize', 12);
% title('Received AM Signal - Time Domain', 'FontSize', 14);
% grid on;
% xlim([0 max(t)]);

% % Zoom in on first 0.01 seconds to see modulation envelope
% figure('Position', [100 100 1200 400]);
% plot(t, signal, 'b', 'LineWidth', 1);
% xlabel('Time (s)', 'FontSize', 12);
% ylabel('Amplitude', 'FontSize', 12);
% title('Received AM Signal - Time Domain (Zoomed)', 'FontSize', 14);
% grid on;
% xlim([0 0.01]);  % First 10ms

%% Frequency Domain Analysis - Manual FFT Implementation
N = length(signal);  % Number of samples
df = fs / N;         % Frequency resolution

% Compute FFT
signal_fft = fft(signal);

% Create frequency vector (only positive frequencies)
f = (0:N/2) * df;

% Take single-sided spectrum (positive frequencies only)
signal_fft_single = signal_fft(1:N/2+1);

% Amplitude normalization (convert to actual amplitudes)
signal_amplitude = abs(signal_fft_single) / N;
signal_amplitude(2:end-1) = 2 * signal_amplitude(2:end-1);  % Double non-DC components

% Convert to dB scale
signal_dB = 20 * log10(signal_amplitude + eps);  % eps prevents log(0)

fprintf('\nFrequency resolution: %.2f Hz\n', df);

% %% Plot Frequency Spectrum
% figure('Position', [100 100 1200 500]);
% plot(f/1000, signal_dB, 'b', 'LineWidth', 1);
% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Magnitude (dB)', 'FontSize', 12);
% title('Received AM Signal - Frequency Domain (Full Spectrum)', 'FontSize', 14);
% grid on;
% xlim([0 fs/2000]);  % Full range to fs/2

% % Zoom in on the region of interest (typically 10-30 kHz for AM signals)
% figure('Position', [100 100 1200 500]);
% plot(f/1000, signal_dB, 'b', 'LineWidth', 1);
% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Magnitude (dB)', 'FontSize', 12);
% title('Received AM Signal - Frequency Domain (Zoomed to AM Band)', 'FontSize', 14);
% grid on;
% xlim([5 35]);  % Adjust this range based on where you see the signal
% ylim([max(signal_dB)-80 max(signal_dB)+5]);  % Show 80 dB dynamic range

% Windowing Techniques
% Apply different window functions and analyze their effects on the FFT
% Looking at rectangular, Hamming, Kaiser, and Blackman windows.

% 1. Rectangular window (no windowing - baseline)
window_rect = ones(N, 1);
signal_rect = signal .* window_rect;

% 2. Hamming window - good general purpose, moderate sidelobe suppression
window_hamming = hamming(N);
signal_hamming = signal .* window_hamming;

% 3. Kaiser window - adjustable sidelobe suppression (beta=8 for ~50dB)
beta = 8;  % Shape parameter - higher = more sidelobe suppression
window_kaiser = kaiser(N, beta);
signal_kaiser = signal .* window_kaiser;

% 4. Blackman window - excellent sidelobe suppression, wider main lobe
window_blackman = blackman(N);
signal_blackman = signal .* window_blackman;

% Compute FFTs for all windowed signals
fft_rect = fft(signal_rect);
fft_hamming = fft(signal_hamming);
fft_kaiser = fft(signal_kaiser);
fft_blackman = fft(signal_blackman);

% Take single-sided spectra
fft_rect_single = fft_rect(1:N/2+1);
fft_hamming_single = fft_hamming(1:N/2+1);
fft_kaiser_single = fft_kaiser(1:N/2+1);
fft_blackman_single = fft_blackman(1:N/2+1);

% Calculate window correction factors (equivalent noise bandwidth)
corr_rect = sum(window_rect) / N;
corr_hamming = sum(window_hamming) / N;
corr_kaiser = sum(window_kaiser) / N;
corr_blackman = sum(window_blackman) / N;

% Amplitude normalization with window correction
amp_rect = abs(fft_rect_single) / N / corr_rect;
amp_hamming = abs(fft_hamming_single) / N / corr_hamming;
amp_kaiser = abs(fft_kaiser_single) / N / corr_kaiser;
amp_blackman = abs(fft_blackman_single) / N / corr_blackman;

% Double non-DC/Nyquist components for single-sided spectrum
amp_rect(2:end-1) = 2 * amp_rect(2:end-1);
amp_hamming(2:end-1) = 2 * amp_hamming(2:end-1);
amp_kaiser(2:end-1) = 2 * amp_kaiser(2:end-1);
amp_blackman(2:end-1) = 2 * amp_blackman(2:end-1);

% Convert to dB
dB_rect = 20 * log10(amp_rect + eps);
dB_hamming = 20 * log10(amp_hamming + eps);
dB_kaiser = 20 * log10(amp_kaiser + eps);
dB_blackman = 20 * log10(amp_blackman + eps);

% %% Visualize Window Functions in Time Domain
% figure('Position', [100 100 1200 600]);

% subplot(2,2,1);
% plot(window_rect, 'b', 'LineWidth', 1.5);
% title('Rectangular Window', 'FontSize', 12);
% xlabel('Sample', 'FontSize', 10);
% ylabel('Amplitude', 'FontSize', 10);
% grid on;
% ylim([0 1.1]);

% subplot(2,2,2);
% plot(window_hamming, 'r', 'LineWidth', 1.5);
% title('Hamming Window', 'FontSize', 12);
% xlabel('Sample', 'FontSize', 10);
% ylabel('Amplitude', 'FontSize', 10);
% grid on;
% ylim([0 1.1]);

% subplot(2,2,3);
% plot(window_kaiser, 'g', 'LineWidth', 1.5);
% title(sprintf('Kaiser Window (BETA=%.1f)', beta), 'FontSize', 12);
% xlabel('Sample', 'FontSize', 10);
% ylabel('Amplitude', 'FontSize', 10);
% grid on;
% ylim([0 1.1]);

% subplot(2,2,4);
% plot(window_blackman, 'm', 'LineWidth', 1.5);
% title('Blackman Window', 'FontSize', 12);
% xlabel('Sample', 'FontSize', 10);
% ylabel('Amplitude', 'FontSize', 10);
% grid on;
% ylim([0 1.1]);

% sgtitle('Window Function Shapes in Time Domain', 'FontSize', 14, 'FontWeight', 'bold');

% %% Compare Spectral Results
% figure('Position', [100 100 1400 800]);

% % Full spectrum comparison
% subplot(2,1,1);
% plot(f/1000, dB_rect, 'b', 'LineWidth', 1, 'DisplayName', 'Rectangular');
% hold on;
% plot(f/1000, dB_hamming, 'r', 'LineWidth', 1, 'DisplayName', 'Hamming');
% plot(f/1000, dB_kaiser, 'g', 'LineWidth', 1, 'DisplayName', sprintf('Kaiser (BETA=%.0f)', beta));
% plot(f/1000, dB_blackman, 'm', 'LineWidth', 1, 'DisplayName', 'Blackman');
% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Magnitude (dB)', 'FontSize', 12);
% title('Window Comparison - Full Spectrum', 'FontSize', 14);
% legend('Location', 'best', 'FontSize', 10);
% grid on;
% xlim([0 fs/2000]);

% % Zoomed to AM signal region
% subplot(2,1,2);
% plot(f/1000, dB_rect, 'b', 'LineWidth', 1.5, 'DisplayName', 'Rectangular');
% hold on;
% plot(f/1000, dB_hamming, 'r', 'LineWidth', 1.5, 'DisplayName', 'Hamming');
% plot(f/1000, dB_kaiser, 'g', 'LineWidth', 1.5, 'DisplayName', sprintf('Kaiser (BETA=%.0f)', beta));
% plot(f/1000, dB_blackman, 'm', 'LineWidth', 1.5, 'DisplayName', 'Blackman');
% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Magnitude (dB)', 'FontSize', 12);
% title('Window Comparison - AM Signal Band (Zoomed)', 'FontSize', 14);
% legend('Location', 'best', 'FontSize', 10);
% grid on;
% xlim([5 35]);  % Adjust
% ylim([max(dB_hamming)-80 max(dB_hamming)+5]);

%% Window Comparison and Band Estimation


% Calculate noise floor for each window (use median of spectrum)
noise_rect = median(dB_rect);
noise_hamming = median(dB_hamming);
noise_kaiser = median(dB_kaiser);
noise_blackman = median(dB_blackman);

fprintf('\n--- Noise Floor Estimates (dB) ---\n');
fprintf('Rectangular:  %.2f dB\n', noise_rect);
fprintf('Hamming:      %.2f dB\n', noise_hamming);
fprintf('Kaiser (BETA=8): %.2f dB\n', noise_kaiser);
fprintf('Blackman:     %.2f dB\n', noise_blackman);

% Calculate peak signal levels
peak_rect = max(dB_rect);
peak_hamming = max(dB_hamming);
peak_kaiser = max(dB_kaiser);
peak_blackman = max(dB_blackman);

fprintf('\n--- Peak Signal Levels (dB) ---\n');
fprintf('Rectangular:  %.2f dB\n', peak_rect);
fprintf('Hamming:      %.2f dB\n', peak_hamming);
fprintf('Kaiser (BETA=8): %.2f dB\n', peak_kaiser);
fprintf('Blackman:     %.2f dB\n', peak_blackman);

% Calculate SNR estimates (peak - noise floor)
snr_rect = peak_rect - noise_rect;
snr_hamming = peak_hamming - noise_hamming;
snr_kaiser = peak_kaiser - noise_kaiser;
snr_blackman = peak_blackman - noise_blackman;

fprintf('\n--- Estimated SNR (dB) ---\n');
fprintf('Rectangular:  %.2f dB\n', snr_rect);
fprintf('Hamming:      %.2f dB\n', snr_hamming);
fprintf('Kaiser (BETA=8): %.2f dB\n', snr_kaiser);
fprintf('Blackman:     %.2f dB\n', snr_blackman);

% Set detection threshold (20 dB above noise floor)
threshold_margin = 20;  % dB above noise
threshold_rect = noise_rect + threshold_margin;
threshold_hamming = noise_hamming + threshold_margin;
threshold_kaiser = noise_kaiser + threshold_margin;
threshold_blackman = noise_blackman + threshold_margin;

fprintf('\n--- Detection Thresholds (dB) ---\n');
fprintf('(Noise floor + %.0f dB)\n', threshold_margin);
fprintf('Rectangular:  %.2f dB\n', threshold_rect);
fprintf('Hamming:      %.2f dB\n', threshold_hamming);
fprintf('Kaiser (BETA=8): %.2f dB\n', threshold_kaiser);
fprintf('Blackman:     %.2f dB\n', threshold_blackman);

% Detect signal bandwidth for each window
detect_rect = find(dB_rect > threshold_rect);
detect_hamming = find(dB_hamming > threshold_hamming);
detect_kaiser = find(dB_kaiser > threshold_kaiser);
detect_blackman = find(dB_blackman > threshold_blackman);


% Function to calculate and display bandwidth
function [fmin_out, fmax_out, fc_out, bw_out] = estimate_bandwidth(f, detect_indices, window_name)
    if ~isempty(detect_indices)
        fmin_raw = f(detect_indices(1));
        fmax_raw = f(detect_indices(end));
        
        % Round to nearest 100 Hz for cleaner values
        fmin_out = round(fmin_raw/100) * 100;
        fmax_out = round(fmax_raw/100) * 100;
        
        fc_out = (fmin_out + fmax_out) / 2;
        bw_out = (fmax_out - fmin_out) / 2;
        
        fprintf('\n--- %s ---\n', window_name);
        fprintf('fmin: %.2f kHz (%.0f Hz)\n', fmin_out/1000, fmin_out);
        fprintf('fmax: %.2f kHz (%.0f Hz)\n', fmax_out/1000, fmax_out);
        fprintf('fc (estimated): %.2f kHz\n', fc_out/1000);
        fprintf('Message BW: %.2f kHz\n', bw_out/1000);
        fprintf('Total occupied BW: %.2f kHz\n', (fmax_out-fmin_out)/1000);
    else
        fprintf('\n--- %s ---\n', window_name);
        fprintf('No signal detected above threshold\n');
        fmin_out = NaN; fmax_out = NaN; fc_out = NaN; bw_out = NaN;
    end
end

% Estimate for each window
[fmin_rect, fmax_rect, fc_rect, bw_rect] = estimate_bandwidth(f, detect_rect, 'Rectangular Window');
[fmin_hamming, fmax_hamming, fc_hamming, bw_hamming] = estimate_bandwidth(f, detect_hamming, 'Hamming Window');
[fmin_kaiser, fmax_kaiser, fc_kaiser, bw_kaiser] = estimate_bandwidth(f, detect_kaiser, 'Kaiser Window (BETA=8)');
[fmin_blackman, fmax_blackman, fc_blackman, bw_blackman] = estimate_bandwidth(f, detect_blackman, 'Blackman Window');

% Calculate measurement consistency
fc_values = [fc_rect, fc_hamming, fc_kaiser, fc_blackman];
fc_mean = mean(fc_values(~isnan(fc_values)));
fc_std = std(fc_values(~isnan(fc_values)));

fprintf('\nCarrier Frequency Consistency:\n');
fprintf('Mean fc: %.2f kHz\n', fc_mean/1000);
fprintf('Std dev: %.2f Hz (%.4f kHz)\n', fc_std, fc_std/1000);
fprintf('Consistency: ');
if fc_std < 100
    fprintf('EXCELLENT (< 100 Hz variation)\n');
elseif fc_std < 500
    fprintf('GOOD (< 500 Hz variation)\n');
else
    fprintf('MODERATE (check signal quality)\n');
end

fprintf('- Rectangular: has best freq resolution but noisy\n');
fprintf('- Hamming: best balance for this application\n');
fprintf('- Kaiser (BETA=8): Similar to Hamming, tunable\n');
fprintf('- Blackman: Cleanest but over-smoothed\n');

% % Store the chosen window results
% fmin = fmin_hamming;
% fmax = fmax_hamming;
% fc_estimated = fc_hamming;
% signal_dB_win = dB_hamming;  % Probably the best choise

% %% Final Visualization - Single Clear Plot
% figure('Position', [100 100 1000 600]);
% plot(f/1000, dB_hamming, 'b', 'LineWidth', 1.5);
% hold on;

% % Mark fmin and fmax
% plot([fmin fmin]/1000, ylim, 'r--', 'LineWidth', 2.5);
% plot([fmax fmax]/1000, ylim, 'g--', 'LineWidth', 2.5);
% plot([fc_estimated fc_estimated]/1000, ylim, 'm--', 'LineWidth', 2);
% yline(threshold_hamming, 'k--', 'LineWidth', 1.5);

% % Add text annotations
% text(fmin/1000, max(dB_hamming)+5, sprintf('  f_{min} = %.2f kHz', fmin/1000), ...
%     'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r');
% text(fmax/1000, max(dB_hamming)-5, sprintf('  f_{max} = %.2f kHz', fmax/1000), ...
%     'FontSize', 11, 'FontWeight', 'bold', 'Color', 'g', 'HorizontalAlignment', 'right');
% text(fc_estimated/1000, max(dB_hamming)-10, sprintf('f_c = %.2f kHz', fc_estimated/1000), ...
%     'FontSize', 11, 'FontWeight', 'bold', 'Color', 'm', 'HorizontalAlignment', 'center');

% xlabel('Frequency (kHz)', 'FontSize', 13);
% ylabel('Magnitude (dB)', 'FontSize', 13);
% title('AM Signal Bandwidth Identification', 'FontSize', 15, 'FontWeight', 'bold');
% grid on;
% xlim([max(0, fmin/1000-5), min(fs/2000, fmax/1000+5)]);  % Auto-zoom to signal ±5kHz
% legend('Signal Spectrum', 'f_{min}', 'f_{max}', 'f_c (estimated)', ...
%     'Detection Threshold', 'Location', 'best', 'FontSize', 10);

%% ===================================================================
%% TASK 2: FIR Bandpass Filter Design
%% ===================================================================

% Load results from Task 1
load('task1_results.mat');
signal = results.signal;
fs = results.fs;
fmin = results.fmin;
fmax = results.fmax;

fprintf('\nFIR bandpass filter fmin and fmax requirements check\n');
fprintf('Using fmin = %.0f Hz, fmax = %.0f Hz\n', fmin, fmax);

%% Filter Specifications
% Passband edges (from Task 1)
fp1 = fmin;  % Lower passband edge
fp2 = fmax;  % Upper passband edge

% Stopband edges (given in assignment)
fstop1 = fmin - 2000;  % Lower stopband edge (below passband)
fstop2 = fmax + 2000;  % Upper stopband edge (above passband)

% Performance requirements
passband_ripple_dB = 0.1;
stopband_atten_dB = 50;

% Calculate transition bandwidth (CORRECT: passband to stopband)
transition_bw = fp1 - fstop1;  % = fmin - (fmin-2000) = 2000 Hz

fprintf('\nFilter Specifications:\n');
fprintf('Passband: %.0f Hz to %.0f Hz\n', fp1, fp2);
fprintf('Stopband: DC-%.0f Hz and %.0f Hz-Nyquist\n', fstop1, fstop2);
fprintf('Transition bandwidth: %.0f Hz\n', transition_bw);

%% Estimate Filter Order
% Using Hamming window formula for FIR filter order
% Formula: N ≈ 3.3 * fs / transition_bw (for Hamming window)

N_estimated = ceil(3.3 * fs / transition_bw);

% Make it odd for symmetry (Type I FIR)
if mod(N_estimated, 2) == 0
    N_estimated = N_estimated + 1;
end

fprintf('\nEstimated filter order: %d taps\n', N_estimated);

% Filter length
M = N_estimated;
n = 0:M-1;  % Sample indices

%% Design Ideal Bandpass Impulse Response
% Normalize frequencies to [0, 1] where 1 is Nyquist (fs/2)
wc1 = 2 * fp1 / fs;  % Normalized lower cutoff
wc2 = 2 * fp2 / fs;  % Normalized upper cutoff

% Center point of filter
M_center = (M - 1) / 2;

% Ideal bandpass impulse response
h_ideal = zeros(1, M);
for i = 1:M
    if i == M_center + 1  % Handle n = 0 case (avoid division by zero)
        h_ideal(i) = wc2 - wc1;
    else
        % Sinc functions for bandpass
        h_ideal(i) = (sin(pi * wc2 * (i - M_center - 1)) - ...
                      sin(pi * wc1 * (i - M_center - 1))) / ...
                     (pi * (i - M_center - 1));
    end
end

fprintf('Ideal impulse response computed\n');

%% Apply Hamming Window
% Generate Hamming window
w_hamming = hamming(M)';

% Apply window to ideal impulse response
h_fir = h_ideal .* w_hamming;

fprintf('Hamming window applied\n');
fprintf('Final filter length: %d taps\n', length(h_fir));

%% Verify Filter Frequency Response
% % Compute frequency response (use FFT for efficiency)
% N_fft = 8192;  % Zero-padding for smooth frequency response
% H = fft(h_fir, N_fft);
% H_mag = abs(H(1:N_fft/2+1));
% H_dB = 20 * log10(H_mag + eps);

% % Frequency vector
% f_response = (0:N_fft/2) * fs / N_fft;

% % Plot frequency response
% figure('Position', [100 100 1000 600]);
% plot(f_response/1000, H_dB, 'b', 'LineWidth', 1.5);
% hold on;

% % Mark specifications
% yline(-passband_ripple_dB, 'g--', 'LineWidth', 1.5, 'Label', 'Passband ripple');
% yline(-stopband_atten_dB, 'r--', 'LineWidth', 1.5, 'Label', 'Stopband spec');
% xline(fp1/1000, 'k--', 'LineWidth', 1);
% xline(fp2/1000, 'k--', 'LineWidth', 1);
% xline(fstop1/1000, 'r--', 'LineWidth', 1);
% xline(fstop2/1000, 'r--', 'LineWidth', 1);

% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Magnitude (dB)', 'FontSize', 12);
% title('FIR Bandpass Filter Frequency Response', 'FontSize', 14);
% grid on;
% xlim([0 fs/2000]);
% ylim([-80 5]);

fprintf('\nFilter verification complete\n');

%% Custom Convolution Implementation and package/header call
fprintf('\nApplying FIR filter via custom convolution...\n');

% Custom convolution function
signal_filtered = custom_conv(signal, h_fir);

fprintf('Filtering complete. Output length: %d samples\n', length(signal_filtered));

% For comparison, also use MATLAB's filter function
signal_filtered_matlab = filter(h_fir, 1, signal);

%% Analyze Filtered Signal in Time and Frequency Domain

% Compute spectrum of filtered signal
N_sig = length(signal_filtered);
signal_filtered_fft = fft(signal_filtered);
f_filtered = (0:N_sig/2) * fs / N_sig;
signal_filtered_fft_single = signal_filtered_fft(1:N_sig/2+1);

% Apply Hamming window for spectral analysis
window_analysis = hamming(N_sig);
signal_filtered_windowed = signal_filtered .* window_analysis;
fft_windowed = fft(signal_filtered_windowed);
fft_windowed_single = fft_windowed(1:N_sig/2+1);

% Normalize
corr = sum(window_analysis) / N_sig;
amp_filtered = abs(fft_windowed_single) / N_sig / corr;
amp_filtered(2:end-1) = 2 * amp_filtered(2:end-1);
dB_filtered = 20 * log10(amp_filtered + eps);

% % Single plot showing before and after
% figure('Position', [100 100 1000 600]);
% plot(f/1000, dB_hamming, 'r', 'LineWidth', 1, 'DisplayName', 'Original Signal');
% hold on;
% plot(f_filtered/1000, dB_filtered, 'b', 'LineWidth', 1.5, 'DisplayName', 'Filtered Signal');
% xline(fp1/1000, 'k--', 'LineWidth', 1);
% xline(fp2/1000, 'k--', 'LineWidth', 1);

% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Magnitude (dB)', 'FontSize', 12);
% title('Effect of Bandpass Filtering', 'FontSize', 14);
% legend('Location', 'best');
% grid on;
% xlim([0 40]);

fprintf('Out-of-band noise significantly reduced\n');
fprintf('Signal passband preserved\n');

%% Save Results for Task 3
results_task2.signal_filtered = signal_filtered;
results_task2.h_fir = h_fir;
results_task2.filter_order = M;
results_task2.fp1 = fp1;
results_task2.fp2 = fp2;

save('task2_results.mat', 'results_task2');

fprintf('\n=== TASK 2 COMPLETE ===\n');
fprintf('Filter order: %d taps\n', M);
fprintf('Results saved to task2_results.mat\n');
fprintf('Ready for Task 3\n\n');

%% ===================================================================
%% TASK 3: Carrier Recovery and Mixing
%% ===================================================================

% Load results from previous tasks
load('task1_results.mat');
load('task2_results.mat');

signal_filtered = results_task2.signal_filtered;
fs = results.fs;

%% Apply Square Law (|signal|^2)
signal_squared = signal_filtered .^ 2;

fprintf('Square law applied to filtered signal\n');
fprintf('Squared signal length: %d samples\n', length(signal_squared));

%% Plot Squared Signal - Time Domain
t = (0:length(signal_squared)-1) / fs;

figure('Position', [100 100 1000 400]);
plot(t, signal_squared, 'b', 'LineWidth', 0.5);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Squared Signal - Time Domain', 'FontSize', 14);
grid on;
xlim([0 min(0.01, max(t))]);  % Show first 10ms

fprintf('Squared signal plotted in time domain\n');

%% Compute Spectrum of Squared Signal
N = length(signal_squared);

% Apply Hamming window for better frequency resolution
window = hamming(N);
signal_squared_windowed = signal_squared .* window;

% Compute FFT
squared_fft = fft(signal_squared_windowed);
squared_fft_single = squared_fft(1:N/2+1);

% Frequency vector
f = (0:N/2) * fs / N;

% Normalize amplitude
window_corr = sum(window) / N;
amp_squared = abs(squared_fft_single) / N / window_corr;
amp_squared(2:end-1) = 2 * amp_squared(2:end-1);

% Convert to dB
dB_squared = 20 * log10(amp_squared + eps);

fprintf('Spectrum of squared signal computed\n');

%% Plot Spectrum and Find 2fc Peak
% figure('Position', [100 100 1000 600]);
% plot(f/1000, dB_squared, 'b', 'LineWidth', 1.5);
% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Magnitude (dB)', 'FontSize', 12);
% title('Spectrum of Squared Signal (Carrier Recovery)', 'FontSize', 14);
% grid on;
% xlim([0 50]);  % Focus on lower frequencies where 2fc should be

% Find the peak (2fc)
% Search in reasonable range (between 20-50 kHz typically)
search_range = (f > 20000) & (f < 50000);
[max_val, max_idx_relative] = max(dB_squared(search_range));

% Get actual index in full array
search_indices = find(search_range);
max_idx = search_indices(max_idx_relative);

fc_double = f(max_idx);
fc_recovered = fc_double / 2;  % Divide by 2 to get actual carrier

% Round to nearest kHz (assignment says fc is exact multiple of 1 kHz)
fc_recovered = round(fc_recovered / 1000) * 1000;

fprintf('Peak at 2fc = %.2f kHz\n', fc_double/1000);
fprintf('Recovered carrier frequency fc = %.2f kHz\n', fc_recovered/1000);
fprintf('(Rounded to nearest 1 kHz as specified)\n');

% % Mark the peak on plot
% hold on;
% plot(fc_double/1000, max_val, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% text(fc_double/1000, max_val + 5, sprintf('  2f_c = %.1f kHz', fc_double/1000), ...
%     'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r');

%% Generate Local Carrier Signal
% Use phi  = 0 for now (will optimise later))
phi = 0;

% Time vector
t = (0:length(signal_filtered)-1)' / fs;

% Generate carrier: cos(2pifc·t + phi)
carrier_local = cos(2 * pi * fc_recovered * t + phi);

fprintf('\nLocal carrier signal generated\n');
fprintf('Carrier frequency: %.2f kHz\n', fc_recovered/1000);
fprintf('Phase: %.2f radians (%.1f degrees)\n', phi, rad2deg(phi));

%% Multiply Filtered Signal with Local Carrier (Product Detector)
signal_mixed = signal_filtered .* carrier_local;

fprintf('Mixing (multiplication) complete\n');
fprintf('Mixed signal length: %d samples\n', length(signal_mixed));

%% Plot Mixed Signal - Time Domain
figure('Position', [100 100 1000 400]);
plot(t, signal_mixed, 'b', 'LineWidth', 0.5);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Mixed Signal (After Product Detector) - Time Domain', 'FontSize', 14);
grid on;
xlim([0 0.02]);  % Show first 20ms

fprintf('Mixed signal plotted in time domain\n');

%% Apply window
signal_mixed_windowed = signal_mixed .* window;

% Compute FFT
mixed_fft = fft(signal_mixed_windowed);
mixed_fft_single = mixed_fft(1:N/2+1);

% Normalising our signal with the amplitude as well as window correction
amp_mixed = abs(mixed_fft_single) / N / window_corr;
amp_mixed(2:end-1) = 2 * amp_mixed(2:end-1);
dB_mixed = 20 * log10(amp_mixed + eps);

% Plot
figure('Position', [100 100 1000 600]);
plot(f/1000, dB_mixed, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Mixed Signal - Frequency Domain', 'FontSize', 14);
grid on;
xlim([0 50]);

% Mark key frequencies
hold on;
xline(4, 'g--', 'LineWidth', 2, 'Label', 'Message BW (4 kHz)');
xline(2*fc_recovered/1000, 'r--', 'LineWidth', 2, 'Label', '2f_c component');

fprintf('Expected components:\n');
fprintf('1. Baseband (0-4 kHz): Audio message m(t)\n');
fprintf('2. High frequency (~%.1f kHz): 2fc component to remove\n', 2*fc_recovered/1000);

%% Save Results for Task 4
results_task3.signal_mixed = signal_mixed;
results_task3.fc_recovered = fc_recovered;
results_task3.carrier_local = carrier_local;
results_task3.phi_initial = phi;

save('task3_results.mat', 'results_task3');

fprintf('Recovered carrier frequency: %.2f kHz\n', fc_recovered/1000);
fprintf('Mixed signal ready for lowpass filtering (Task 4)\n');
fprintf('Results saved to task3_results.mat\n\n');

%% ===================================================================
%% TASK 4: IIR Lowpass Filter Design
%% ===================================================================

% Load results from Task 3
load('task3_results.mat');
load('task1_results.mat');

signal_mixed = results_task3.signal_mixed;
fs = results.fs;

%% Design IIR Butterworth Lowpass Filter
% Specifications from assignment
filter_order = 4;
cutoff_freq = 4000;  % 4 kHz
filter_type = 'Butterworth';

fprintf('Filter Specifications:\n');
fprintf('Order: %d\n', filter_order);
fprintf('Cutoff frequency: %.0f Hz (%.1f kHz)\n', cutoff_freq, cutoff_freq/1000);
fprintf('Type: %s\n', filter_type);

% Normalize cutoff frequency (Wn must be between 0 and 1, where 1 is Nyquist)
Wn = cutoff_freq / (fs/2);

fprintf('Normalized cutoff frequency: %.4f\n', Wn);

% Design Butterworth filter - get coefficients
[b, a] = butter(filter_order, Wn, 'low');

fprintf('\nFilter coefficients obtained:\n');
fprintf('Numerator (b) coefficients: %d values\n', length(b));
fprintf('Denominator (a) coefficients: %d values\n', length(a));

% Display coefficients
fprintf('\nb (numerator):\n');
fprintf('  %.6f\n', b);
fprintf('\na (denominator):\n');
fprintf('  %.6f\n', a);

%% Verify Filter Frequency Response
% Compute frequency response
N_freq = 8192;  % Number of frequency points
[H, f_response] = freqz(b, a, N_freq, fs);

% Convert to magnitude (dB) and phase
H_mag = abs(H);
H_dB = 20 * log10(H_mag);
H_phase = angle(H);

% % Plot magnitude response
% figure('Position', [100 100 1000 700]);

% subplot(2,1,1);
% plot(f_response/1000, H_dB, 'b', 'LineWidth', 1.5);
% hold on;
% xline(cutoff_freq/1000, 'r--', 'LineWidth', 2, 'Label', 'Cutoff (4 kHz)');
% yline(-3, 'g--', 'LineWidth', 1.5, 'Label', '-3 dB');
% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Magnitude (dB)', 'FontSize', 12);
% title('IIR Butterworth Lowpass Filter - Magnitude Response', 'FontSize', 14);
% grid on;
% xlim([0 20]);
% ylim([-80 5]);

% subplot(2,1,2);
% plot(f_response/1000, rad2deg(H_phase), 'b', 'LineWidth', 1.5);
% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Phase (degrees)', 'FontSize', 12);
% title('Phase Response', 'FontSize', 14);
% grid on;
% xlim([0 20]);

fprintf('\nFilter frequency response verified\n');

% Check -3dB point
[~, idx_cutoff] = min(abs(f_response - cutoff_freq));
attenuation_at_cutoff = H_dB(idx_cutoff);
fprintf('Attenuation at %.0f Hz: %.2f dB\n', cutoff_freq, attenuation_at_cutoff);

%% Apply Filter Using MATLAB filter() function
% Using filter() to apply IIR filter with the difference equation (need to check if this is acceptable for cw)
signal_demod_matlab = filter(b, a, signal_mixed);

fprintf('Output signal length: %d samples\n', length(signal_demod_matlab));

%% Apply Filter Using custom_iir_filter() function

signal_demod_custom = custom_iir_filter(b, a, signal_mixed);

fprintf('Output signal length: %d samples\n', length(signal_demod_custom));

% Verify both methods give same result
difference = max(abs(signal_demod_matlab - signal_demod_custom));
fprintf('\nVerification: Max difference between MATLAB and custom = %.2e\n', difference);

if difference < 1e-10
    fprintf(' Both methods match \n');
elseif difference < 1e-6
    fprintf('✓ GOOD: Both methods match (minor numerical differences)\n');
else
    warning('Methods differ significantly - check implementation');
end

% Use custom implementation for remaining analysis
signal_demod = signal_demod_custom;

%% Analyse Demodulated Signal - Time Domain
% t = (0:length(signal_demod)-1) / fs;

% figure('Position', [100 100 1000 400]);
% plot(t, signal_demod, 'b', 'LineWidth', 0.8);
% xlabel('Time (s)', 'FontSize', 12);
% ylabel('Amplitude', 'FontSize', 12);
% title('Demodulated Signal (After Lowpass Filter) - Time Domain', 'FontSize', 14);
% grid on;
% xlim([0 max(t)]);

% % Zoom to show detail
% figure('Position', [100 100 1000 400]);
% plot(t, signal_demod, 'b', 'LineWidth', 1);
% xlabel('Time (s)', 'FontSize', 12);
% ylabel('Amplitude', 'FontSize', 12);
% title('Demodulated Signal - Time Domain (Zoomed)', 'FontSize', 14);
% grid on;
% xlim([0 2]);  % 2  seconds

%% Analyse Demodulated Signal - Frequency Domain
N = length(signal_demod);

% Apply window for spectral analysis
window = hamming(N);
signal_demod_windowed = signal_demod .* window;

% Compute FFT
demod_fft = fft(signal_demod_windowed);
demod_fft_single = demod_fft(1:N/2+1);

% Frequency vector
f = (0:N/2) * fs / N;

% Normalize
window_corr = sum(window) / N;
amp_demod = abs(demod_fft_single) / N / window_corr;
amp_demod(2:end-1) = 2 * amp_demod(2:end-1);
dB_demod = 20 * log10(amp_demod + eps);

% % Plot
% figure('Position', [100 100 1000 600]);
% plot(f/1000, dB_demod, 'b', 'LineWidth', 1.5);
% hold on;
% xline(cutoff_freq/1000, 'r--', 'LineWidth', 2, 'Label', 'Filter cutoff (4 kHz)');
% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Magnitude (dB)', 'FontSize', 12);
% title('Demodulated Signal - Frequency Domain', 'FontSize', 14);
% grid on;
% xlim([0 20]);

fprintf('Audio message bandwidth: 0-4 kHz\n');
fprintf('High frequency components (>4 kHz) removed by lowpass filter\n');

% Check energy distribution
energy_passband = sum(amp_demod(f <= cutoff_freq).^2);
energy_stopband = sum(amp_demod(f > cutoff_freq).^2);
energy_ratio_dB = 10*log10(energy_passband / energy_stopband);

fprintf('Energy in passband (0-4 kHz): %.2f%%\n', 100*energy_passband/(energy_passband+energy_stopband));
fprintf('Energy in stopband (>4 kHz): %.2f%%\n', 100*energy_stopband/(energy_passband+energy_stopband));
fprintf('Passband/Stopband energy ratio: %.1f dB\n', energy_ratio_dB);

%% Compare Mixed Signal (before) vs Demodulated (after)
% Compute spectrum of mixed signal for comparison
signal_mixed_windowed = signal_mixed .* window;
mixed_fft = fft(signal_mixed_windowed);
mixed_fft_single = mixed_fft(1:N/2+1);
amp_mixed = abs(mixed_fft_single) / N / window_corr;
amp_mixed(2:end-1) = 2 * amp_mixed(2:end-1);
dB_mixed = 20 * log10(amp_mixed + eps);

% figure('Position', [100 100 1000 600]);
% plot(f/1000, dB_mixed, 'r', 'LineWidth', 1, 'DisplayName', 'Before filtering (mixed)');
% hold on;
% plot(f/1000, dB_demod, 'b', 'LineWidth', 1.5, 'DisplayName', 'After filtering (demodulated)');
% xline(cutoff_freq/1000, 'k--', 'LineWidth', 2, 'DisplayName', 'Cutoff (4 kHz)');
% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Magnitude (dB)', 'FontSize', 12);
% title('Effect of Lowpass Filter on Mixed Signal', 'FontSize', 14);
% legend('Location', 'best', 'FontSize', 10);
% grid on;
% xlim([0 50]);

%% Save Results for Task 5
results_task4.signal_demod = signal_demod;
results_task4.b = b;
results_task4.a = a;
results_task4.filter_order = filter_order;
results_task4.cutoff_freq = cutoff_freq;

save('task4_results.mat', 'results_task4');
fprintf('Results saved to task4_results.mat\n\n');

%% ===================================================================
%% TASK 5: Phase Optimization and Audio Output
%% ===================================================================

% Load all previous results
load('task1_results.mat');
load('task2_results.mat');
load('task3_results.mat');
load('task4_results.mat');

signal_filtered = results_task2.signal_filtered;
fc_recovered = results_task3.fc_recovered;
b = results_task4.b;
a = results_task4.a;
fs = results.fs;

%% Test Initial Phase (phi = 0)
phi_initial = 0;

% Generate carrier with phi = 0
t = (0:length(signal_filtered)-1)' / fs;
carrier_initial = cos(2 * pi * fc_recovered * t + phi_initial);

% Mix
signal_mixed_initial = signal_filtered .* carrier_initial;

% Filter
signal_demod_initial = custom_iir_filter(b, a, signal_mixed_initial);

% Calculate peak amplitude
peak_amp_initial = max(abs(signal_demod_initial));
rms_initial = sqrt(mean(signal_demod_initial.^2));

fprintf('\nInitial phase phi = 0:\n');
fprintf('Peak amplitude: %.4f\n', peak_amp_initial);
fprintf('RMS amplitude: %.4f\n', rms_initial);

%% Phase Optimization - Sweep from 0 to pi
fprintf('\n phase sweep from 0 to pi radians\n');

% Phase values to test
phi_values = linspace(0, pi, 50);  % Test 50 values from 0 to π
num_phases = length(phi_values);

% Store results
peak_amplitudes = zeros(num_phases, 1);
rms_amplitudes = zeros(num_phases, 1);

% Test each phase
for i = 1:num_phases
    phi = phi_values(i);
    
    % Generate carrier with current phase
    carrier = cos(2 * pi * fc_recovered * t + phi);
    
    % Mix
    signal_mixed = signal_filtered .* carrier;
    
    % Filter
    signal_demod = custom_iir_filter(b, a, signal_mixed);
    
    % Calculate metrics
    peak_amplitudes(i) = max(abs(signal_demod));
    rms_amplitudes(i) = sqrt(mean(signal_demod.^2));
    
    % Progress indicator
    if mod(i, 10) == 0
        fprintf('  Tested %d/%d phases...\n', i, num_phases);
    end
end

fprintf('Phase sweep complete\n');

%% Find Optimal Phase
[max_peak, idx_peak] = max(peak_amplitudes);
phi_optimal = phi_values(idx_peak);

[max_rms, idx_rms] = max(rms_amplitudes);
phi_optimal_rms = phi_values(idx_rms);

fprintf('Optimal phase (peak): phi = %.4f rad (%.2f°)\n', phi_optimal, rad2deg(phi_optimal));
fprintf('Maximum peak amplitude: %.4f\n', max_peak);
fprintf('Improvement over phi = 0: %.2f%%\n', 100*(max_peak/peak_amp_initial - 1));

fprintf('\nOptimal phase (RMS): phi = %.4f rad (%.2f°)\n', phi_optimal_rms, rad2deg(phi_optimal_rms));
fprintf('Maximum RMS amplitude: %.4f\n', max_rms);

% % Plot phase optimization curve
% figure('Position', [100 100 1000 600]);

% subplot(2,1,1);
% plot(rad2deg(phi_values), peak_amplitudes, 'b-', 'LineWidth', 1.5);
% hold on;
% plot(rad2deg(phi_optimal), max_peak, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% xlabel('Phase φ (degrees)', 'FontSize', 12);
% ylabel('Peak Amplitude', 'FontSize', 12);
% title('Phase Optimization - Peak Amplitude', 'FontSize', 14);
% grid on;
% text(rad2deg(phi_optimal), max_peak*1.05, sprintf('  Optimal: %.1f°', rad2deg(phi_optimal)), ...
%     'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');

% subplot(2,1,2);
% plot(rad2deg(phi_values), rms_amplitudes, 'b-', 'LineWidth', 1.5);
% hold on;
% plot(rad2deg(phi_optimal_rms), max_rms, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% xlabel('Phase phi (degrees)', 'FontSize', 12);
% ylabel('RMS Amplitude', 'FontSize', 12);
% title('Phase Optimization - RMS Amplitude', 'FontSize', 14);
% grid on;

%% Generate Final Demodulated Signal with Optimal Phase
fprintf('\nGenerating final demodulated signal with optimal phase...\n');

% Use optimal phase (from peak amplitude)
carrier_optimal = cos(2 * pi * fc_recovered * t + phi_optimal);

% Mix
signal_mixed_optimal = signal_filtered .* carrier_optimal;

% Filter
signal_demod_final = custom_iir_filter(b, a, signal_mixed_optimal);

fprintf('Final demodulated signal generated\n');
fprintf('Signal length: %.2f seconds\n', length(signal_demod_final)/fs);

%% Plot Final Demodulated Signal
t_final = (0:length(signal_demod_final)-1) / fs;

% figure('Position', [100 100 1000 500]);
% plot(t_final, signal_demod_final, 'b', 'LineWidth', 0.8);
% xlabel('Time (s)', 'FontSize', 12);
% ylabel('Amplitude', 'FontSize', 12);
% title(sprintf('Final Demodulated Audio Signal (φ = %.2f°)', rad2deg(phi_optimal)), 'FontSize', 14);
% grid on;
% xlim([0 max(t_final)]);

% % Also plot zoomed view
% figure('Position', [100 100 1000 500]);
% plot(t_final, signal_demod_final, 'b', 'LineWidth', 1);
% xlabel('Time (s)', 'FontSize', 12);
% ylabel('Amplitude', 'FontSize', 12);
% title('Final Demodulated Signal (Zoomed)', 'FontSize', 14);
% grid on;
% xlim([0 min(3, max(t_final))]);  

fprintf('Final signal plotted\n');

%% Normalize and Play Audio
amplification_factor = 3;  % Increase volume
signal_amplified = signal_demod_final * amplification_factor; % remove this line and above

% Normalize to [-1, 1] range for audio playback
signal_audio = signal_amplified / max(abs(signal_demod_final)); % change signal_amplified to signal_demod_final if it doesn't work.

fprintf('Playing demodulated audio message...\n');

playback_rate = fs * 0.8;  % Slow down by 30%
sound(signal_audio, playback_rate);

% Wait for playback to finish
pause(length(signal_audio)/fs + 0.5);

fprintf('Audio playback complete\n');

%% Identify the 3-Letter Message
fprintf('Please listen to the audio and identify the 3-letter message.\n');
fprintf('Enter the message below:\n');

% Uncomment the next line to interactively input the message
message = input('3-letter message: ', 's');

% For now, leave space to manually record
% message = 'KDH';  

fprintf('\nIdentified message: %s\n', message);

%% Save Audio to File
fprintf('\nSaving audio to file...\n');

% Save as WAV file
audiowrite('demodulated_message_Sahas_T.wav', signal_audio, fs);
fprintf('Audio saved as: demodulated_message.wav\n');
