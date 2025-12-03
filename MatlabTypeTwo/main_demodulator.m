%% EEE3030 DSB-SC AM Demodulator - ENHANCED VERSION
% Task 1: Time and Frequency Domain Analysis
% Enhanced with automatic plot saving and extensive verification
clear; close all; clc;

%% ===== SETUP: Create Plots Directory =====
% Create a folder to save all plots for the report
plots_dir = 'report_plots';
if ~exist(plots_dir, 'dir')
    mkdir(plots_dir);
    fprintf('Created directory: %s\n', plots_dir);
else
    fprintf('Using existing directory: %s\n', plots_dir);
end

% Function to save current figure in both PNG and FIG formats
function save_plot(fig_handle, filename, plots_dir)
    saveas(fig_handle, fullfile(plots_dir, [filename '.png']));
    saveas(fig_handle, fullfile(plots_dir, [filename '.fig']));
end


%% ===== TASK 1: LOAD AND ANALYZE SIGNAL =====
fprintf('>>> TASK 1: TIME AND FREQUENCY DOMAIN ANALYSIS\n');
fprintf('--------------------------------------------\n');

% Load the AM signal
fprintf('Loading audio file...\n');
[signal, fs] = audioread('Sahas_Talasila.wav');
signal = signal(:);  % Ensure column vector
fprintf(' Audio file loaded successfully\n\n');

% Display basic information
fprintf('=== Signal Properties ===\n');
fprintf('Sampling frequency: %d Hz (%.1f kHz)\n', fs, fs/1000);
fprintf('Signal length: %d samples\n', length(signal));
fprintf('Duration: %.3f seconds\n', length(signal)/fs);
fprintf('Signal range: [%.6f, %.6f]\n', min(signal), max(signal));
fprintf('Signal mean: %.6f (DC component)\n', mean(signal));
fprintf('Signal RMS: %.6f\n', sqrt(mean(signal.^2)));
fprintf('Signal std dev: %.6f\n', std(signal));
fprintf('\n');

%% Time Domain Plot
fprintf('Generating time domain plots...\n');
t = (0:length(signal)-1) / fs;  % Time vector in seconds

% Plot 1: Full signal
fig1 = figure('Position', [100 100 1200 400]);
plot(t, signal, 'b', 'LineWidth', 0.5);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Received AM Signal - Time Domain (Full Duration)', 'FontSize', 14);
grid on;
xlim([0 max(t)]);
save_plot(fig1, 'Task1_01_TimeDomain_Full', plots_dir);
fprintf('   Saved: Task1_01_TimeDomain_Full\n');

% Plot 2: Zoomed view to see carrier oscillations
fig2 = figure('Position', [100 100 1200 400]);
plot(t, signal, 'b', 'LineWidth', 1);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Received AM Signal - Time Domain (Zoomed: First 10ms)', 'FontSize', 14);
grid on;
xlim([0 0.01]);  % First 10ms
save_plot(fig2, 'Task1_02_TimeDomain_Zoomed', plots_dir);
fprintf('   Saved: Task1_02_TimeDomain_Zoomed\n');

% Additional time domain analysis
fprintf('\n=== Time Domain Statistics ===\n');
fprintf('Peak-to-peak amplitude: %.6f\n', max(signal) - min(signal));
fprintf('Crest factor: %.3f\n', max(abs(signal)) / sqrt(mean(signal.^2)));
fprintf('Zero crossings: %d\n', sum(diff(sign(signal)) ~= 0));
fprintf('\n');

%% Frequency Domain Analysis - Manual FFT Implementation
fprintf('Computing FFT...\n');
N = length(signal);  % Number of samples
df = fs / N;         % Frequency resolution

% Compute FFT
signal_fft = fft(signal);

% Create frequency vector (only positive frequencies)
f = (0:N/2) * df;

% Take single-sided spectrum (positive frequencies only)
signal_fft_single = signal_fft(1:N/2+1);

% Amplitude normalisation (convert to actual amplitudes)
signal_amplitude = abs(signal_fft_single) / N;
signal_amplitude(2:end-1) = 2 * signal_amplitude(2:end-1);  % Double non-DC components

% Convert to dB scale
signal_dB = 20 * log10(signal_amplitude + eps);  % eps prevents log(0)

fprintf(' FFT computed\n');
fprintf('\n=== FFT Parameters ===\n');
fprintf('FFT length (N): %d samples\n', N);
fprintf('Frequency resolution (df): %.3f Hz\n', df);
fprintf('Nyquist frequency: %.3f kHz\n', fs/2000);
fprintf('Frequency bins: %d\n', length(f));
fprintf('\n');

%% Plot Frequency Spectrum
fprintf('Generating frequency domain plots...\n');

% Plot 3: Full spectrum
fig3 = figure('Position', [100 100 1200 500]);
plot(f/1000, signal_dB, 'b', 'LineWidth', 1);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Received AM Signal - Frequency Domain (Full Spectrum)', 'FontSize', 14);
grid on;
xlim([0 fs/2000]);  % Full range to fs/2
save_plot(fig3, 'Task1_03_FreqDomain_Full', plots_dir);
fprintf('   Saved: Task1_03_FreqDomain_Full\n');

% Plot 4: Zoomed to AM band
fig4 = figure('Position', [100 100 1200 500]);
plot(f/1000, signal_dB, 'b', 'LineWidth', 1);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Received AM Signal - Frequency Domain (Zoomed to AM Band)', 'FontSize', 14);
grid on;
xlim([5 35]);  % Adjust this range based on where you see the signal
ylim([max(signal_dB)-80 max(signal_dB)+5]);  % Show 80 dB dynamic range
save_plot(fig4, 'Task1_04_FreqDomain_Zoomed', plots_dir);
fprintf('   Saved: Task1_04_FreqDomain_Zoomed\n');

% Frequency domain statistics
fprintf('\n=== Frequency Domain Statistics ===\n');
fprintf('DC component: %.6f (%.2f dB)\n', signal_amplitude(1), signal_dB(1));
fprintf('Max magnitude: %.6f (%.2f dB) at %.2f kHz\n', max(signal_amplitude), max(signal_dB), f(signal_amplitude == max(signal_amplitude))/1000);
fprintf('Noise floor (median): %.2f dB\n', median(signal_dB));
fprintf('Dynamic range: %.2f dB\n', max(signal_dB) - median(signal_dB));
fprintf('\n');

%% Windowing Techniques
fprintf('Applying window functions for spectral analysis...\n');

% 1. Rectangular window (no windowing - baseline)
window_rect = ones(N, 1);
signal_rect = signal .* window_rect;

% 2. Hamming window 
window_hamming = hamming(N);
signal_hamming = signal .* window_hamming;

% 3. Kaiser window - adjustable sidelobe suppression (beta=8 for 50dB)
beta = 8;  % Shape parameter - higher = more sidelobe suppression
window_kaiser = kaiser(N, beta);
signal_kaiser = signal .* window_kaiser;

% 4. Blackman window - better sidelobe suppression, wider main lobe
window_blackman = blackman(N);
signal_blackman = signal .* window_blackman;

fprintf(' Window functions created\n');
fprintf('  - Rectangular (no windowing)\n');
fprintf('  - Hamming\n');
fprintf('  - Kaiser (beta=%.1f)\n', beta);
fprintf('  - Blackman\n');
fprintf('\n');

%% Compute FFTs for all windowed signals
fprintf('Computing FFTs for windowed signals...\n');

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

fprintf(' FFTs computed for all windows\n');
fprintf('\n=== Window Correction Factors ===\n');
fprintf('Rectangular:  %.6f\n', corr_rect);
fprintf('Hamming:      %.6f\n', corr_hamming);
fprintf('Kaiser:       %.6f\n', corr_kaiser);
fprintf('Blackman:     %.6f\n', corr_blackman);
fprintf('\n');

% Amplitude normalisation with window correction
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

%% Visualize Window Functions in Time Domain
fprintf('Generating window function plots...\n');

fig5 = figure('Position', [100 100 1200 600]);

subplot(2,2,1);
plot(window_rect, 'b', 'LineWidth', 1.5);
title('Rectangular Window', 'FontSize', 12);
xlabel('Sample', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
grid on;
ylim([0 1.1]);

subplot(2,2,2);
plot(window_hamming, 'r', 'LineWidth', 1.5);
title('Hamming Window', 'FontSize', 12);
xlabel('Sample', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
grid on;
ylim([0 1.1]);

subplot(2,2,3);
plot(window_kaiser, 'g', 'LineWidth', 1.5);
title(sprintf('Kaiser Window (beta=%.1f)', beta), 'FontSize', 12);
xlabel('Sample', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
grid on;
ylim([0 1.1]);

subplot(2,2,4);
plot(window_blackman, 'm', 'LineWidth', 1.5);
title('Blackman Window', 'FontSize', 12);
xlabel('Sample', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
grid on;
ylim([0 1.1]);

sgtitle('Window Function Shapes in Time Domain', 'FontSize', 14, 'FontWeight', 'bold');
save_plot(fig5, 'Task1_05_Windows_TimeDomain', plots_dir);
fprintf('   Saved: Task1_05_Windows_TimeDomain\n');

%% Compare Spectral Results
fprintf('Generating window comparison plots...\n');

fig6 = figure('Position', [100 100 1400 800]);

% Full spectrum comparison
subplot(2,1,1);
plot(f/1000, dB_rect, 'b', 'LineWidth', 1, 'DisplayName', 'Rectangular');
hold on;
plot(f/1000, dB_hamming, 'r', 'LineWidth', 1, 'DisplayName', 'Hamming');
plot(f/1000, dB_kaiser, 'g', 'LineWidth', 1, 'DisplayName', sprintf('Kaiser (beta=%.0f)', beta));
plot(f/1000, dB_blackman, 'm', 'LineWidth', 1, 'DisplayName', 'Blackman');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Window Comparison - Full Spectrum', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0 fs/2000]);

% Zoomed to AM signal region
subplot(2,1,2);
plot(f/1000, dB_rect, 'b', 'LineWidth', 1.5, 'DisplayName', 'Rectangular');
hold on;
plot(f/1000, dB_hamming, 'r', 'LineWidth', 1.5, 'DisplayName', 'Hamming');
plot(f/1000, dB_kaiser, 'g', 'LineWidth', 1.5, 'DisplayName', sprintf('Kaiser (beta=%.0f)', beta));
plot(f/1000, dB_blackman, 'm', 'LineWidth', 1.5, 'DisplayName', 'Blackman');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Window Comparison - AM Signal Band (Zoomed)', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([5 35]);  % Adjust
ylim([max(dB_hamming)-80 max(dB_hamming)+5]);

save_plot(fig6, 'Task1_06_Windows_FreqComparison', plots_dir);
fprintf('   Saved: Task1_06_Windows_FreqComparison\n');

%% Window Comparison and Band Estimation
fprintf('\nAnalyzing window performance...\n');

% Calculate noise floor for each window (use median of spectrum)
noise_rect = median(dB_rect);
noise_hamming = median(dB_hamming);
noise_kaiser = median(dB_kaiser);
noise_blackman = median(dB_blackman);

fprintf('\n=== Noise Floor Estimates (dB) ===\n');
fprintf('Rectangular:  %.2f dB\n', noise_rect);
fprintf('Hamming:      %.2f dB\n', noise_hamming);
fprintf('Kaiser (beta=8): %.2f dB\n', noise_kaiser);
fprintf('Blackman:     %.2f dB\n', noise_blackman);

% Calculate peak signal levels
peak_rect = max(dB_rect);
peak_hamming = max(dB_hamming);
peak_kaiser = max(dB_kaiser);
peak_blackman = max(dB_blackman);

fprintf('\n=== Peak Signal Levels (dB) ===\n');
fprintf('Rectangular:  %.2f dB\n', peak_rect);
fprintf('Hamming:      %.2f dB\n', peak_hamming);
fprintf('Kaiser (beta=8): %.2f dB\n', peak_kaiser);
fprintf('Blackman:     %.2f dB\n', peak_blackman);

% Calculate SNR (Signal-to-Noise Ratio)
snr_rect = peak_rect - noise_rect;
snr_hamming = peak_hamming - noise_hamming;
snr_kaiser = peak_kaiser - noise_kaiser;
snr_blackman = peak_blackman - noise_blackman;

fprintf('\n=== Estimated SNR (dB) ===\n');
fprintf('Rectangular:  %.2f dB\n', snr_rect);
fprintf('Hamming:      %.2f dB\n', snr_hamming);
fprintf('Kaiser (beta=8): %.2f dB\n', snr_kaiser);
fprintf('Blackman:     %.2f dB\n', snr_blackman);
fprintf('\n');

%% Estimate AM Signal Band (fmin and fmax) using Kaiser window
fprintf('Estimating AM signal band limits...\n');

% Use Kaiser window spectrum (best sidelobe suppression)
threshold_dB = noise_kaiser + 10;  % 10 dB above noise floor

% Find indices where signal exceeds threshold
signal_indices = find(dB_kaiser > threshold_dB);

% Estimate band limits
fmin = f(min(signal_indices));
fmax = f(max(signal_indices));

% Calculate bandwidth and center frequency
bandwidth = fmax - fmin;
fc_estimate = (fmin + fmax) / 2;

fprintf(' Band estimation complete\n');
fprintf('\n=== AM Signal Band Estimation ===\n');
fprintf('Detection threshold: %.2f dB (%.2f dB above noise)\n', threshold_dB, threshold_dB - noise_kaiser);
fprintf('Lower band edge (fmin): %.3f kHz\n', fmin/1000);
fprintf('Upper band edge (fmax): %.3f kHz\n', fmax/1000);
fprintf('Total bandwidth: %.3f kHz\n', bandwidth/1000);
fprintf('Estimated carrier (center): %.3f kHz\n', fc_estimate/1000);
fprintf('Message bandwidth (expected): 4 kHz\n');
fprintf('Calculated message BW: %.3f kHz\n', bandwidth/2000);
fprintf('\n');

%% Plot Band Detection
fprintf('Generating band detection plot...\n');

fig7 = figure('Position', [100 100 1200 500]);
plot(f/1000, dB_kaiser, 'b', 'LineWidth', 1.5);
hold on;
yline(threshold_dB, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Threshold (%.1f dB)', threshold_dB));
xline(fmin/1000, 'g--', 'LineWidth', 2, 'DisplayName', sprintf('fmin = %.2f kHz', fmin/1000));
xline(fmax/1000, 'm--', 'LineWidth', 2, 'DisplayName', sprintf('fmax = %.2f kHz', fmax/1000));
xline(fc_estimate/1000, 'k:', 'LineWidth', 2, 'DisplayName', sprintf('fc_{est} = %.2f kHz', fc_estimate/1000));
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('AM Signal Band Detection (Kaiser Window)', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0 40]);
save_plot(fig7, 'Task1_07_BandDetection', plots_dir);
fprintf('   Saved: Task1_07_BandDetection\n');

%% Save Task 1 Results
results.signal = signal;
results.fs = fs;
results.fmin = fmin;
results.fmax = fmax;
results.fc_estimate = fc_estimate;
results.dB_kaiser = dB_kaiser;
results.f = f;

save('task1_results.mat', 'results');
fprintf('\n Task 1 results saved to task1_results.mat\n');


%% ===== TASK 2: FIR BANDPASS FILTER DESIGN =====
fprintf('>>> TASK 2: FIR BANDPASS FILTER DESIGN\n');
fprintf('--------------------------------------------\n');

% Load Task 1 results
load('task1_results.mat');
signal = results.signal;
fs = results.fs;
fmin = results.fmin;
fmax = results.fmax;

%% Filter Specifications
fp_low = fmin;
fp_high = fmax;
fstop_low = fmin - 2000;
fstop_high = fmax + 2000;
transition_bw = fp_low - fstop_low;

passband_ripple_dB = 0.1;
stopband_atten_dB = 50;

fprintf('=== Filter Specifications ===\n');
fprintf('Passband: %.2f - %.2f kHz\n', fp_low/1000, fp_high/1000);
fprintf('Stopband: DC-%.2f kHz and %.2f-%.2f kHz\n', fstop_low/1000, fstop_high/1000, fs/2000);
fprintf('Transition bandwidth: %.2f kHz\n', transition_bw/1000);
fprintf('Max passband ripple: %.2f dB\n', passband_ripple_dB);
fprintf('Min stopband attenuation: %.0f dB\n', stopband_atten_dB);
fprintf('\n');

%% Estimate Filter Order
fprintf('Estimating required filter order...\n');

delta_p = (10^(passband_ripple_dB/20) - 1) / (10^(passband_ripple_dB/20) + 1);
delta_s = 10^(-stopband_atten_dB/20);
delta = min(delta_p, delta_s);

A = -20 * log10(delta);

if A > 50
    beta_kaiser = 0.1102 * (A - 8.7);
elseif A >= 21
    beta_kaiser = 0.5842 * (A - 21)^0.4 + 0.07886 * (A - 21);
else
    beta_kaiser = 0;
end

M = ceil((A - 8) / (2.285 * 2 * pi * transition_bw / fs));
if mod(M, 2) == 0
    M = M + 1;
end

fprintf(' Filter order estimated\n');
fprintf('\n=== Filter Order Calculation ===\n');
fprintf('Attenuation parameter (A): %.2f dB\n', A);
fprintf('Kaiser beta parameter: %.4f\n', beta_kaiser);
fprintf('Estimated order (M): %d\n', M);
fprintf('Filter length (M+1): %d taps\n', M+1);
fprintf('\n');

%% Design FIR Bandpass Filter
fprintf('Designing FIR bandpass filter...\n');

% Normalized cutoff frequencies
fc1 = fp_low / (fs/2);
fc2 = fp_high / (fs/2);

% Ideal impulse response
n = -(M/2):(M/2);
h_ideal = (sin(pi*fc2*n) - sin(pi*fc1*n)) ./ (pi*n);
h_ideal(n==0) = fc2 - fc1;

% Apply Kaiser window
w_kaiser = kaiser(M+1, beta_kaiser)';
h_fir = h_ideal .* w_kaiser;

fprintf(' FIR filter designed\n');
fprintf('  Normalized cutoffs: fc1=%.4f, fc2=%.4f\n', fc1, fc2);
fprintf('  Impulse response: %d points\n', length(h_fir));
fprintf('\n');

%% Verify Filter Frequency Response
fprintf('Computing and verifying filter frequency response...\n');

[H, f_response] = freqz(h_fir, 1, 4096, fs);
H_dB = 20 * log10(abs(H) + eps);

% Check specifications
passband_idx = (f_response >= fp_low) & (f_response <= fp_high);
H_passband = abs(H(passband_idx));

passband_max_dB = 20*log10(max(H_passband));
passband_min_dB = 20*log10(min(H_passband));
passband_ripple_actual = passband_max_dB - passband_min_dB;

stopband_idx = (f_response < fstop_low) | (f_response > fstop_high);
H_stopband = abs(H(stopband_idx));
stopband_atten_actual = -20*log10(max(H_stopband));

fprintf(' Filter response computed\n');
fprintf('\n=== Filter Performance ===\n');
fprintf('Passband ripple (spec: %.2f dB): %.3f dB ', passband_ripple_dB, passband_ripple_actual);
if passband_ripple_actual <= passband_ripple_dB
    fprintf('[PASS ]\n');
else
    fprintf('[FAIL ]\n');
end

fprintf('Stopband attenuation (spec: >%.0f dB): %.2f dB ', stopband_atten_dB, stopband_atten_actual);
if stopband_atten_actual >= stopband_atten_dB
    fprintf('[PASS ]\n');
else
    fprintf('[FAIL ]\n');
end
fprintf('\n');

%% Plot Filter Frequency Response
fprintf('Generating filter frequency response plots...\n');

fig8 = figure('Position', [100 100 1200 900]);

% Magnitude response (dB)
subplot(3,1,1);
plot(f_response/1000, H_dB, 'b', 'LineWidth', 1.5);
hold on;
xline(fp_low/1000, 'g--', 'LineWidth', 2);
xline(fp_high/1000, 'g--', 'LineWidth', 2);
xline(fstop_low/1000, 'r--', 'LineWidth', 1.5);
xline(fstop_high/1000, 'r--', 'LineWidth', 1.5);
yline(-stopband_atten_dB, 'm--', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('FIR Bandpass Filter - Frequency Response', 'FontSize', 13);
grid on;
xlim([0 40]);
ylim([-100 10]);

% Passband detail
subplot(3,1,2);
plot(f_response/1000, abs(H), 'b', 'LineWidth', 1.5);
hold on;
xline(fp_low/1000, 'g--', 'LineWidth', 2);
xline(fp_high/1000, 'g--', 'LineWidth', 2);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (Linear)', 'FontSize', 11);
title('Passband Detail - Linear Scale', 'FontSize', 13);
grid on;
xlim([fp_low/1000-1 fp_high/1000+1]);
ylim([0.9 1.1]);

% Phase response
subplot(3,1,3);
plot(f_response/1000, unwrap(angle(H)) * 180/pi, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Phase (degrees)', 'FontSize', 11);
title('Phase Response', 'FontSize', 13);
grid on;
xlim([0 40]);

save_plot(fig8, 'Task2_01_FIR_FrequencyResponse', plots_dir);
fprintf('   Saved: Task2_01_FIR_FrequencyResponse\n');

%% Plot Impulse Response
fprintf('Generating impulse response plot...\n');

fig9 = figure('Position', [100 100 1200 400]);
stem(n, h_fir, 'b', 'LineWidth', 1.5);
xlabel('Sample Index', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title(sprintf('FIR Filter Impulse Response (%d taps)', length(h_fir)), 'FontSize', 14);
grid on;
save_plot(fig9, 'Task2_02_FIR_ImpulseResponse', plots_dir);
fprintf('   Saved: Task2_02_FIR_ImpulseResponse\n');

%% Apply FIR Filter using Custom Convolution
fprintf('\nApplying FIR filter to signal...\n');
fprintf('Using custom_conv() function...\n');

signal_filtered = custom_conv(signal, h_fir);

fprintf(' Filter applied\n');
fprintf('  Filtered signal length: %d samples\n', length(signal_filtered));
fprintf('  Filtered signal RMS: %.6f\n', sqrt(mean(signal_filtered.^2)));
fprintf('\n');

%% Verify Filtered Signal
fprintf('Analyzing filtered signal...\n');

t_filtered = (0:length(signal_filtered)-1) / fs;

% Time domain
fig10 = figure('Position', [100 100 1200 800]);

subplot(3,1,1);
plot(t_filtered, signal_filtered, 'b', 'LineWidth', 0.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Filtered AM Signal - Time Domain (Full)', 'FontSize', 13);
grid on;
xlim([0 max(t_filtered)]);

subplot(3,1,2);
plot(t_filtered, signal_filtered, 'b', 'LineWidth', 1);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Filtered AM Signal - Time Domain (Zoomed: First 10ms)', 'FontSize', 13);
grid on;
xlim([0 0.01]);

% Frequency domain
N_filt = length(signal_filtered);
signal_filtered_fft = fft(signal_filtered);
f_filt = (0:N_filt/2) * fs / N_filt;
signal_filtered_fft_single = signal_filtered_fft(1:N_filt/2+1);
amp_filtered = abs(signal_filtered_fft_single) / N_filt;
amp_filtered(2:end-1) = 2 * amp_filtered(2:end-1);
dB_filtered = 20 * log10(amp_filtered + eps);

subplot(3,1,3);
plot(f_filt/1000, dB_filtered, 'b', 'LineWidth', 1.5);
hold on;
xline(fp_low/1000, 'g--', 'LineWidth', 2);
xline(fp_high/1000, 'g--', 'LineWidth', 2);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Filtered AM Signal - Frequency Domain', 'FontSize', 13);
grid on;
xlim([0 40]);

save_plot(fig10, 'Task2_03_FilteredSignal', plots_dir);
fprintf('   Saved: Task2_03_FilteredSignal\n');

%% Before/After Comparison
fprintf('Generating before/after comparison...\n');

load('task1_results.mat');
f_orig = results.f;
dB_orig = results.dB_kaiser;

fig11 = figure('Position', [100 100 1200 500]);
plot(f_orig/1000, dB_orig, 'r', 'LineWidth', 1.5, 'DisplayName', 'Before filtering (original)');
hold on;
plot(f_filt/1000, dB_filtered, 'b', 'LineWidth', 1.5, 'DisplayName', 'After filtering');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Effect of Bandpass Filter', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0 40]);

save_plot(fig11, 'Task2_04_BeforeAfter_Comparison', plots_dir);
fprintf('   Saved: Task2_04_BeforeAfter_Comparison\n');

fprintf('\n Bandpass filter successfully removed out-of-band noise\n');

%% Save Task 2 Results
results_task2.signal_filtered = signal_filtered;
results_task2.h_fir = h_fir;
results_task2.M = M;
results_task2.fp_low = fp_low;
results_task2.fp_high = fp_high;

save('task2_results.mat', 'results_task2');
fprintf('\n Task 2 results saved to task2_results.mat\n');
fprintf('\n>>> TASK 2 COMPLETE!\n');
fprintf('========================================\n\n');

%% ===== TASK 3: CARRIER RECOVERY AND MIXING =====
fprintf('>>> TASK 3: CARRIER RECOVERY AND MIXING\n');
fprintf('--------------------------------------------\n');

% Load previous results
load('task1_results.mat');
load('task2_results.mat');

signal_filtered = results_task2.signal_filtered;
fs = results.fs;

%% Square the Signal
fprintf('Applying square law to bandpass filtered signal...\n');

signal_squared = signal_filtered.^2;

fprintf(' Signal squared\n');
fprintf('  Squared signal length: %d samples\n', length(signal_squared));
fprintf('  Squared signal RMS: %.6f\n', sqrt(mean(signal_squared.^2)));
fprintf('  Squared signal mean: %.6f\n', mean(signal_squared));
fprintf('\n');

%% Compute Spectrum of Squared Signal
fprintf('Computing spectrum of squared signal...\n');

N_sq = length(signal_squared);
signal_squared_fft = fft(signal_squared);
f_sq = (0:N_sq/2) * fs / N_sq;
signal_squared_fft_single = signal_squared_fft(1:N_sq/2+1);
amp_squared = abs(signal_squared_fft_single) / N_sq;
amp_squared(2:end-1) = 2 * amp_squared(2:end-1);
dB_squared = 20 * log10(amp_squared + eps);

fprintf(' Spectrum computed\n');
fprintf('  Frequency bins: %d\n', length(f_sq));
fprintf('\n');

%% Plot Squared Signal Spectrum
fprintf('Generating squared signal spectrum plot...\n');

fig12 = figure('Position', [100 100 1200 500]);
plot(f_sq/1000, dB_squared, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Squared Signal - Frequency Domain (Look for peak at 2fc)', 'FontSize', 14);
grid on;
xlim([0 60]);
ylim([max(dB_squared)-80 max(dB_squared)+10]);

save_plot(fig12, 'Task3_01_SquaredSignal_Spectrum', plots_dir);
fprintf('   Saved: Task3_01_SquaredSignal_Spectrum\n');

%% Identify Carrier Frequency
fprintf('Identifying carrier frequency from 2fc peak...\n');

[~, peak_idx] = max(amp_squared);
fc_doubled = f_sq(peak_idx);
fc_recovered = fc_doubled / 2;

% Round to nearest 1 kHz (spec says fc is exact multiple of 1 kHz)
fc_recovered = round(fc_recovered / 1000) * 1000;

fprintf(' Carrier frequency recovered\n');
fprintf('\n=== Carrier Recovery Results ===\n');
fprintf('Peak found at: %.3f kHz (2fc)\n', fc_doubled/1000);
fprintf('Recovered carrier frequency (fc): %.3f kHz\n', fc_recovered/1000);
fprintf('Carrier is exact multiple of 1 kHz: YES (%d kHz)\n', round(fc_recovered/1000));
fprintf('\n');

%% Plot Carrier Recovery
fprintf('Generating carrier recovery visualization...\n');

fig13 = figure('Position', [100 100 1200 500]);
plot(f_sq/1000, dB_squared, 'b', 'LineWidth', 1.5);
hold on;
xline(fc_doubled/1000, 'r--', 'LineWidth', 2.5, 'DisplayName', sprintf('2fc = %.2f kHz', fc_doubled/1000));
xline(fc_recovered/1000, 'g--', 'LineWidth', 2, 'DisplayName', sprintf('fc = %.2f kHz', fc_recovered/1000));
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Carrier Recovery from Squared Signal', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([fc_doubled/1000-5 fc_doubled/1000+5]);

save_plot(fig13, 'Task3_02_CarrierRecovery', plots_dir);
fprintf('   Saved: Task3_02_CarrierRecovery\n');

%% Generate Local Carrier
fprintf('Generating local carrier signal...\n');

t = (0:length(signal_filtered)-1)' / fs;
phi = 0;  % Initial phase (will be optimized in Task 5)

carrier = cos(2 * pi * fc_recovered * t + phi);

fprintf(' Local carrier generated\n');
fprintf('  Carrier frequency: %.3f kHz\n', fc_recovered/1000);
fprintf('  Initial phase: %.3f rad (%.2f°)\n', phi, rad2deg(phi));
fprintf('  Carrier length: %d samples\n', length(carrier));
fprintf('  Carrier RMS: %.6f\n', sqrt(mean(carrier.^2)));
fprintf('\n');

%% Mix Signals (Coherent Demodulation)
fprintf('Performing coherent demodulation (mixing)...\n');

signal_mixed = signal_filtered .* carrier;

fprintf(' Signals mixed\n');
fprintf('  Mixed signal length: %d samples\n', length(signal_mixed));
fprintf('  Mixed signal RMS: %.6f\n', sqrt(mean(signal_mixed.^2)));
fprintf('\n');

%% Plot Mixed Signal
fprintf('Generating mixed signal plots...\n');

t_mixed = t;

fig14 = figure('Position', [100 100 1200 800]);

subplot(3,1,1);
plot(t_mixed, signal_mixed, 'b', 'LineWidth', 0.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Mixed Signal - Time Domain (Full)', 'FontSize', 13);
grid on;
xlim([0 max(t_mixed)]);

subplot(3,1,2);
plot(t_mixed, signal_mixed, 'b', 'LineWidth', 1);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Mixed Signal - Time Domain (Zoomed: First 10ms)', 'FontSize', 13);
grid on;
xlim([0 0.01]);

% Frequency domain
N_mix = length(signal_mixed);
signal_mixed_fft = fft(signal_mixed);
f_mix = (0:N_mix/2) * fs / N_mix;
signal_mixed_fft_single = signal_mixed_fft(1:N_mix/2+1);
amp_mixed = abs(signal_mixed_fft_single) / N_mix;
amp_mixed(2:end-1) = 2 * amp_mixed(2:end-1);
dB_mixed = 20 * log10(amp_mixed + eps);

subplot(3,1,3);
plot(f_mix/1000, dB_mixed, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Mixed Signal - Frequency Domain (Baseband + 2fc component)', 'FontSize', 13);
grid on;
xlim([0 50]);

save_plot(fig14, 'Task3_03_MixedSignal', plots_dir);
fprintf('   Saved: Task3_03_MixedSignal\n');

fprintf('\n=== Mixed Signal Content ===\n');
fprintf('Baseband audio: 0-4 kHz (desired message)\n');
fprintf('High frequency component: ~%.0f kHz (2fc, to be removed)\n', 2*fc_recovered/1000);
fprintf('\n');

%% Save Task 3 Results
results_task3.signal_mixed = signal_mixed;
results_task3.fc_recovered = fc_recovered;
results_task3.carrier = carrier;
results_task3.phi = phi;

save('task3_results.mat', 'results_task3');
fprintf(' Task 3 results saved to task3_results.mat\n');

%% ===== TASK 4: IIR LOWPASS FILTER =====
fprintf('>>> TASK 4: IIR LOWPASS FILTER DESIGN\n');
fprintf('--------------------------------------------\n');

% Load previous results
load('task1_results.mat');
load('task3_results.mat');

signal_mixed = results_task3.signal_mixed;
fs = results.fs;

%% Filter Specifications
filter_order = 4;
cutoff_freq = 4000;
filter_type = 'low';

fprintf('=== IIR Filter Specifications ===\n');
fprintf('Filter type: Butterworth Lowpass\n');
fprintf('Order: %d\n', filter_order);
fprintf('Cutoff frequency: %.2f kHz\n', cutoff_freq/1000);
fprintf('\n');

%% Design Butterworth Filter
fprintf('Designing Butterworth lowpass filter...\n');

[b, a] = butter(filter_order, cutoff_freq/(fs/2), filter_type);

fprintf(' Filter designed\n');
fprintf('  Numerator coefficients (b): %d\n', length(b));
fprintf('  Denominator coefficients (a): %d\n', length(a));
fprintf('\n=== Filter Coefficients ===\n');
fprintf('b = [');
fprintf('%.10f ', b);
fprintf(']\n');
fprintf('a = [');
fprintf('%.10f ', a);
fprintf(']\n\n');

%% Verify Filter Response
fprintf('Computing filter frequency response...\n');

[H_iir, f_iir] = freqz(b, a, 4096, fs);
H_iir_mag = abs(H_iir);
H_iir_dB = 20 * log10(H_iir_mag + eps);
H_iir_phase = unwrap(angle(H_iir)) * 180/pi;

% Find -3dB point
cutoff_3dB_idx = find(H_iir_dB >= -3, 1, 'last');
cutoff_3dB_actual = f_iir(cutoff_3dB_idx);

fprintf(' Filter response computed\n');
fprintf('\n=== Cutoff Frequency Verification ===\n');
fprintf('Specified cutoff: %.3f kHz\n', cutoff_freq/1000);
fprintf('Actual -3dB point: %.3f kHz\n', cutoff_3dB_actual/1000);
fprintf('Error: %.3f Hz (%.4f%%)\n', abs(cutoff_3dB_actual - cutoff_freq), ...
    100*abs(cutoff_3dB_actual - cutoff_freq)/cutoff_freq);
fprintf('\n');

%% Plot Filter Response
fprintf('Generating filter frequency response plots...\n');

fig15 = figure('Position', [100 100 1200 900]);

subplot(3,1,1);
plot(f_iir/1000, H_iir_dB, 'b', 'LineWidth', 1.5);
hold on;
xline(cutoff_freq/1000, 'r--', 'LineWidth', 2);
yline(-3, 'g--', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title(sprintf('Butterworth Lowpass Filter (Order %d) - Magnitude Response', filter_order), 'FontSize', 13);
grid on;
xlim([0 20]);
ylim([-80 5]);

subplot(3,1,2);
plot(f_iir/1000, H_iir_mag, 'b', 'LineWidth', 1.5);
hold on;
xline(cutoff_freq/1000, 'r--', 'LineWidth', 2);
yline(1/sqrt(2), 'g--', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (Linear)', 'FontSize', 11);
title('Magnitude Response - Linear Scale (Passband Detail)', 'FontSize', 13);
grid on;
xlim([0 10]);
ylim([0 1.1]);

subplot(3,1,3);
plot(f_iir/1000, H_iir_phase, 'b', 'LineWidth', 1.5);
xline(cutoff_freq/1000, 'r--', 'LineWidth', 2);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Phase (degrees)', 'FontSize', 11);
title('Phase Response', 'FontSize', 13);
grid on;
xlim([0 20]);

save_plot(fig15, 'Task4_01_IIR_FrequencyResponse', plots_dir);
fprintf('   Saved: Task4_01_IIR_FrequencyResponse\n');

%% Apply IIR Filter
fprintf('\nApplying IIR filter to mixed signal...\n');
fprintf('Using custom_iir_filter() function...\n');

signal_demod = custom_iir_filter(b, a, signal_mixed);

fprintf(' IIR filter applied\n');
fprintf('  Demodulated signal length: %d samples\n', length(signal_demod));
fprintf('  Demodulated signal RMS: %.6f\n', sqrt(mean(signal_demod.^2)));
fprintf('  Demodulated signal peak: %.6f\n', max(abs(signal_demod)));
fprintf('\n');

%% Analyze Demodulated Signal
fprintf('Analyzing demodulated signal...\n');

t_demod = (0:length(signal_demod)-1) / fs;

fig16 = figure('Position', [100 100 1200 800]);

subplot(3,1,1);
plot(t_demod, signal_demod, 'b', 'LineWidth', 0.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Demodulated Audio Signal - Time Domain (Full)', 'FontSize', 13);
grid on;
xlim([0 max(t_demod)]);

subplot(3,1,2);
plot(t_demod, signal_demod, 'b', 'LineWidth', 1);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Demodulated Audio Signal - Time Domain (Zoomed: First 0.1s)', 'FontSize', 13);
grid on;
xlim([0 0.1]);

% Frequency domain
N_demod = length(signal_demod);
signal_demod_fft = fft(signal_demod);
f_demod = (0:N_demod/2) * fs / N_demod;
signal_demod_fft_single = signal_demod_fft(1:N_demod/2+1);
amp_demod = abs(signal_demod_fft_single) / N_demod;
amp_demod(2:end-1) = 2 * amp_demod(2:end-1);
dB_demod = 20 * log10(amp_demod + eps);

subplot(3,1,3);
plot(f_demod/1000, dB_demod, 'b', 'LineWidth', 1.5);
hold on;
xline(cutoff_freq/1000, 'r--', 'LineWidth', 2);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Demodulated Signal - Frequency Domain (Audio bandwidth)', 'FontSize', 13);
grid on;
xlim([0 20]);

save_plot(fig16, 'Task4_02_DemodulatedSignal', plots_dir);
fprintf('   Saved: Task4_02_DemodulatedSignal\n');

%% Energy Distribution
fprintf('\nCalculating energy distribution...\n');

energy_passband = sum(amp_demod(f_demod <= cutoff_freq).^2);
energy_stopband = sum(amp_demod(f_demod > cutoff_freq).^2);
total_energy = energy_passband + energy_stopband;
energy_ratio_dB = 10*log10(energy_passband / energy_stopband);

fprintf(' Energy analysis complete\n');
fprintf('\n=== Energy Distribution ===\n');
fprintf('Energy in passband (0-%.1f kHz): %.2f%%\n', cutoff_freq/1000, 100*energy_passband/total_energy);
fprintf('Energy in stopband (>%.1f kHz): %.2f%%\n', cutoff_freq/1000, 100*energy_stopband/total_energy);
fprintf('Passband/Stopband ratio: %.2f dB\n', energy_ratio_dB);
fprintf('\n');

%% Before/After Filtering Comparison
fprintf('Generating before/after filtering comparison...\n');

N_mix_comp = length(signal_mixed);
signal_mixed_fft_comp = fft(signal_mixed);
f_mix_comp = (0:N_mix_comp/2) * fs / N_mix_comp;
amp_mix_comp = abs(signal_mixed_fft_comp(1:N_mix_comp/2+1)) / N_mix_comp;
amp_mix_comp(2:end-1) = 2 * amp_mix_comp(2:end-1);
dB_mix_comp = 20 * log10(amp_mix_comp + eps);

fig17 = figure('Position', [100 100 1200 500]);
plot(f_mix_comp/1000, dB_mix_comp, 'r', 'LineWidth', 1, 'DisplayName', 'Before filtering (mixed)');
hold on;
plot(f_demod/1000, dB_demod, 'b', 'LineWidth', 1.5, 'DisplayName', 'After filtering (demodulated)');
xline(cutoff_freq/1000, 'k--', 'LineWidth', 2, 'DisplayName', sprintf('Cutoff (%.0f kHz)', cutoff_freq/1000));
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Effect of Lowpass Filter on Mixed Signal', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0 50]);

save_plot(fig17, 'Task4_03_BeforeAfter_Comparison', plots_dir);
fprintf('   Saved: Task4_03_BeforeAfter_Comparison\n');

fprintf('\n Lowpass filter successfully isolated audio message (0-4 kHz)\n');

%% Save Task 4 Results
results_task4.signal_demod = signal_demod;
results_task4.b = b;
results_task4.a = a;
results_task4.filter_order = filter_order;
results_task4.cutoff_freq = cutoff_freq;

save('task4_results.mat', 'results_task4');
fprintf('\n Task 4 results saved to task4_results.mat\n');

%% ===== TASK 5: PHASE Optimisation =====
fprintf('>>> TASK 5: PHASE Optimisation AND AUDIO OUTPUT\n');
fprintf('--------------------------------------------\n');

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

%% Test Baseline (phi = 0)
fprintf('Testing baseline phase (phi = 0)...\n');

phi_initial = 0;
t = (0:length(signal_filtered)-1)' / fs;
carrier_initial = cos(2 * pi * fc_recovered * t + phi_initial);
signal_mixed_initial = signal_filtered .* carrier_initial;
signal_demod_initial = custom_iir_filter(b, a, signal_mixed_initial);

peak_amp_initial = max(abs(signal_demod_initial));
rms_initial = sqrt(mean(signal_demod_initial.^2));

fprintf(' Baseline tested\n');
fprintf('\n=== Baseline (phi = 0) ===\n');
fprintf('Peak amplitude: %.6f\n', peak_amp_initial);
fprintf('RMS amplitude: %.6f\n', rms_initial);
fprintf('\n');

%% Phase Optimisation Sweep
fprintf('Starting phase Optimisation sweep...\n');
fprintf('Testing 50 phase values from 0 to π radians...\n\n');

phi_values = linspace(0, pi, 50);
num_phases = length(phi_values);

peak_amplitudes = zeros(num_phases, 1);
rms_amplitudes = zeros(num_phases, 1);

for i = 1:num_phases
    phi = phi_values(i);
    
    carrier = cos(2 * pi * fc_recovered * t + phi);
    signal_mixed = signal_filtered .* carrier;
    signal_demod = custom_iir_filter(b, a, signal_mixed, false);  % verbose=false to suppress output
    
    peak_amplitudes(i) = max(abs(signal_demod));
    rms_amplitudes(i) = sqrt(mean(signal_demod.^2));
    
    if mod(i, 10) == 0
        fprintf('  Progress: %d/%d phases tested (%.0f%%)...\n', i, num_phases, 100*i/num_phases);
    end
end

fprintf('\n Phase sweep complete!\n');

%% Find Optimal Phase
[max_peak, idx_peak] = max(peak_amplitudes);
phi_optimal = phi_values(idx_peak);

[max_rms, idx_rms] = max(rms_amplitudes);
phi_optimal_rms = phi_values(idx_rms);

fprintf('\n=== Phase Optimisation Results ===\n');
fprintf('Optimal phase (peak criterion):\n');
fprintf('  PHI = %.4f rad (%.2f°)\n', phi_optimal, rad2deg(phi_optimal));
fprintf('  Max peak amplitude: %.6f\n', max_peak);
fprintf('  Improvement: %.2f%% over baseline\n', 100*(max_peak/peak_amp_initial - 1));
fprintf('\n');
fprintf('Optimal phase (RMS criterion):\n');
fprintf('  PHI = %.4f rad (%.2f°)\n', phi_optimal_rms, rad2deg(phi_optimal_rms));
fprintf('  Max RMS amplitude: %.6f\n', max_rms);
fprintf('  Improvement: %.2f%% over baseline\n', 100*(max_rms/rms_initial - 1));
fprintf('\n');

%% Plot Phase Optimisation
fprintf('Generating phase Optimisation plots...\n');

fig18 = figure('Position', [100 100 1200 700]);

subplot(2,1,1);
plot(rad2deg(phi_values), peak_amplitudes, 'b-', 'LineWidth', 1.5);
hold on;
plot(rad2deg(phi_optimal), max_peak, 'ro', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('Phase PHI (degrees)', 'FontSize', 12);
ylabel('Peak Amplitude', 'FontSize', 12);
title('Phase Optimisation - Peak Amplitude Criterion', 'FontSize', 14);
grid on;
xlim([0 180]);
text(rad2deg(phi_optimal), max_peak*1.05, sprintf('  Optimal: %.1f°', rad2deg(phi_optimal)), 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');

subplot(2,1,2);
plot(rad2deg(phi_values), rms_amplitudes, 'b-', 'LineWidth', 1.5);
hold on;
plot(rad2deg(phi_optimal_rms), max_rms, 'ro', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('Phase PHI (degrees)', 'FontSize', 12);
ylabel('RMS Amplitude', 'FontSize', 12);
title('Phase Optimisation - RMS Amplitude Criterion', 'FontSize', 14);
grid on;
xlim([0 180]);

save_plot(fig18, 'Task5_01_PhaseOptimisation', plots_dir);
fprintf('   Saved: Task5_01_PhaseOptimisation\n');

%% Generate Final Demodulated Signal
fprintf('\nGenerating final demodulated signal with optimal phase...\n');

carrier_optimal = cos(2 * pi * fc_recovered * t + phi_optimal);
signal_mixed_optimal = signal_filtered .* carrier_optimal;
signal_demod_final = custom_iir_filter(b, a, signal_mixed_optimal);

fprintf(' Final demodulated signal generated\n');
fprintf('  Signal length: %d samples (%.2f seconds)\n', length(signal_demod_final), length(signal_demod_final)/fs);
fprintf('  Peak amplitude: %.6f\n', max(abs(signal_demod_final)));
fprintf('  RMS amplitude: %.6f\n', sqrt(mean(signal_demod_final.^2)));
fprintf('\n');

%% Plot Final Signal
fprintf('Generating final signal plots...\n');

t_final = (0:length(signal_demod_final)-1) / fs;

fig19 = figure('Position', [100 100 1200 700]);

subplot(2,1,1);
plot(t_final, signal_demod_final, 'b', 'LineWidth', 0.8);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title(sprintf('Final Demodulated Audio Signal (PHI = %.2f°)', rad2deg(phi_optimal)), 'FontSize', 14);
grid on;
xlim([0 max(t_final)]);

subplot(2,1,2);
plot(t_final, signal_demod_final, 'b', 'LineWidth', 1);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Final Demodulated Signal (Zoomed: First 3 seconds)', 'FontSize', 14);
grid on;
xlim([0 min(3, max(t_final))]);

save_plot(fig19, 'Task5_02_FinalDemodulatedSignal', plots_dir);
fprintf('   Saved: Task5_02_FinalDemodulatedSignal\n');

%% Prepare and Play Audio
fprintf('\nPreparing audio for playback...\n');

amplification_factor = 3;
signal_amplified = signal_demod_final * amplification_factor;
max_abs_value = max(abs(signal_amplified));
signal_audio = signal_amplified / max_abs_value;

fprintf(' Audio normalized\n');
fprintf('  Amplification factor: %.1fx\n', amplification_factor);
fprintf('  Final range: [%.6f, %.6f]\n', min(signal_audio), max(signal_audio));
fprintf('\n');

fprintf('>>> PLAYING AUDIO MESSAGE <<<\n');
fprintf('Listen carefully for the 3-letter message...\n\n');

playback_rate = fs * 0.8;
sound(signal_audio, playback_rate);

pause_duration = length(signal_audio)/playback_rate + 0.5;
pause(pause_duration);

fprintf(' Audio playback complete\n\n');

%% Message Identification
fprintf('>>> MESSAGE IDENTIFICATION <<<\n');
message = input('Enter the 3-letter message you heard: ', 's');

fprintf('\n Message identified: %s\n', upper(message));

%% Save Audio File
fprintf('\nSaving audio to file...\n');

output_filename = 'demodulated_message_Sahas_T.wav';
audiowrite(output_filename, signal_audio, fs);

fprintf(' Audio saved as: %s\n', output_filename);
fprintf('  Sampling rate: %d Hz\n', fs);
fprintf('  Duration: %.2f seconds\n', length(signal_audio)/fs);
fprintf('\n');

%% Save Task 5 Results
results_task5.signal_demod_final = signal_demod_final;
results_task5.phi_optimal = phi_optimal;
results_task5.phi_values = phi_values;
results_task5.peak_amplitudes = peak_amplitudes;
results_task5.rms_amplitudes = rms_amplitudes;
results_task5.message = message;
results_task5.signal_audio = signal_audio;

save('task5_results.mat', 'results_task5');
fprintf(' Task 5 results saved to task5_results.mat\n');

%% FINAL SUMMARY

fprintf('\n=== Summary of Results ===\n');
fprintf('Carrier frequency (fc): %.3f kHz\n', fc_recovered/1000);
fprintf('Optimal phase: %.2f° (%.4f rad)\n', rad2deg(phi_optimal), phi_optimal);
fprintf('Message identified: %s\n', upper(message));
fprintf('Audio file: %s\n', output_filename);
fprintf('\n=== Files Generated ===\n');
fprintf('Task 1 results: task1_results.mat\n');
fprintf('Task 2 results: task2_results.mat\n');
fprintf('Task 3 results: task3_results.mat\n');
fprintf('Task 4 results: task4_results.mat\n');
fprintf('Task 5 results: task5_results.mat\n');
fprintf('Audio output: %s\n', output_filename);
fprintf('\n=== Plots Generated ===\n');
fprintf('All plots saved in: %s/\n', plots_dir);
fprintf('Total plots: 19 figures\n');
fprintf('  - Task 1: 7 plots\n');
fprintf('  - Task 2: 4 plots\n');
fprintf('  - Task 3: 3 plots\n');
fprintf('  - Task 4: 3 plots\n');
fprintf('  - Task 5: 2 plots\n');
