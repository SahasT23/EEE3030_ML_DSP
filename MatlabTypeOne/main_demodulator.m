%% Load the AM signal
[signal, fs] = audioread('Sahas_Talasila.wav');
signal = signal(:);  % Ensure column vector

% Display basic information
fprintf('Sampling frequency: %d Hz\n', fs);
fprintf('Signal length: %d samples\n', length(signal));
fprintf('Duration: %.2f seconds\n', length(signal)/fs);

% Time Domain Analysis
t = (0:length(signal)-1) / fs;  % Time vector in seconds

% Plot full signal
figure('Position', [100 100 1200 400]);
plot(t, signal, 'b', 'LineWidth', 0.5);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Received AM Signal - Time Domain', 'FontSize', 14);
grid on;
xlim([0 max(t)]);

% Zoom in on first 10ms to see modulation envelope
figure('Position', [100 100 1200 400]);
plot(t, signal, 'b', 'LineWidth', 1);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Received AM Signal - Time Domain (Zoomed)', 'FontSize', 14);
grid on;
xlim([0 0.01]);  % First 10ms

% Frequency Domain Analysis - Manual FFT
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

%% Plot Frequency Spectrum
figure('Position', [100 100 1200 500]);
plot(f/1000, signal_dB, 'b', 'LineWidth', 1);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Received AM Signal - Frequency Domain', 'FontSize', 14);
grid on;
xlim([0 fs/2000]);  % Full range to Nyquist

% Zoom in on AM signal region (typically 10-30 kHz)
figure('Position', [100 100 1200 500]);
plot(f/1000, signal_dB, 'b', 'LineWidth', 1);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Received AM Signal - Frequency Domain (Zoomed)', 'FontSize', 14);
grid on;
xlim([5 35]);  % Adjust based on where you see signal
ylim([max(signal_dB)-80 max(signal_dB)+5]);  % 80 dB dynamic range

%% Apply Different Window Functions

% 1. Rectangular window (no windowing - baseline)
window_rect = ones(N, 1);
signal_rect = signal .* window_rect;

% 2. Hamming window
window_hamming = hamming(N);
signal_hamming = signal .* window_hamming;

% 3. Kaiser window (beta=8 for ~50dB sidelobe suppression)
beta = 8;
window_kaiser = kaiser(N, beta);
signal_kaiser = signal .* window_kaiser;

% 4. Blackman window
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

% Calculate window correction factors
corr_rect = sum(window_rect) / N;
corr_hamming = sum(window_hamming) / N;
corr_kaiser = sum(window_kaiser) / N;
corr_blackman = sum(window_blackman) / N;

% Amplitude normalization with window correction
amp_rect = abs(fft_rect_single) / N / corr_rect;
amp_hamming = abs(fft_hamming_single) / N / corr_hamming;
amp_kaiser = abs(fft_kaiser_single) / N / corr_kaiser;
amp_blackman = abs(fft_blackman_single) / N / corr_blackman;

% Double non-DC/Nyquist components
amp_rect(2:end-1) = 2 * amp_rect(2:end-1);
amp_hamming(2:end-1) = 2 * amp_hamming(2:end-1);
amp_kaiser(2:end-1) = 2 * amp_kaiser(2:end-1);
amp_blackman(2:end-1) = 2 * amp_blackman(2:end-1);

% Convert to dB
dB_rect = 20 * log10(amp_rect + eps);
dB_hamming = 20 * log10(amp_hamming + eps);
dB_kaiser = 20 * log10(amp_kaiser + eps);
dB_blackman = 20 * log10(amp_blackman + eps);

%% Compare Window Effects
% figure('Position', [100 100 1400 800]);

% % Zoomed to AM signal region
% plot(f/1000, dB_rect, 'b', 'LineWidth', 1.5, 'DisplayName', 'Rectangular');
% hold on;
% plot(f/1000, dB_hamming, 'r', 'LineWidth', 1.5, 'DisplayName', 'Hamming');
% plot(f/1000, dB_kaiser, 'g', 'LineWidth', 1.5, 'DisplayName', sprintf('Kaiser (β=%.0f)', beta));
% plot(f/1000, dB_blackman, 'm', 'LineWidth', 1.5, 'DisplayName', 'Blackman');
% xlabel('Frequency (kHz)', 'FontSize', 12);
% ylabel('Magnitude (dB)', 'FontSize', 12);
% title('Window Comparison - AM Signal Band', 'FontSize', 14);
% legend('Location', 'best', 'FontSize', 10);
% grid on;
% xlim([5 35]);  
% ylim([max(dB_hamming)-80 max(dB_hamming)+5]);

%% Estimate Carrier Frequency from Spectrum

% Calculate noise floor and threshold
noise_hamming = median(dB_hamming);
threshold_hamming = noise_hamming + 20;  % 20 dB above noise

% Detect signal bandwidth
detect_hamming = find(dB_hamming > threshold_hamming);

if ~isempty(detect_hamming)
    fmin_detected = f(detect_hamming(1));
    fmax_detected = f(detect_hamming(end));
    
    % Round to nearest 100 Hz
    fmin_detected = round(fmin_detected/100) * 100;
    fmax_detected = round(fmax_detected/100) * 100;
    
    % Estimate carrier frequency (center of detected band)
    fc_estimated = (fmin_detected + fmax_detected) / 2;
    
    fprintf('\n--- Detected Signal Band ---\n');
    fprintf('fmin (detected): %.2f kHz (%.0f Hz)\n', fmin_detected/1000, fmin_detected);
    fprintf('fmax (detected): %.2f kHz (%.0f Hz)\n', fmax_detected/1000, fmax_detected);
    fprintf('fc (estimated): %.2f kHz (%.0f Hz)\n', fc_estimated/1000, fc_estimated);
end

%% Set Bandpass Filter Edges - IMPORTANT
% The message bandwidth B = 4kHz
% Therefore, bandpass filter should be fc ± 4kHz

fmin = fc_estimated - 3300;  % fc - 4kHz
fmax = fc_estimated + 3300;  % fc + 4kHz

fprintf('\n=== Bandpass Filter Design Parameters ===\n');
fprintf('Carrier frequency (fc): %.0f Hz (%.2f kHz)\n', fc_estimated, fc_estimated/1000);
fprintf('Bandpass filter edges:\n');
fprintf('  fmin = fc - 4kHz = %.0f Hz (%.2f kHz)\n', fmin, fmin/1000);
fprintf('  fmax = fc + 4kHz = %.0f Hz (%.2f kHz)\n', fmax, fmax/1000);
fprintf('  Bandwidth = %.0f Hz (%.2f kHz)\n', fmax-fmin, (fmax-fmin)/1000);

%% Save Task 1 Results
results.signal = signal;
results.fs = fs;
results.fmin = fmin;  % fc - 4kHz
results.fmax = fmax;  % fc + 4kHz
results.fc_estimated = fc_estimated;
results.signal_dB = dB_hamming;
results.f = f;

save('task1_results.mat', 'results');
fprintf('\nTask 1 results saved to task1_results.mat\n');

%% TASK 2: FIR Bandpass Filter Design
load('task1_results.mat');
signal = results.signal;
fs = results.fs;
fmin = results.fmin;  % fc - 4kHz
fmax = results.fmax;  % fc + 4kHz

fprintf('\n=== TASK 2: FIR Bandpass Filter Design ===\n');
fprintf('Passband: %.0f Hz to %.0f Hz\n', fmin, fmax);

%% Filter Specifications
% Passband edges (from Task 1)
fp1 = fmin;  % Lower passband edge (fc - 4kHz)
fp2 = fmax;  % Upper passband edge (fc + 4kHz)

% Stopband edges (assignment specification)
fstop1 = fmin - 2000;  % Lower stopband edge
fstop2 = fmax + 2000;  % Upper stopband edge

% Performance requirements
passband_ripple_dB = 0.1;
stopband_atten_dB = 50;

% Transition bandwidth
transition_bw = fp1 - fstop1;  % = 2000 Hz

fprintf('\nFilter Specifications:\n');
fprintf('Passband: %.0f Hz to %.0f Hz\n', fp1, fp2);
fprintf('Stopband: DC-%.0f Hz and %.0f Hz-Nyquist\n', fstop1, fstop2);
fprintf('Transition bandwidth: %.0f Hz\n', transition_bw);
fprintf('Passband ripple: %.1f dB\n', passband_ripple_dB);
fprintf('Stopband attenuation: %.0f dB\n', stopband_atten_dB);

%% Estimate Filter Order
% Using Hamming window formula: N ≈ 3.3 * fs / transition_bw
N_estimated = ceil(6.0 * fs / transition_bw);

% Make it odd for Type I FIR (symmetric)
if mod(N_estimated, 2) == 0
    N_estimated = N_estimated + 1;
end

fprintf('\nEstimated filter order: %d taps\n', N_estimated);

M = N_estimated;  % Filter length

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
        n = i - M_center - 1;
        h_ideal(i) = (sin(pi * wc2 * n) - sin(pi * wc1 * n)) / (pi * n);
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
% Compute frequency response (use FFT with zero-padding)
N_fft = 8192;  % Zero-padding for smooth frequency response
H = fft(h_fir, N_fft);
H_mag = abs(H(1:N_fft/2+1));
H_dB = 20 * log10(H_mag + eps);

% Frequency vector
f_response = (0:N_fft/2) * fs / N_fft;

% Plot frequency response
figure('Position', [100 100 1000 600]);
plot(f_response/1000, H_dB, 'b', 'LineWidth', 1.5);
hold on;

% Mark specifications
yline(-passband_ripple_dB, 'g--', 'LineWidth', 1.5, 'Label', 'Passband ripple');
yline(-stopband_atten_dB, 'r--', 'LineWidth', 1.5, 'Label', 'Stopband spec');
xline(fp1/1000, 'k--', 'LineWidth', 1);
xline(fp2/1000, 'k--', 'LineWidth', 1);
xline(fstop1/1000, 'r--', 'LineWidth', 1);
xline(fstop2/1000, 'r--', 'LineWidth', 1);

xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('FIR Bandpass Filter Frequency Response', 'FontSize', 14);
grid on;
xlim([0 fs/2000]);
ylim([-80 5]);

%% Apply FIR Filter via Custom Convolution
fprintf('\nApplying FIR filter via custom convolution...\n');

signal_filtered = custom_conv(signal, h_fir);

fprintf('Filtering complete. Output length: %d samples\n', length(signal_filtered));

%% Analyze Filtered Signal
N_sig = length(signal_filtered);

% Apply Hamming window for spectral analysis
window_analysis = hamming(N_sig);
signal_filtered_windowed = signal_filtered .* window_analysis;

% Compute FFT
signal_filtered_fft = fft(signal_filtered_windowed);
f_filtered = (0:N_sig/2) * fs / N_sig;
signal_filtered_fft_single = signal_filtered_fft(1:N_sig/2+1);

% Amplitude calculation with window correction
window_corr = sum(window_analysis) / N_sig;
amp_filtered = abs(signal_filtered_fft_single) / N_sig / window_corr;
amp_filtered(2:end-1) = 2 * amp_filtered(2:end-1);
dB_filtered = 20 * log10(amp_filtered + eps);

% Plot filtered signal spectrum
figure('Position', [100 100 1000 600]);
plot(f_filtered/1000, dB_filtered, 'b', 'LineWidth', 1.5);
hold on;
xline(fp1/1000, 'g--', 'LineWidth', 2, 'Label', 'fmin');
xline(fp2/1000, 'r--', 'LineWidth', 2, 'Label', 'fmax');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Filtered Signal Spectrum', 'FontSize', 14);
grid on;
xlim([0 fs/2000]);

%% Save Task 2 Results
results_task2.signal_filtered = signal_filtered;
results_task2.h_fir = h_fir;
results_task2.filter_order = M;
results_task2.fp1 = fp1;
results_task2.fp2 = fp2;

save('task2_results.mat', 'results_task2');
fprintf('Task 2 results saved to task2_results.mat\n\n');

%% TASK 3: Carrier Recovery
load('task1_results.mat');
load('task2_results.mat');

signal_filtered = results_task2.signal_filtered;
fs = results.fs;
fc_estimated = results.fc_estimated;

fprintf('\n=== TASK 3: Carrier Recovery ===\n');
fprintf('Initial carrier estimate: %.2f Hz\n', fc_estimated);

%% Square the Filtered Signal
signal_squared = signal_filtered .^ 2;

fprintf('Signal squared (| |²)\n');

%% Compute FFT of Squared Signal with High Resolution
N_squared = length(signal_squared);

% Apply window
window = hamming(N_squared);
signal_squared_windowed = signal_squared .* window;

% Zero-padding for higher frequency resolution
N_fft = 2^nextpow2(N_squared * 8);  % 8x zero padding
df = fs / N_fft;

fprintf('FFT size: %d points\n', N_fft);
fprintf('Frequency resolution: %.2f Hz\n', df);

% Compute FFT
squared_fft = fft(signal_squared_windowed, N_fft);
f_fft = (0:N_fft/2) * df;
squared_fft_single = squared_fft(1:N_fft/2+1);
squared_mag = abs(squared_fft_single);

%% Find Peak at 2fc
% Search region: ±1 kHz around 2*fc_estimated
search_margin = 1000;  % Hz
fc2_min = 2 * fc_estimated - search_margin;
fc2_max = 2 * fc_estimated + search_margin;

% Find indices in search region
search_idx = find(f_fft >= fc2_min & f_fft <= fc2_max);

% Find peak
[~, peak_idx_local] = max(squared_mag(search_idx));
peak_idx = search_idx(peak_idx_local);

% Recovered carrier frequency (divide by 2)
fc_recovered = f_fft(peak_idx) / 2;

fprintf('\n--- Carrier Recovery Results ---\n');
fprintf('Peak at 2fc: %.2f Hz\n', f_fft(peak_idx));
fprintf('Recovered carrier frequency: %.2f Hz (%.2f kHz)\n', fc_recovered, fc_recovered/1000);
fprintf('Difference from estimate: %.2f Hz\n', fc_recovered - fc_estimated);

%% Plot Squared Signal Spectrum
figure('Position', [100 100 1000 600]);
plot(f_fft/1000, 20*log10(squared_mag + eps), 'b', 'LineWidth', 1);
hold on;
xline(f_fft(peak_idx)/1000, 'r--', 'LineWidth', 2);
text(f_fft(peak_idx)/1000, max(20*log10(squared_mag + eps))-5, ...
    sprintf('  2fc = %.2f kHz', f_fft(peak_idx)/1000), 'Color', 'r', 'FontWeight', 'bold');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Squared Signal Spectrum - Carrier Recovery', 'FontSize', 14);
grid on;
xlim([fc2_min/1000-1, fc2_max/1000+1]);

%% Generate Local Carrier
% Time vector
t = (0:length(signal_filtered)-1)' / fs;

% Generate recovered carrier with phi = 0 (will optimize in Task 5)
phi = 0;
carrier = cos(2 * pi * fc_recovered * t + phi);

fprintf('\nLocal carrier generated\n');
fprintf('Frequency: %.2f Hz\n', fc_recovered);
fprintf('Phase: %.2f rad (%.2f°)\n', phi, rad2deg(phi));

%% Mix (Multiply) Signal with Carrier
signal_mixed = signal_filtered .* carrier;

fprintf('Signal mixed with carrier\n');

%% Analyze Mixed Signal Spectrum
N_mixed = length(signal_mixed);
window_mixed = hamming(N_mixed);
signal_mixed_windowed = signal_mixed .* window_mixed;

mixed_fft = fft(signal_mixed_windowed);
f_mixed = (0:N_mixed/2) * fs / N_mixed;
mixed_fft_single = mixed_fft(1:N_mixed/2+1);

window_corr_mixed = sum(window_mixed) / N_mixed;
amp_mixed = abs(mixed_fft_single) / N_mixed / window_corr_mixed;
amp_mixed(2:end-1) = 2 * amp_mixed(2:end-1);
dB_mixed = 20 * log10(amp_mixed + eps);

% Plot mixed signal spectrum
figure('Position', [100 100 1000 600]);
plot(f_mixed/1000, dB_mixed, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Mixed Signal Spectrum (Before Lowpass Filter)', 'FontSize', 14);
grid on;
xlim([0 50]);

%% Save Task 3 Results
results_task3.fc_recovered = fc_recovered;
results_task3.signal_mixed = signal_mixed;
results_task3.carrier = carrier;

save('task3_results.mat', 'results_task3');
fprintf('\nTask 3 results saved to task3_results.mat\n\n');

%% TASK 4: IIR Lowpass Filter Design
load('task1_results.mat');
load('task2_results.mat');
load('task3_results.mat');

signal_mixed = results_task3.signal_mixed;
fs = results.fs;

fprintf('\n=== TASK 4: IIR Lowpass Filter Design ===\n');

%% Design IIR Lowpass Filter
% Filter specifications
filter_order = 4;         % 4th order
cutoff_freq = 4000;       % 4 kHz cutoff

% Normalize cutoff frequency (relative to Nyquist)
Wn = cutoff_freq / (fs/2);

fprintf('Filter Specifications:\n');
fprintf('Type: Butterworth\n');
fprintf('Order: %d\n', filter_order);
fprintf('Cutoff frequency: %.0f Hz\n', cutoff_freq);
fprintf('Normalized cutoff: %.4f\n', Wn);

% Design filter using butter()
[b, a] = butter(filter_order, Wn, 'low');

fprintf('\nFilter coefficients:\n');
fprintf('Numerator (b): ');
fprintf('%.6f ', b);
fprintf('\n');
fprintf('Denominator (a): ');
fprintf('%.6f ', a);
fprintf('\n');

%% Verify Filter Frequency Response
[H, f_response] = freqz(b, a, 8192, fs);
H_dB = 20 * log10(abs(H));

figure('Position', [100 100 1000 600]);
plot(f_response/1000, H_dB, 'b', 'LineWidth', 1.5);
hold on;
yline(-3, 'r--', 'LineWidth', 2, 'Label', '-3dB (cutoff)');
xline(cutoff_freq/1000, 'k--', 'LineWidth', 2);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('IIR Lowpass Filter Frequency Response', 'FontSize', 14);
grid on;
xlim([0 20]);
ylim([-80 5]);

%% Apply IIR Filter via Custom Implementation
fprintf('\nApplying custom IIR filter...\n');

% signal_demod = custom_iir_filter(b, a, signal_mixed);

% First lowpass stage
signal_demod_stage1 = custom_iir_filter(b, a, signal_mixed);

% Second lowpass stage to remove remaining high-frequency noise
[b2, a2] = butter(6, 3200/(fs/2), 'low');
signal_demod = filter(b2, a2, signal_demod_stage1);

fprintf('Demodulation complete\n');

%% Analyze Demodulated Signal
N_demod = length(signal_demod);
window_demod = hamming(N_demod);
signal_demod_windowed = signal_demod .* window_demod;

demod_fft = fft(signal_demod_windowed);
f_demod = (0:N_demod/2) * fs / N_demod;
demod_fft_single = demod_fft(1:N_demod/2+1);

window_corr_demod = sum(window_demod) / N_demod;
amp_demod = abs(demod_fft_single) / N_demod / window_corr_demod;
amp_demod(2:end-1) = 2 * amp_demod(2:end-1);
dB_demod = 20 * log10(amp_demod + eps);

% Plot demodulated signal spectrum
figure('Position', [100 100 1000 600]);
plot(f_demod/1000, dB_demod, 'b', 'LineWidth', 1.5);
hold on;
xline(cutoff_freq/1000, 'r--', 'LineWidth', 2, 'Label', 'Cutoff (4 kHz)');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Demodulated Signal - Frequency Domain', 'FontSize', 14);
grid on;
xlim([0 20]);

%% Energy Distribution Check
energy_passband = sum(amp_demod(f_demod <= cutoff_freq).^2);
energy_stopband = sum(amp_demod(f_demod > cutoff_freq).^2);
energy_ratio_dB = 10*log10(energy_passband / energy_stopband);

fprintf('\n--- Energy Distribution ---\n');
fprintf('Passband (0-4 kHz): %.2f%%\n', 100*energy_passband/(energy_passband+energy_stopband));
fprintf('Stopband (>4 kHz): %.2f%%\n', 100*energy_stopband/(energy_passband+energy_stopband));
fprintf('Passband/Stopband ratio: %.1f dB\n', energy_ratio_dB);

%% Save Task 4 Results
results_task4.signal_demod = signal_demod;
results_task4.b = b;
results_task4.a = a;
results_task4.filter_order = filter_order;
results_task4.cutoff_freq = cutoff_freq;

save('task4_results.mat', 'results_task4');
fprintf('\nTask 4 results saved to task4_results.mat\n\n');

%% TASK 5: Phase Optimization and Audio Output
load('task1_results.mat');
load('task2_results.mat');
load('task3_results.mat');
load('task4_results.mat');

signal_filtered = results_task2.signal_filtered;
fc_recovered = results_task3.fc_recovered;
b = results_task4.b;
a = results_task4.a;
fs = results.fs;

fprintf('\n=== TASK 5: Phase Optimization ===\n');

%% Test Initial Phase (φ = 0)
phi_initial = 0;

% Time vector
t = (0:length(signal_filtered)-1)' / fs;

% Generate carrier with φ = 0
carrier_initial = cos(2 * pi * fc_recovered * t + phi_initial);

% Mix
signal_mixed_initial = signal_filtered .* carrier_initial;

% Filter
signal_demod_initial = custom_iir_filter(b, a, signal_mixed_initial);

% Calculate metrics
peak_amp_initial = max(abs(signal_demod_initial));
rms_initial = sqrt(mean(signal_demod_initial.^2));

fprintf('Initial phase (φ = 0):\n');
fprintf('  Peak amplitude: %.4f\n', peak_amp_initial);
fprintf('  RMS amplitude: %.4f\n', rms_initial);

%% Phase Optimization - Sweep from 0 to π
fprintf('\nPhase sweep from 0 to π radians...\n');

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

fprintf('\n--- Optimal Phase Results ---\n');
fprintf('Based on peak amplitude:\n');
fprintf('  φ_optimal = %.4f rad (%.2f°)\n', phi_optimal, rad2deg(phi_optimal));
fprintf('  Maximum peak: %.4f\n', max_peak);
fprintf('  Improvement over φ=0: %.2f%%\n', 100*(max_peak/peak_amp_initial - 1));

fprintf('\nBased on RMS amplitude:\n');
fprintf('  φ_optimal = %.4f rad (%.2f°)\n', phi_optimal_rms, rad2deg(phi_optimal_rms));
fprintf('  Maximum RMS: %.4f\n', max_rms);

%% Plot Phase Optimization Curve
figure('Position', [100 100 1000 600]);

subplot(2,1,1);
plot(rad2deg(phi_values), peak_amplitudes, 'b-', 'LineWidth', 1.5);
hold on;
plot(rad2deg(phi_optimal), max_peak, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Phase φ (degrees)', 'FontSize', 12);
ylabel('Peak Amplitude', 'FontSize', 12);
title('Phase Optimization - Peak Amplitude', 'FontSize', 14);
grid on;
text(rad2deg(phi_optimal), max_peak*1.05, ...
    sprintf('  Optimal: %.1f°', rad2deg(phi_optimal)), ...
    'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');

subplot(2,1,2);
plot(rad2deg(phi_values), rms_amplitudes, 'b-', 'LineWidth', 1.5);
hold on;
plot(rad2deg(phi_optimal_rms), max_rms, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Phase φ (degrees)', 'FontSize', 12);
ylabel('RMS Amplitude', 'FontSize', 12);
title('Phase Optimization - RMS Amplitude', 'FontSize', 14);
grid on;

%% Generate Final Demodulated Signal with Optimal Phase
fprintf('\nGenerating final demodulated signal...\n');

% Use optimal phase (from peak amplitude)
carrier_optimal = cos(2 * pi * fc_recovered * t + phi_optimal);

% Mix
signal_mixed_optimal = signal_filtered .* carrier_optimal;

% Filter
signal_demod_final = custom_iir_filter(b, a, signal_mixed_optimal);

fprintf('Final signal generated\n');
fprintf('Duration: %.2f seconds\n', length(signal_demod_final)/fs);

%% Plot Final Demodulated Signal
t_final = (0:length(signal_demod_final)-1) / fs;

figure('Position', [100 100 1000 500]);
plot(t_final, signal_demod_final, 'b', 'LineWidth', 0.8);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title(sprintf('Final Demodulated Audio (φ = %.2f°)', rad2deg(phi_optimal)), 'FontSize', 14);
grid on;
xlim([0 max(t_final)]);

% Zoomed view
figure('Position', [100 100 1000 500]);
plot(t_final, signal_demod_final, 'b', 'LineWidth', 1);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Final Demodulated Signal (Zoomed)', 'FontSize', 14);
grid on;
xlim([0 min(3, max(t_final))]);

%% Normalize and Play Audio
% Normalize to [-1, 1] range
signal_audio = signal_demod_final / max(abs(signal_demod_final));

fprintf('\n=== Playing Audio ===\n');
fprintf('Listen carefully for the 3-letter message...\n');

% Play audio
sound(signal_audio, fs);

% Wait for playback to finish
pause(length(signal_audio)/fs + 1);

fprintf('Playback complete\n');

%% Identify the 3-Letter Message
fprintf('\n=== Message Identification ===\n');
message = input('Enter the 3-letter message you heard: ', 's');

fprintf('\nIdentified message: %s\n', upper(message));

%% Save Audio to File
output_filename = 'demodulated_message_Sahas_T.wav';

audiowrite(output_filename, signal_audio, fs);

fprintf('\nAudio saved as: %s\n', output_filename);
fprintf('\n=== ALL TASKS COMPLETE ===\n');