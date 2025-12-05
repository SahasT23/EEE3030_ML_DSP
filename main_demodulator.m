%% Sub-task 1.1: Reading the Audio File and Basic Signal Information
% EEE3030 DSP Assignment - Task 1

clear all;
close all;
clc;

%% Read the audio file
% Replace 'your_signal.wav' with your actual filename
filename = 'Sahas_Talasila.wav';
[x, fs] = audioread(filename);

% If stereo, convert to mono by taking first channel
if size(x, 2) > 1
    x = x(:, 1);
    fprintf('Note: Stereo file detected, using first channel only.\n\n');
end

%% Calculate basic signal properties
N = length(x);                      % Total number of samples
duration = N / fs;                  % Signal duration in seconds
freq_resolution = fs / N;           % Frequency resolution (bin width) in Hz
nyquist_freq = fs / 2;              % Maximum representable frequency

%% Display signal information
fprintf('SIGNAL PROPERTIES SUMMARY\n');
fprintf('Filename:              %s\n', filename);
fprintf('Sampling frequency:    %d Hz\n', fs);
fprintf('Number of samples:     %d\n', N);
fprintf('Signal duration:       %.4f seconds\n', duration);
fprintf('Frequency resolution:  %.4f Hz\n', freq_resolution);
fprintf('Nyquist frequency:     %d Hz\n', nyquist_freq);

%% Calculate amplitude statistics
max_amplitude = max(x);
min_amplitude = min(x);
peak_to_peak = max_amplitude - min_amplitude;
rms_amplitude = sqrt(mean(x.^2));

fprintf('AMPLITUDE STATISTICS\n');
fprintf('Maximum amplitude:     %.6f\n', max_amplitude);
fprintf('Minimum amplitude:     %.6f\n', min_amplitude);
fprintf('Peak-to-peak:          %.6f\n', peak_to_peak);
fprintf('RMS amplitude:         %.6f\n', rms_amplitude);

%% Sub-task 1.2: Time Domain Analysis

% Create time vector
t = (0:N-1) / fs;

% Plot full signal
figure('Name', 'Time Domain - Full Signal', 'Position', [100, 100, 1200, 400]);
plot(t, x, 'b', 'LineWidth', 0.3);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Time Domain Signal - Full Duration');
grid on;
xlim([0, duration]);

% Plot zoomed section to see carrier oscillations
% Adjust zoom_start based on where interesting content appears
% zoom_start = 0.1;  % Start time in seconds (adjust as needed)
% zoom_duration = 0.005;  % 5 ms window to see carrier cycles

% figure('Name', 'Time Domain - Zoomed', 'Position', [100, 100, 1200, 400]);
% plot(t, x, 'b', 'LineWidth', 0.5);
% xlabel('Time (seconds)');
% ylabel('Amplitude');
% title(sprintf('Time Domain Signal - Zoomed (%.1f ms window)', zoom_duration*1000));
% grid on;
% xlim([zoom_start, zoom_start + zoom_duration]);

fprintf('TIME DOMAIN OBSERVATIONS\n');
fprintf('Signal duration:       %.4f seconds\n', duration);
% fprintf('Zoom window:           %.1f ms starting at %.2f s\n', zoom_duration*1000, zoom_start);

%% Sub-task 1.3: Frequency Domain Analysis - FFT Implementation

% Compute the FFT of the signal
X = fft(x);

% The FFT output is complex - compute the magnitude
X_magnitude = abs(X);

% Normalise by dividing by N to get correct amplitude scaling
X_normalised = X_magnitude / N;

% Create single-sided spectrum (positive frequencies only)
% We need bins from 0 (DC) to N/2 (Nyquist)
num_bins_single_sided = floor(N/2) + 1;
X_single_sided = X_normalised(1:num_bins_single_sided);

% Double the amplitude for all bins except DC and Nyquist
% This accounts for the energy in the negative frequencies we discarded
X_single_sided(2:end-1) = 2 * X_single_sided(2:end-1);

% Create the frequency vector for the single-sided spectrum
% Each bin k corresponds to frequency f = k * fs / N
f = (0:num_bins_single_sided-1) * fs / N;

% Convert to decibels for logarithmic scaling
% We add eps (smallest positive number) to avoid log(0) = -infinity
X_dB = 20 * log10(X_single_sided + eps);

% Create figure for the frequency spectrum
figure('Name', 'Frequency Spectrum - Linear Amplitude', 'Position', [100, 100, 1200, 500]);

% Plot with linear amplitude scaling first
plot(f/1000, X_single_sided, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Amplitude (normalised)', 'FontSize', 12);
title('Frequency Spectrum of Received AM Signal - Linear Amplitude Scale', 'FontSize', 14);
grid on;
xlim([0, fs/2000]);  % Display from 0 to Nyquist frequency in kHz

% Create figure for the frequency spectrum in dB
figure('Name', 'Frequency Spectrum - dB Scale', 'Position', [100, 100, 1200, 500]);

% Plot with logarithmic (dB) scaling
plot(f/1000, X_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Frequency Spectrum of Received AM Signal - Logarithmic (dB) Scale', 'FontSize', 14);
grid on;
xlim([0, fs/2000]);  % Display from 0 to Nyquist frequency in kHz

% Set a reasonable lower limit for dB scale to avoid showing very low noise values
ylim([-120, max(X_dB) + 10]);

% Display frequency domain statistics
fprintf('FREQUENCY DOMAIN ANALYSIS\n');
fprintf('FFT length (N):            %d bins\n', N);
fprintf('Frequency resolution:      %.4f Hz\n', fs/N);
fprintf('Nyquist frequency:         %d Hz\n', fs/2);
fprintf('Single-sided spectrum bins: %d\n', num_bins_single_sided);
fprintf('--------------------------------------------\n');
fprintf('Maximum spectral amplitude: %.6f\n', max(X_single_sided));
fprintf('Maximum magnitude (dB):     %.2f dB\n', max(X_dB));
fprintf('--------------------------------------------\n');

% Find the frequency bin with maximum energy (likely near carrier)
[max_amplitude, max_bin_index] = max(X_single_sided);
freq_at_max = f(max_bin_index);
fprintf('Frequency at max amplitude: %.2f Hz (%.2f kHz)\n', freq_at_max, freq_at_max/1000);

% Estimate approximate signal band by finding where energy is above noise floor
% This is a rough estimate - visual inspection will be more accurate
noise_floor_estimate_dB = median(X_dB);  % Median gives robust estimate of noise floor
signal_threshold_dB = noise_floor_estimate_dB + 10;  % 10 dB above noise floor

fprintf('NOISE FLOOR ESTIMATION\n');
fprintf('Estimated noise floor:     %.2f dB\n', noise_floor_estimate_dB);
fprintf('Signal threshold (+10dB):  %.2f dB\n', signal_threshold_dB);

%% Sub-task 1.4: Spectral Leakage and Windowing

% First, let's visualise the window functions to understand their properties

% Create a sample index vector for window visualisation
n_window = (0:N-1)';

% Generate Rectangular window (for comparison - this is what we implicitly used before)
rectangular_window = ones(N, 1);

% Generate Hamming window using the formula from course notes
% w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
hamming_window = 0.54 - 0.46 * cos(2 * pi * n_window / (N - 1));

% Generate Hanning window for comparison
% w[n] = 0.5 - 0.5 * cos(2*pi*n / (N-1))
hanning_window = 0.5 - 0.5 * cos(2 * pi * n_window / (N - 1));

% Generate Blackman window for comparison
% w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
blackman_window = 0.42 - 0.5 * cos(2 * pi * n_window / (N - 1)) + 0.08 * cos(4 * pi * n_window / (N - 1));

% Calculate coherent gain for each window (needed for amplitude correction)
CG_rectangular = sum(rectangular_window) / N;
CG_hamming = sum(hamming_window) / N;
CG_hanning = sum(hanning_window) / N;
CG_blackman = sum(blackman_window) / N;

% Display window properties
fprintf('WINDOW FUNCTION PROPERTIES\n');
fprintf('Window Type      | Coherent Gain | Correction Factor\n');
fprintf('-----------------+---------------+------------------\n');
fprintf('Rectangular      | %.6f      | %.6f\n', CG_rectangular, 1/CG_rectangular);
fprintf('Hamming          | %.6f      | %.6f\n', CG_hamming, 1/CG_hamming);
fprintf('Hanning          | %.6f      | %.6f\n', CG_hanning, 1/CG_hanning);
fprintf('Blackman         | %.6f      | %.6f\n', CG_blackman, 1/CG_blackman);

% Plot the window functions (showing only a portion for clarity)
% We'll show the first 1000 samples to see the window shape
samples_to_show = min(1000, N);

figure('Name', 'Window Functions', 'Position', [100, 100, 1200, 500]);
plot(0:samples_to_show-1, rectangular_window(1:samples_to_show), 'b', 'LineWidth', 1.5);
hold on;
plot(0:samples_to_show-1, hamming_window(1:samples_to_show), 'r', 'LineWidth', 1.5);
plot(0:samples_to_show-1, hanning_window(1:samples_to_show), 'g', 'LineWidth', 1.5);
plot(0:samples_to_show-1, blackman_window(1:samples_to_show), 'm', 'LineWidth', 1.5);
hold off;
xlabel('Sample Index (n)', 'FontSize', 12);
ylabel('Window Amplitude', 'FontSize', 12);
title('Comparison of Window Functions (First 1000 Samples)', 'FontSize', 14);
legend('Rectangular', 'Hamming', 'Hanning', 'Blackman', 'Location', 'east');
grid on;
xlim([0, samples_to_show-1]);
ylim([0, 1.1]);

%% Apply Hamming window to the signal

% Multiply signal by Hamming window (element-wise)
x_hamming = x .* hamming_window;

% Compute FFT of windowed signal
X_hamming = fft(x_hamming);

% Compute magnitude
X_hamming_magnitude = abs(X_hamming);

% Normalise by N and correct for window coherent gain
X_hamming_normalised = X_hamming_magnitude / N / CG_hamming;

% Create single-sided spectrum
X_hamming_single = X_hamming_normalised(1:num_bins_single_sided);
X_hamming_single(2:end-1) = 2 * X_hamming_single(2:end-1);

% Convert to dB
X_hamming_dB = 20 * log10(X_hamming_single + eps);

%% Also compute Hanning and Blackman for comparison

% Hanning windowed signal
x_hanning = x .* hanning_window;
X_hanning = fft(x_hanning);
X_hanning_magnitude = abs(X_hanning);
X_hanning_normalised = X_hanning_magnitude / N / CG_hanning;
X_hanning_single = X_hanning_normalised(1:num_bins_single_sided);
X_hanning_single(2:end-1) = 2 * X_hanning_single(2:end-1);
X_hanning_dB = 20 * log10(X_hanning_single + eps);

% Blackman windowed signal
x_blackman = x .* blackman_window;
X_blackman = fft(x_blackman);
X_blackman_magnitude = abs(X_blackman);
X_blackman_normalised = X_blackman_magnitude / N / CG_blackman;
X_blackman_single = X_blackman_normalised(1:num_bins_single_sided);
X_blackman_single(2:end-1) = 2 * X_blackman_single(2:end-1);
X_blackman_dB = 20 * log10(X_blackman_single + eps);

%% Plot comparison: Rectangular vs Hamming windowed spectrum

figure('Name', 'Spectrum Comparison: Rectangular vs Hamming', 'Position', [100, 100, 1200, 600]);

subplot(2,1,1);
plot(f/1000, X_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Frequency Spectrum - Rectangular Window (No Windowing Applied)', 'FontSize', 14);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_dB) + 10]);

subplot(2,1,2);
plot(f/1000, X_hamming_dB, 'r', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Frequency Spectrum - Hamming Window Applied', 'FontSize', 14);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_hamming_dB) + 10]);

%% Plot all three windows for comparison

figure('Name', 'Spectrum Comparison: All Windows', 'Position', [100, 100, 1200, 800]);

subplot(4,1,1);
plot(f/1000, X_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 10);
ylabel('Magnitude (dB)', 'FontSize', 10);
title('Rectangular Window', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_dB) + 10]);

subplot(4,1,2);
plot(f/1000, X_hanning_dB, 'g', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 10);
ylabel('Magnitude (dB)', 'FontSize', 10);
title('Hanning Window (44 dB Stopband Attenuation)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_hanning_dB) + 10]);

subplot(4,1,3);
plot(f/1000, X_hamming_dB, 'r', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 10);
ylabel('Magnitude (dB)', 'FontSize', 10);
title('Hamming Window (53 dB Stopband Attenuation)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_hamming_dB) + 10]);

subplot(4,1,4);
plot(f/1000, X_blackman_dB, 'm', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 10);
ylabel('Magnitude (dB)', 'FontSize', 10);
title('Blackman Window (74 dB Stopband Attenuation)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_blackman_dB) + 10]);

%% Display spectral statistics comparison

fprintf('SPECTRAL COMPARISON\n');
fprintf('Window Type      | Max (dB)  | Noise Floor Est (dB)\n');
fprintf('-----------------+-----------+--------------------\n');
fprintf('Rectangular      | %8.2f  | %8.2f\n', max(X_dB), median(X_dB));
fprintf('Hanning          | %8.2f  | %8.2f\n', max(X_hanning_dB), median(X_hanning_dB));
fprintf('Hamming          | %8.2f  | %8.2f\n', max(X_hamming_dB), median(X_hamming_dB));
fprintf('Blackman         | %8.2f  | %8.2f\n', max(X_blackman_dB), median(X_blackman_dB));

% Find peak frequency for each window (should be similar)
[~, max_idx_rect] = max(X_single_sided);
[~, max_idx_hamming] = max(X_hamming_single);
[~, max_idx_hanning] = max(X_hanning_single);
[~, max_idx_blackman] = max(X_blackman_single);

fprintf('PEAK FREQUENCY DETECTION\n');
fprintf('Window Type      | Peak Frequency (Hz)\n');
fprintf('-----------------+--------------------\n');
fprintf('Rectangular      | %.2f\n', f(max_idx_rect));
fprintf('Hanning          | %.2f\n', f(max_idx_hanning));
fprintf('Hamming          | %.2f\n', f(max_idx_hamming));
fprintf('Blackman         | %.2f\n', f(max_idx_blackman));

%% Sub-task 1.5: Identifying the AM Signal Band

% We will use the Hamming-windowed spectrum for this analysis
% X_hamming_dB and f were computed in previous sub-tasks

% Plot the full spectrum with the Hamming window
figure('Name', 'AM Signal Band Identification', 'Position', [100, 100, 1200, 600]);

% Full spectrum plot
subplot(2,1,1);
plot(f/1000, X_hamming_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Full Spectrum - Hamming Window (Identify Signal Band Region)', 'FontSize', 14);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_hamming_dB) + 10]);

% Calculate and display noise floor reference line
noise_floor_dB = median(X_hamming_dB);
hold on;
yline(noise_floor_dB, 'r--', 'LineWidth', 1.5);
yline(noise_floor_dB + 10, 'g--', 'LineWidth', 1.5);
legend('Spectrum', 'Noise Floor (median)', 'Noise Floor + 10 dB', 'Location', 'northeast');
hold off;

% Focused plot around the signal region
% ADJUST THESE VALUES based on your observation of the full spectrum
% Initial estimate - modify after seeing your spectrum
f_centre_estimate = 16000;  % Hz - CHANGE THIS based on your signal
plot_bandwidth = 15000;     % Hz - width of region to display

subplot(2,1,2);
plot(f/1000, X_hamming_dB, 'b', 'LineWidth', 0.5);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('Focused View of AM Signal Band', 'FontSize', 14);
grid on;
xlim([(f_centre_estimate - plot_bandwidth/2)/1000, (f_centre_estimate + plot_bandwidth/2)/1000]);
ylim([-100, max(X_hamming_dB) + 10]);

hold on;
yline(noise_floor_dB, 'r--', 'LineWidth', 1.5);
yline(noise_floor_dB + 10, 'g--', 'LineWidth', 1.5);
hold off;

%% Manual identification of signal band

fmin_measured = 12000;  % Hz - lower edge of AM signal band
fmax_measured = 20000;  % Hz - upper edge of AM signal band

% Calculate derived parameters
fc_estimated = (fmin_measured + fmax_measured) / 2;
bandwidth_measured = fmax_measured - fmin_measured;

% Round carrier to nearest 1 kHz (as per assignment specification)
fc_rounded = round(fc_estimated / 1000) * 1000;

% Recalculate fmin and fmax based on rounded carrier and 8 kHz bandwidth
fmin_final = fc_rounded - 4000;
fmax_final = fc_rounded + 4000;

% Display results
fprintf('       AM SIGNAL BAND IDENTIFICATION       \n');
fprintf('Noise floor estimate:      %.2f dB\n', noise_floor_dB);
fprintf('--------------------------------------------\n');
fprintf('MEASURED VALUES (from visual inspection):\n');
fprintf('  f_min (measured):        %d Hz (%.1f kHz)\n', fmin_measured, fmin_measured/1000);
fprintf('  f_max (measured):        %d Hz (%.1f kHz)\n', fmax_measured, fmax_measured/1000);
fprintf('  Bandwidth (measured):    %d Hz (%.1f kHz)\n', bandwidth_measured, bandwidth_measured/1000);
fprintf('  f_c (calculated):        %.1f Hz (%.2f kHz)\n', fc_estimated, fc_estimated/1000);
fprintf('--------------------------------------------\n');
fprintf('FINAL VALUES (carrier rounded to 1 kHz):\n');
fprintf('  f_c (rounded):           %d Hz (%.1f kHz)\n', fc_rounded, fc_rounded/1000);
fprintf('  f_min (final):           %d Hz (%.1f kHz)\n', fmin_final, fmin_final/1000);
fprintf('  f_max (final):           %d Hz (%.1f kHz)\n', fmax_final, fmax_final/1000);
fprintf('  Bandwidth (final):       %d Hz (%.1f kHz)\n', fmax_final - fmin_final, (fmax_final - fmin_final)/1000);

%% Verification plot with identified band marked

figure('Name', 'Verified AM Signal Band', 'Position', [100, 100, 1200, 500]);
plot(f/1000, X_hamming_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('AM Signal Spectrum with Identified Band Boundaries', 'FontSize', 14);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_hamming_dB) + 10]);

hold on;
% Mark the final fmin and fmax
xline(fmin_final/1000, 'r-', 'LineWidth', 2);
xline(fmax_final/1000, 'r-', 'LineWidth', 2);
xline(fc_rounded/1000, 'g--', 'LineWidth', 2);

% Add shaded region for signal band
y_limits = ylim;
patch([fmin_final/1000, fmax_final/1000, fmax_final/1000, fmin_final/1000], ...
      [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
      'green', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

legend('Spectrum', 'f_{min}', 'f_{max}', 'f_c (carrier)', 'Signal Band', 'Location', 'northeast');
hold off;

%% Save key variables for Task 2
% These will be used for filter design
fprintf('PARAMETERS FOR TASK 2\n');
fprintf('Bandpass filter passband edges:\n');
fprintf('  f_p1 (lower passband):   %d Hz\n', fmin_final);
fprintf('  f_p2 (upper passband):   %d Hz\n', fmax_final);
fprintf('Bandpass filter stopband edges:\n');
fprintf('  f_s1 (lower stopband):   %d Hz\n', fmin_final - 2000);
fprintf('  f_s2 (upper stopband):   %d Hz\n', fmax_final + 2000);

%% Sub-task 2.1: FIR Bandpass Filter Design
% Using Impulse Response Truncation (IRT) method from course notes

%% Define filter specifications
fc = 16000;              % Carrier frequency in Hz (from Task 1)
fmin = fc - 4000;        % Lower passband edge (Hz)
fmax = fc + 4000;        % Upper passband edge (Hz)

% Stopband edges (as specified in assignment)
fstop_lower = fmin - 2000;   % Lower stopband edge (Hz)
fstop_upper = fmax + 2000;   % Upper stopband edge (Hz)

% Transition bandwidth
transition_bandwidth = 2000;  % Hz

% Calculate normalised frequencies
% Cutoff frequencies are at the centre of the transition bands
Fc1 = (fmin - 1000) / fs;    % Lower cutoff (normalised)
Fc2 = (fmax + 1000) / fs;    % Upper cutoff (normalised)

% Calculate normalised transition width
delta_F = transition_bandwidth / fs;

% Display specifications
fprintf('FIR BANDPASS FILTER SPECIFICATIONS\n');
fprintf('Sampling frequency:        %d Hz\n', fs);
fprintf('--------------------------------------------\n');
fprintf('FREQUENCY SPECIFICATIONS:\n');
fprintf('  Lower stopband edge:     %d Hz\n', fstop_lower);
fprintf('  Lower passband edge:     %d Hz\n', fmin);
fprintf('  Upper passband edge:     %d Hz\n', fmax);
fprintf('  Upper stopband edge:     %d Hz\n', fstop_upper);
fprintf('  Transition bandwidth:    %d Hz\n', transition_bandwidth);
fprintf('--------------------------------------------\n');
fprintf('NORMALISED FREQUENCIES:\n');
fprintf('  Fc1 (lower cutoff):      %.6f\n', Fc1);
fprintf('  Fc2 (upper cutoff):      %.6f\n', Fc2);
fprintf('  Delta F (transition):    %.6f\n', delta_F);
fprintf('--------------------------------------------\n');
fprintf('PERFORMANCE SPECIFICATIONS:\n');
fprintf('  Max passband ripple:     0.1 dB\n');
fprintf('  Min stopband atten:      50 dB\n');

%% Calculate required filter length
% Using Hamming window: transition width = 3.3/N
% Therefore N = 3.3 / delta_F

N_calculated = 3.3 / delta_F;
N = ceil(N_calculated);

% Ensure N is odd for symmetric filter
if mod(N, 2) == 0
    N = N + 1;
end

M = (N - 1) / 2;  % Number of coefficients either side of centre

fprintf('FILTER LENGTH CALCULATION\n');
fprintf('Window function:           Hamming\n');
fprintf('Transition width formula:  3.3/N\n');
fprintf('Calculated N:              %.2f\n', N_calculated);
fprintf('Rounded N (odd):           %d\n', N);
fprintf('M (half-length):           %d\n', M);

%% Design the ideal bandpass impulse response
% h_BP[n] = 2*Fc2*sinc(2*Fc2*n) - 2*Fc1*sinc(2*Fc1*n)

% Create coefficient index vector
% n ranges from -M to +M (centred at 0)
n_ideal = -M:M;

% Calculate ideal impulse response for bandpass filter
% Handle n=0 case separately to avoid division by zero
h_ideal = zeros(1, N);

for i = 1:N
    n = n_ideal(i);
    if n == 0
        % For n = 0: h[0] = 2*Fc2 - 2*Fc1
        h_ideal(i) = 2*Fc2 - 2*Fc1;
    else
        % For n != 0: h[n] = 2*Fc2*sinc(2*Fc2*n) - 2*Fc1*sinc(2*Fc1*n)
        % sinc(x) = sin(pi*x)/(pi*x), but here we use sin(2*pi*Fc*n)/(2*pi*Fc*n)
        term1 = 2*Fc2 * sin(n * 2*pi*Fc2) / (n * 2*pi*Fc2);
        term2 = 2*Fc1 * sin(n * 2*pi*Fc1) / (n * 2*pi*Fc1);
        h_ideal(i) = term1 - term2;
    end
end

fprintf('IDEAL IMPULSE RESPONSE\n');
fprintf('Centre coefficient h[M]:   %.6f\n', h_ideal(M+1));
fprintf('First coefficient h[0]:    %.6f\n', h_ideal(1));
fprintf('Last coefficient h[N-1]:   %.6f\n', h_ideal(N));
fprintf('Sum of coefficients:       %.6f\n', sum(h_ideal));

%% Generate Hamming window
% w[n] = 0.54 - 0.46*cos(2*pi*n/(N-1)) for n = 0, 1, ..., N-1

n_window = 0:N-1;
hamming_win = 0.54 - 0.46 * cos(2 * pi * n_window / (N - 1));

%% Apply window to ideal impulse response
h_windowed = h_ideal .* hamming_win;

fprintf('WINDOWED IMPULSE RESPONSE\n');
fprintf('Window type:               Hamming\n');
fprintf('Centre coefficient:        %.6f\n', h_windowed(M+1));
fprintf('First coefficient:         %.6f\n', h_windowed(1));
fprintf('Last coefficient:          %.6f\n', h_windowed(N));
fprintf('Sum of coefficients:       %.6f\n', sum(h_windowed));

%% Plot the filter design process

figure('Name', 'FIR Filter Design', 'Position', [100, 100, 1200, 800]);

% Plot 1: Ideal impulse response (unwindowed)
subplot(3,1,1);
stem(n_ideal, h_ideal, 'b', 'LineWidth', 0.5, 'MarkerSize', 3);
xlabel('Sample Index (n)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Ideal Bandpass Impulse Response (Unwindowed)', 'FontSize', 12);
grid on;
xlim([-M-5, M+5]);

% Plot 2: Hamming window
subplot(3,1,2);
stem(n_ideal, hamming_win, 'r', 'LineWidth', 0.5, 'MarkerSize', 3);
xlabel('Sample Index (n)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Hamming Window Function', 'FontSize', 12);
grid on;
xlim([-M-5, M+5]);
ylim([0, 1.1]);

% Plot 3: Windowed impulse response (final filter coefficients)
subplot(3,1,3);
stem(n_ideal, h_windowed, 'g', 'LineWidth', 0.5, 'MarkerSize', 3);
xlabel('Sample Index (n)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Final FIR Filter Coefficients (Windowed Impulse Response)', 'FontSize', 12);
grid on;
xlim([-M-5, M+5]);

%% Store filter coefficients for later use
h_bp = h_windowed;  % Final bandpass filter coefficients

fprintf('FILTER DESIGN COMPLETE\n');
fprintf('Filter coefficients stored in: h_bp\n');
fprintf('Number of taps:            %d\n', length(h_bp));
fprintf('Filter delay:              %d samples (%.4f ms)\n', M, M/fs*1000);

%% Sub-task 2.2: Frequency Response Verification

% Compute frequency response using zero-padded FFT
N_fft = 8192;  % Zero-pad for smooth frequency response plot
H = fft(h_bp, N_fft);
H_magnitude = abs(H);
H_dB = 20 * log10(H_magnitude + eps);
H_phase = angle(H);

% Unwrap phase for clearer visualisation
H_phase_unwrapped = unwrap(H_phase);

% Create frequency vector (single-sided)
f_response = (0:N_fft/2) * fs / N_fft;
H_dB_single = H_dB(1:N_fft/2+1);
H_phase_single = H_phase_unwrapped(1:N_fft/2+1);

%% Plot frequency response - Magnitude

figure('Name', 'Filter Frequency Response - Magnitude', 'Position', [100, 100, 1200, 600]);

% Full spectrum view
subplot(2,1,1);
plot(f_response/1000, H_dB_single, 'b', 'LineWidth', 1);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('FIR Bandpass Filter - Magnitude Response (Full Spectrum)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-100, 10]);

% Add specification lines
hold on;
xline(fmin/1000, 'g--', 'LineWidth', 1.5);
xline(fmax/1000, 'g--', 'LineWidth', 1.5);
xline(fstop_lower/1000, 'r--', 'LineWidth', 1.5);
xline(fstop_upper/1000, 'r--', 'LineWidth', 1.5);
yline(-50, 'm--', 'LineWidth', 1.5);
yline(-0.1, 'c--', 'LineWidth', 1);
yline(0.1, 'c--', 'LineWidth', 1);
legend('Response', 'f_{min}', 'f_{max}', 'f_{stop,lower}', 'f_{stop,upper}', ...
       '-50 dB spec', 'Passband ripple limits', 'Location', 'southwest');
hold off;

% Passband detail view
subplot(2,1,2);
plot(f_response/1000, H_dB_single, 'b', 'LineWidth', 1);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('FIR Bandpass Filter - Passband Detail', 'FontSize', 12);
grid on;
xlim([(fmin-3000)/1000, (fmax+3000)/1000]);
ylim([-80, 5]);

hold on;
xline(fmin/1000, 'g--', 'LineWidth', 1.5);
xline(fmax/1000, 'g--', 'LineWidth', 1.5);
xline(fstop_lower/1000, 'r--', 'LineWidth', 1.5);
xline(fstop_upper/1000, 'r--', 'LineWidth', 1.5);
yline(-50, 'm--', 'LineWidth', 1.5);
yline(0, 'k-', 'LineWidth', 0.5);
hold off;

%% Plot phase response

figure('Name', 'Filter Frequency Response - Phase', 'Position', [100, 100, 1200, 400]);

plot(f_response/1000, H_phase_single, 'b', 'LineWidth', 1);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Phase (radians)', 'FontSize', 11);
title('FIR Bandpass Filter - Phase Response', 'FontSize', 12);
grid on;
xlim([(fmin-3000)/1000, (fmax+3000)/1000]);

hold on;
xline(fmin/1000, 'g--', 'LineWidth', 1.5);
xline(fmax/1000, 'g--', 'LineWidth', 1.5);
legend('Phase', 'f_{min}', 'f_{max}', 'Location', 'southwest');
hold off;

%% Measure actual filter performance

% Find indices for passband and stopband regions
passband_indices = find(f_response >= fmin & f_response <= fmax);
stopband_lower_indices = find(f_response <= fstop_lower);
stopband_upper_indices = find(f_response >= fstop_upper & f_response <= fs/2);

% Measure passband ripple
passband_gain_dB = H_dB_single(passband_indices);
passband_max = max(passband_gain_dB);
passband_min = min(passband_gain_dB);
passband_ripple = passband_max - passband_min;

% Measure stopband attenuation
stopband_lower_max = max(H_dB_single(stopband_lower_indices));
stopband_upper_max = max(H_dB_single(stopband_upper_indices));
stopband_max = max(stopband_lower_max, stopband_upper_max);

% Calculate group delay (should be constant = M for linear phase)
group_delay_samples = M;
group_delay_ms = M / fs * 1000;

%% Display verification results

fprintf('FILTER VERIFICATION RESULTS\n');
fprintf('PASSBAND PERFORMANCE:\n');
fprintf('  Frequency range:         %d - %d Hz\n', fmin, fmax);
fprintf('  Maximum gain:            %.4f dB\n', passband_max);
fprintf('  Minimum gain:            %.4f dB\n', passband_min);
fprintf('  Peak-to-peak ripple:     %.4f dB\n', passband_ripple);
fprintf('  Specification:           < 0.1 dB\n');
if passband_ripple < 0.1
    fprintf('  Status:                  PASS\n');
else
    fprintf('  Status:                  FAIL\n');
end
fprintf('--------------------------------------------\n');
fprintf('STOPBAND PERFORMANCE:\n');
fprintf('  Lower stopband max:      %.2f dB\n', stopband_lower_max);
fprintf('  Upper stopband max:      %.2f dB\n', stopband_upper_max);
fprintf('  Worst-case attenuation:  %.2f dB\n', -stopband_max);
fprintf('  Specification:           > 50 dB\n');
if stopband_max < -50
    fprintf('  Status:                  PASS\n');
else
    fprintf('  Status:                  FAIL\n');
end
fprintf('--------------------------------------------\n');
fprintf('PHASE RESPONSE:\n');
fprintf('  Filter type:             Linear phase (symmetric coefficients)\n');
fprintf('  Group delay:             %d samples (%.4f ms)\n', group_delay_samples, group_delay_ms);

%% Overall verification summary

fprintf('VERIFICATION SUMMARY\n');
if passband_ripple < 0.1 && stopband_max < -50
    fprintf('  Filter meets all specifications.\n');
    fprintf('  Ready to apply to AM signal.\n');
else
    fprintf('  Filter does not meet specifications.\n');
    fprintf('  Consider increasing filter order (N).\n');
end

%% Sub-task 2.3: Verify Custom Convolution Implementation

fprintf('CUSTOM CONVOLUTION VERIFICATION\n');

% Create test signal
test_length = 1000;
test_signal = randn(1, test_length);
test_filter = ones(1, 10) / 10;  % Simple moving average

% Compare custom implementation with MATLAB's conv()
y_custom = custom_conv(test_signal, test_filter);
y_matlab = conv(test_signal, test_filter, 'same');

% Calculate error
max_error = max(abs(y_custom - y_matlab));
mean_error = mean(abs(y_custom - y_matlab));

fprintf('Test signal length:        %d samples\n', test_length);
fprintf('Test filter length:        %d taps\n', length(test_filter));
fprintf('Maximum absolute error:    %.2e\n', max_error);
fprintf('Mean absolute error:       %.2e\n', mean_error);

if max_error < 1e-10
    fprintf('Status:                    PASS (matches MATLAB conv)\n');
else
    fprintf('Status:                    CHECK IMPLEMENTATION\n');
end

% Time comparison
num_iterations = 10;

tic;
for i = 1:num_iterations
    y_temp = custom_conv(test_signal, test_filter);
end
time_custom = toc / num_iterations;

tic;
for i = 1:num_iterations
    y_temp = conv(test_signal, test_filter, 'same');
end
time_matlab = toc / num_iterations;

fprintf('TIMING COMPARISON\n');
fprintf('Custom implementation:     %.4f seconds\n', time_custom);
fprintf('MATLAB conv():             %.4f seconds\n', time_matlab);
fprintf('Ratio (custom/MATLAB):     %.1fx\n', time_custom/time_matlab);

%% Sub-task 2.4: Apply Bandpass Filter to AM Signal

fprintf('APPLYING BANDPASS FILTER\n');

% Apply the bandpass filter using custom convolution function
fprintf('Filtering signal using custom_conv() function\n');
tic;
x_filtered = custom_conv(x, h_bp);
filter_time = toc;

fprintf('Filtering complete.\n');
fprintf('Time taken:                %.4f seconds\n', filter_time);
fprintf('Input signal length:       %d samples\n', length(x));
fprintf('Filter length:             %d taps\n', length(h_bp));
fprintf('Output signal length:      %d samples\n', length(x_filtered));

%% Time domain comparison

figure('Name', 'Bandpass Filter - Time Domain', 'Position', [100, 100, 1200, 700]);

% Original signal
subplot(2,1,1);
plot(t, x, 'b', 'LineWidth', 0.3);
xlabel('Time (seconds)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Original Signal (Before Bandpass Filtering)', 'FontSize', 12);
grid on;
xlim([0, duration]);

% Filtered signal
subplot(2,1,2);
plot(t, x_filtered, 'r', 'LineWidth', 0.3);
xlabel('Time (seconds)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Filtered Signal (After Bandpass Filtering)', 'FontSize', 12);
grid on;
xlim([0, duration]);

%% Frequency domain comparison

% Compute spectrum of filtered signal using Hamming window
x_filtered_windowed = x_filtered .* hamming_window;
X_filtered = fft(x_filtered_windowed);
X_filtered_magnitude = abs(X_filtered);
X_filtered_normalised = X_filtered_magnitude / N / CG_hamming;
X_filtered_single = X_filtered_normalised(1:num_bins_single_sided);
X_filtered_single(2:end-1) = 2 * X_filtered_single(2:end-1);
X_filtered_dB = 20 * log10(X_filtered_single + eps);

figure('Name', 'Bandpass Filter - Frequency Domain', 'Position', [100, 100, 1200, 700]);

% Original spectrum
subplot(2,1,1);
plot(f/1000, X_hamming_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Original Signal Spectrum (Before Bandpass Filtering)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_hamming_dB) + 10]);

hold on;
xline(fmin/1000, 'g--', 'LineWidth', 1.5);
xline(fmax/1000, 'g--', 'LineWidth', 1.5);
legend('Spectrum', 'f_{min}', 'f_{max}', 'Location', 'northeast');
hold off;

% Filtered spectrum
subplot(2,1,2);
plot(f/1000, X_filtered_dB, 'r', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Filtered Signal Spectrum (After Bandpass Filtering)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_filtered_dB) + 10]);

hold on;
xline(fmin/1000, 'g--', 'LineWidth', 1.5);
xline(fmax/1000, 'g--', 'LineWidth', 1.5);
legend('Spectrum', 'f_{min}', 'f_{max}', 'Location', 'northeast');
hold off;

%% Calculate noise reduction statistics

% Measure power in passband and stopband before and after filtering

% Passband power (should be similar before and after)
passband_indices_signal = find(f >= fmin & f <= fmax);
passband_power_before = mean(X_hamming_single(passband_indices_signal).^2);
passband_power_after = mean(X_filtered_single(passband_indices_signal).^2);

% Stopband power (should be much lower after filtering)
stopband_indices_lower = find(f <= fstop_lower);
stopband_indices_upper = find(f >= fstop_upper & f <= fs/2);
stopband_indices_signal = [stopband_indices_lower, stopband_indices_upper];

stopband_power_before = mean(X_hamming_single(stopband_indices_signal).^2);
stopband_power_after = mean(X_filtered_single(stopband_indices_signal).^2);

% Calculate noise reduction in dB
noise_reduction_dB = 10 * log10(stopband_power_before / stopband_power_after);

% Calculate signal-to-noise improvement
snr_before = 10 * log10(passband_power_before / stopband_power_before);
snr_after = 10 * log10(passband_power_after / stopband_power_after);
snr_improvement = snr_after - snr_before;

fprintf('--------------------------------------------\n');
fprintf('FILTERING PERFORMANCE\n');
fprintf('PASSBAND POWER:\n');
fprintf('  Before filtering:        %.6e\n', passband_power_before);
fprintf('  After filtering:         %.6e\n', passband_power_after);
fprintf('  Change:                  %.2f dB\n', 10*log10(passband_power_after/passband_power_before));
fprintf('--------------------------------------------\n');
fprintf('STOPBAND POWER:\n');
fprintf('  Before filtering:        %.6e\n', stopband_power_before);
fprintf('  After filtering:         %.6e\n', stopband_power_after);
fprintf('  Noise reduction:         %.2f dB\n', noise_reduction_dB);
fprintf('--------------------------------------------\n');
fprintf('SIGNAL-TO-NOISE RATIO:\n');
fprintf('  SNR before filtering:    %.2f dB\n', snr_before);
fprintf('  SNR after filtering:     %.2f dB\n', snr_after);
fprintf('  SNR improvement:         %.2f dB\n', snr_improvement);
fprintf('--------------------------------------------\n');

%% Amplitude statistics comparison

fprintf('AMPLITUDE STATISTICS\n');
fprintf('                           Before      After\n');
fprintf('  Maximum amplitude:       %.4f      %.4f\n', max(abs(x)), max(abs(x_filtered)));
fprintf('  RMS amplitude:           %.4f      %.4f\n', sqrt(mean(x.^2)), sqrt(mean(x_filtered.^2)));
fprintf('  Standard deviation:      %.4f      %.4f\n', std(x), std(x_filtered));
fprintf('--------------------------------------------\n');

%% Task 3: Carrier Recovery and Mixing
%% Sub-task 3.1: Carrier Recovery Using Square Law

fprintf('       TASK 3: CARRIER RECOVERY            \n');

% Apply square law to the bandpass filtered signal
x_squared = x_filtered .^ 2;

fprintf('Square law applied to filtered signal.\n');
fprintf('Squared signal length:     %d samples\n', length(x_squared));

%% Compute spectrum of squared signal

% Apply Hamming window
x_squared_windowed = x_squared .* hamming_window;

% Compute FFT
X_squared = fft(x_squared_windowed);
X_squared_magnitude = abs(X_squared);
X_squared_normalised = X_squared_magnitude / N / CG_hamming;

% Single-sided spectrum
X_squared_single = X_squared_normalised(1:num_bins_single_sided);
X_squared_single(2:end-1) = 2 * X_squared_single(2:end-1);
X_squared_dB = 20 * log10(X_squared_single + eps);

%% Plot squared signal spectrum

figure('Name', 'Carrier Recovery - Squared Signal Spectrum', 'Position', [100, 100, 1200, 600]);

% Full spectrum
subplot(2,1,1);
plot(f/1000, X_squared_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Spectrum of Squared Signal (Full Range)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_squared_dB) + 10]);
hold on;

% Mark expected 2fc region (dotted)
xline(2*fc_rounded/1000, 'r--', 'LineWidth', 2);

%   CARRIER PEAK DETECTION

% Define search range around 2fc
search_range_low = 2*fc_rounded - 5000;   % Hz
search_range_high = 2*fc_rounded + 5000;  % Hz

% Find indices of search region
search_indices = find(f >= search_range_low & f <= search_range_high);

% Extract the region to search for peaks
X_search = X_squared_single(search_indices);
f_search = f(search_indices);

% Use findpeaks
[peaks, locs] = findpeaks(X_search, f_search);

% Get the highest peak
[peak_value, max_idx] = max(peaks);
f_2fc_measured = locs(max_idx);

% Calculate measured carrier frequency
fc_measured = f_2fc_measured / 2;
fc_final = round(fc_measured / 1000) * 1000;

% Mark the detected peak on the full spectrum
plot(f_2fc_measured/1000, 20*log10(peak_value), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
legend('Spectrum', 'Expected 2f_c', 'Detected 2f_c Peak', 'Location', 'northeast');
hold off;

%% Zoomed view around 2fc
subplot(2,1,2);
plot(f/1000, X_squared_dB, 'b', 'LineWidth', 0.5);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Spectrum of Squared Signal (Zoomed Around 2f_c)', 'FontSize', 12);
grid on;
xlim([(2*fc_rounded - 15000)/1000, (2*fc_rounded + 15000)/1000]);
ylim([-80, max(X_squared_dB) + 10]);
hold on;

% Mark expected 2fc
xline(2*fc_rounded/1000, 'r--', 'LineWidth', 2);

% Mark detected 2fc
plot(f_2fc_measured/1000, 20*log10(peak_value), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
legend('Spectrum', 'Expected 2f_c', 'Detected 2f_c Peak', 'Location', 'northeast');
hold off;

%% Display carrier recovery results

fprintf('CARRIER FREQUENCY DETECTION\n');
fprintf('Search range:              %d - %d Hz\n', search_range_low, search_range_high);
fprintf('Peak found at 2fc:         %.2f Hz\n', f_2fc_measured);
fprintf('Calculated fc:             %.2f Hz\n', fc_measured);
fprintf('Rounded fc (to 1 kHz):     %d Hz (%.1f kHz)\n', fc_final, fc_final/1000);
fprintf('--------------------------------------------\n');
fprintf('Initial estimate (Task 1): %d Hz\n', fc_rounded);
fprintf('Difference:                %.2f Hz\n', abs(fc_final - fc_rounded));

if fc_final == fc_rounded
    fprintf('\nCarrier frequency CONFIRMED: %d Hz\n', fc_final);
else
    fprintf('\nWARNING: Carrier frequency differs from Task 1 estimate.\n');
    fprintf('Using newly measured value: %d Hz\n', fc_final);
    fc_rounded = fc_final;
    fmin = fc_final - 4000;
    fmax = fc_final + 4000;
end

%% Sub-task 3.2: Carrier Generation and Mixing

fprintf('\n============================================\n');
fprintf('       CARRIER GENERATION AND MIXING       \n');
fprintf('============================================\n');

% Set initial phase to zero (will be optimised in Task 5)
phi = 0;

% Generate local carrier signal
% carrier(t) = cos(2*pi*fc*t + phi)
carrier = cos(2 * pi * fc_final * t + phi);

fprintf('Carrier signal generated.\n');
fprintf('Carrier frequency:         %d Hz\n', fc_final);
fprintf('Initial phase:             %.4f radians (%.2f degrees)\n', phi, phi*180/pi);
fprintf('Carrier signal length:     %d samples\n', length(carrier));

%% Multiply filtered AM signal with carrier (mixing)

x_mixed = x_filtered .* carrier;

fprintf('Mixing complete.\n');
fprintf('Mixed signal length:       %d samples\n', length(x_mixed));

%% Time domain plots

figure('Name', 'Mixing Process - Time Domain', 'Position', [100, 100, 1200, 800]);

% Filtered AM signal
subplot(3,1,1);
plot(t, x_filtered, 'b', 'LineWidth', 0.3);
xlabel('Time (seconds)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Bandpass Filtered AM Signal (Input to Mixer)', 'FontSize', 12);
grid on;
xlim([0, duration]);

% Local carrier signal (show only a short segment to see oscillations)
subplot(3,1,2);
plot(t, carrier, 'g', 'LineWidth', 0.3);
xlabel('Time (seconds)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title(sprintf('Local Carrier Signal: cos(2\\pi \\cdot %d \\cdot t + %.2f)', fc_final, phi), 'FontSize', 12);
grid on;
xlim([0, duration]);

% Mixed signal
subplot(3,1,3);
plot(t, x_mixed, 'r', 'LineWidth', 0.3);
xlabel('Time (seconds)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Mixed Signal (Filtered AM × Carrier)', 'FontSize', 12);
grid on;
xlim([0, duration]);

%% Frequency domain analysis of mixed signal

% Apply Hamming window
x_mixed_windowed = x_mixed .* hamming_window;

% Compute FFT
X_mixed = fft(x_mixed_windowed);
X_mixed_magnitude = abs(X_mixed);
X_mixed_normalised = X_mixed_magnitude / N / CG_hamming;

% Single-sided spectrum
X_mixed_single = X_mixed_normalised(1:num_bins_single_sided);
X_mixed_single(2:end-1) = 2 * X_mixed_single(2:end-1);
X_mixed_dB = 20 * log10(X_mixed_single + eps);

%% Frequency domain plots

figure('Name', 'Mixing Process - Frequency Domain', 'Position', [100, 100, 1200, 700]);

% Spectrum before mixing (filtered AM signal)
subplot(2,1,1);
plot(f/1000, X_filtered_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Spectrum Before Mixing (Bandpass Filtered AM Signal)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_filtered_dB) + 10]);

hold on;
xline(fc_final/1000, 'r--', 'LineWidth', 1.5);
legend('Spectrum', 'f_c', 'Location', 'northeast');
hold off;

% Spectrum after mixing
subplot(2,1,2);
plot(f/1000, X_mixed_dB, 'r', 'LineWidth', 0.3);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Spectrum After Mixing (Shows Baseband and 2f_c Components)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_mixed_dB) + 10]);

hold on;
xline(0, 'g--', 'LineWidth', 1.5);
xline(2*fc_final/1000, 'm--', 'LineWidth', 1.5);
legend('Spectrum', 'Baseband (0 Hz)', '2f_c', 'Location', 'northeast');
hold off;

%% Analyse the frequency components

% Find power in baseband region (0 to 4 kHz - message bandwidth)
baseband_indices = find(f >= 0 & f <= 4000);
baseband_power = mean(X_mixed_single(baseband_indices).^2);

% Find power in 2fc region (2fc ± 4 kHz)
double_fc_indices = find(f >= (2*fc_final - 4000) & f <= (2*fc_final + 4000));
double_fc_power = mean(X_mixed_single(double_fc_indices).^2);

% Find power in noise region (between baseband and 2fc)
noise_region_low = 8000;  % Above baseband
noise_region_high = 2*fc_final - 8000;  % Below 2fc component
if noise_region_high > noise_region_low
    noise_indices = find(f >= noise_region_low & f <= noise_region_high);
    noise_power = mean(X_mixed_single(noise_indices).^2);
else
    noise_power = 0;
end

fprintf('       MIXED SIGNAL ANALYSIS               \n');
fprintf('FREQUENCY COMPONENTS:\n');
fprintf('  Baseband (0-4 kHz):      %.6e (signal)\n', baseband_power);
fprintf('  2fc region:              %.6e (to be filtered)\n', double_fc_power);
if noise_power > 0
    fprintf('  Noise region:            %.6e\n', noise_power);
end
fprintf('--------------------------------------------\n');
fprintf('The baseband component contains the message signal.\n');
fprintf('The 2fc component will be removed by lowpass filter.\n');

%% Amplitude statistics

fprintf('       AMPLITUDE COMPARISON                \n');
fprintf('                           Filtered    Mixed\n');
fprintf('  Maximum amplitude:       %.4f      %.4f\n', max(abs(x_filtered)), max(abs(x_mixed)));
fprintf('  RMS amplitude:           %.4f      %.4f\n', sqrt(mean(x_filtered.^2)), sqrt(mean(x_mixed.^2)));
fprintf('  Standard deviation:      %.4f      %.4f\n', std(x_filtered), std(x_mixed));