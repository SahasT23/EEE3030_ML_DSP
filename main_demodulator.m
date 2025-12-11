%% Sub-task 1.1: Reading the Audio File and Basic Signal Information
% EEE3030 DSP Assignment - Task 1

clear all;
close all;
clc;

%% Read the audio file
filename = 'Sahas_Talasila.wav';
[x, fs] = audioread(filename);

% If stereo, convert to mono by taking first channel
if size(x, 2) > 1
    x = x(:, 1);
    fprintf('Note: Stereo file detected, using first channel only.\n\n');
end

% audioread() returns a column vector, but we need row vectors throughout
if iscolumn(x)
    x = x';  % Transpose to row vector
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

% Plot spectrogram
figure;
spectrogram(x, 1024, 512, 1024, fs, 'yaxis');
title('Spectrogram of Sahas\_Talasila.wav');
xlabel('Time (s)');
ylabel('Frequency (kHz)');

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

% Create a sample index vector for window visualisation
n_window = (0:N-1)';

% Generate windows - BUT TRANSPOSE THEM TO ROW VECTORS
rectangular_window = ones(1, N);  % Changed to row vector

% Generate Hamming window - as row vector
hamming_window = (0.54 - 0.46 * cos(2 * pi * (0:N-1) / (N - 1)));

% Generate Hanning window - as row vector  
hanning_window = (0.5 - 0.5 * cos(2 * pi * (0:N-1) / (N - 1)));

% Generate Blackman window - as row vector
blackman_window = (0.42 - 0.5 * cos(2 * pi * (0:N-1) / (N - 1)) + 0.08 * cos(4 * pi * (0:N-1) / (N - 1)));


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
passband_indices_signal = [stopband_indices_lower, stopband_indices_upper];

stopband_power_before = mean(X_hamming_single(passband_indices_signal).^2);
stopband_power_after = mean(X_filtered_single(passband_indices_signal).^2);

% Calculate noise reduction in dB
noise_reduction_dB = 10 * log10(stopband_power_before / stopband_power_after);

% Calculate signal-to-noise improvement
snr_before = 10 * log10(passband_power_before / stopband_power_before);
snr_after = 10 * log10(passband_power_after / stopband_power_after);
snr_improvement = snr_after - snr_before;

fprintf('--------------------------------------------\n');
fprintf('FILTERING PERFORMANCE\n');
fprintf('STOPBAND POWER:\n');
fprintf('  Before filtering:        %.6e\n', passband_power_before);
fprintf('  After filtering:         %.6e\n', passband_power_after);
fprintf('  Change:                  %.2f dB\n', 10*log10(passband_power_after/passband_power_before));
fprintf('--------------------------------------------\n');
fprintf('PASSBAND POWER:\n');
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

%% Task 4: IIR Lowpass Filter Design
%% Sub-task 4.1: IIR Lowpass Filter Design

fprintf('TASK 4: IIR LOWPASS FILTER\n');

% Filter specifications
filter_order = 4;
fc_lowpass = 4000;  % Cutoff frequency in Hz

% Normalised cutoff frequency for MATLAB's butter()
% MATLAB uses normalised frequency where 1 = Nyquist frequency (fs/2)
Wn = fc_lowpass / (fs/2);

fprintf('FILTER SPECIFICATIONS:\n');
fprintf('  Filter type:             Butterworth (maximally flat)\n');
fprintf('  Filter order:            %d\n', filter_order);
fprintf('  Cutoff frequency:        %d Hz\n', fc_lowpass);
fprintf('  Sampling frequency:      %d Hz\n', fs);
fprintf('  Normalised cutoff (Wn):  %.6f\n', Wn);

%% Design the Butterworth filter using bilinear transform

% MATLAB's butter() function implements the bilinear transform method
% It returns coefficients for the transfer function H(z) = B(z)/A(z)
[b_iir, a_iir] = butter(filter_order, Wn, 'low');

fprintf('       IIR FILTER COEFFICIENTS             \n');
fprintf('Numerator coefficients (b):\n');
for i = 1:length(b_iir)
    fprintf('  b[%d] = %.10f\n', i-1, b_iir(i));
end
fprintf('--------------------------------------------\n');
fprintf('Denominator coefficients (a):\n');
for i = 1:length(a_iir)
    fprintf('  a[%d] = %.10f\n', i-1, a_iir(i));
end

%% Verify coefficient properties

fprintf('       COEFFICIENT ANALYSIS                \n');
fprintf('Number of numerator coefficients:   %d\n', length(b_iir));
fprintf('Number of denominator coefficients: %d\n', length(a_iir));
fprintf('Sum of numerator coefficients:      %.10f\n', sum(b_iir));
fprintf('Sum of denominator coefficients:    %.10f\n', sum(a_iir));
fprintf('DC gain (H(z=1)):                   %.6f\n', sum(b_iir)/sum(a_iir));


%% Plot filter coefficients

figure('Name', 'IIR Filter Coefficients', 'Position', [100, 100, 1000, 500]);

subplot(1,2,1);
stem(0:length(b_iir)-1, b_iir, 'b', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Coefficient Index', 'FontSize', 11);
ylabel('Value', 'FontSize', 11);
title('Numerator Coefficients (b)', 'FontSize', 12);
grid on;

subplot(1,2,2);
stem(0:length(a_iir)-1, a_iir, 'r', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Coefficient Index', 'FontSize', 11);
ylabel('Value', 'FontSize', 11);
title('Denominator Coefficients (a)', 'FontSize', 12);
grid on;

%% Sub-task 4.2: IIR Frequency Response Verification
fprintf('IIR FREQUENCY RESPONSE VERIFICATION \n');

% Compute frequency response using freqz
N_freq = 8192;  % Number of frequency points
[H_iir, f_iir] = freqz(b_iir, a_iir, N_freq, fs);

% Magnitude response in dB
H_iir_magnitude = abs(H_iir);
H_iir_dB = 20 * log10(H_iir_magnitude + eps);

% Phase response
H_iir_phase = angle(H_iir);
H_iir_phase_unwrapped = unwrap(H_iir_phase);

figure('Name', 'IIR Filter Frequency Response', 'Position', [100, 100, 1200, 700]);

% Full spectrum magnitude response
subplot(2,2,1);
plot(f_iir/1000, H_iir_dB, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('IIR Lowpass Filter - Magnitude Response (Full Range)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-100, 5]);

hold on;
xline(fc_lowpass/1000, 'r--', 'LineWidth', 1.5);
yline(-3, 'g--', 'LineWidth', 1.5);
yline(0, 'k-', 'LineWidth', 0.5);
legend('Response', 'f_c = 4 kHz', '-3 dB', '0 dB', 'Location', 'southwest');
hold off;

% Passband detail
subplot(2,2,2);
plot(f_iir/1000, H_iir_dB, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Passband Detail', 'FontSize', 12);
grid on;
xlim([0, 8]);
ylim([-10, 2]);

hold on;
xline(fc_lowpass/1000, 'r--', 'LineWidth', 1.5);
yline(-3, 'g--', 'LineWidth', 1.5);
yline(0, 'k-', 'LineWidth', 0.5);
legend('Response', 'f_c = 4 kHz', '-3 dB', '0 dB', 'Location', 'southwest');
hold off;

% Phase response
subplot(2,2,3);
plot(f_iir/1000, H_iir_phase_unwrapped * 180/pi, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Phase (degrees)', 'FontSize', 11);
title('Phase Response', 'FontSize', 12);
grid on;
xlim([0, 10]);

hold on;
xline(fc_lowpass/1000, 'r--', 'LineWidth', 1.5);
legend('Phase', 'f_c = 4 kHz', 'Location', 'southwest');
hold off;

% Linear magnitude (to show Butterworth shape)
subplot(2,2,4);
plot(f_iir/1000, H_iir_magnitude, 'b', 'LineWidth', 1.5);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (linear)', 'FontSize', 11);
title('Linear Magnitude Response', 'FontSize', 12);
grid on;
xlim([0, 20]);
ylim([0, 1.1]);

hold on;
xline(fc_lowpass/1000, 'r--', 'LineWidth', 1.5);
yline(0.707, 'g--', 'LineWidth', 1.5);
yline(1.0, 'k-', 'LineWidth', 0.5);
legend('Response', 'f_c = 4 kHz', '0.707 (-3 dB)', 'Unity', 'Location', 'northeast');
hold off;

% Find -3 dB point
idx_3dB = find(H_iir_dB <= -3, 1, 'first');
if ~isempty(idx_3dB)
    f_3dB_actual = f_iir(idx_3dB);
else
    f_3dB_actual = NaN;
end

% Measure gain at specific frequencies
f_test_points = [100, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000];
gain_at_test_points = zeros(size(f_test_points));

for i = 1:length(f_test_points)
    [~, idx] = min(abs(f_iir - f_test_points(i)));
    gain_at_test_points(i) = H_iir_dB(idx);
end

% Measure rolloff rate (attenuation per octave after cutoff)
% Compare attenuation at 8 kHz (1 octave above 4 kHz) and 16 kHz (2 octaves)
[~, idx_8k] = min(abs(f_iir - 8000));
[~, idx_16k] = min(abs(f_iir - 16000));
atten_8k = H_iir_dB(idx_8k);
atten_16k = H_iir_dB(idx_16k);
rolloff_per_octave = atten_8k - atten_16k;

fprintf('CUTOFF FREQUENCY VERIFICATION:\n');
fprintf('  Specified cutoff:        %d Hz\n', fc_lowpass);
fprintf('  Measured -3 dB point:    %.2f Hz\n', f_3dB_actual);
fprintf('  Error:                   %.2f Hz\n', abs(f_3dB_actual - fc_lowpass));

fprintf('\nGAIN AT TEST FREQUENCIES:\n');
fprintf('  Frequency (Hz)    Gain (dB)\n');
fprintf('  ------------      ---------\n');
for i = 1:length(f_test_points)
    fprintf('  %6d            %7.2f\n', f_test_points(i), gain_at_test_points(i));
end

fprintf('\nROLLOFF CHARACTERISTICS:\n');
fprintf('  Attenuation at 8 kHz:    %.2f dB\n', atten_8k);
fprintf('  Attenuation at 16 kHz:   %.2f dB\n', atten_16k);
fprintf('  Rolloff (8k to 16k):     %.2f dB/octave\n', rolloff_per_octave);
fprintf('  Expected (4th order):    ~24 dB/octave\n');
fprintf('\nBUTTERWORTH VERIFICATION:\n');

% Check passband flatness (should be monotonic, no ripple)
passband_idx = find(f_iir <= fc_lowpass);
passband_gain = H_iir_dB(passband_idx);
passband_ripple = max(passband_gain) - min(passband_gain);

fprintf('  Passband ripple:         %.4f dB\n', passband_ripple);

if passband_ripple < 0.1
    fprintf('  Passband:                MAXIMALLY FLAT (as expected)\n');
else
    fprintf('  Passband:                UNEXPECTED RIPPLE\n');
end

% Check gain at cutoff
[~, idx_fc] = min(abs(f_iir - fc_lowpass));
gain_at_fc = H_iir_dB(idx_fc);
fprintf('  Gain at cutoff (4 kHz):  %.2f dB (should be ~-3 dB)\n', gain_at_fc);

if abs(gain_at_fc - (-3)) < 0.5
    fprintf('  Cutoff definition:       CORRECT\n');
else
    fprintf('  Cutoff definition:       CHECK DESIGN\n');
end

figure('Name', 'IIR Filter Pole-Zero Plot', 'Position', [100, 100, 600, 600]);

% Get poles and zeros
zeros_iir = roots(b_iir);
poles_iir = roots(a_iir);

% Plot unit circle
theta_circle = linspace(0, 2*pi, 100);
plot(cos(theta_circle), sin(theta_circle), 'k--', 'LineWidth', 1);
hold on;

% Plot zeros
plot(real(zeros_iir), imag(zeros_iir), 'bo', 'MarkerSize', 10, 'LineWidth', 2);

% Plot poles
plot(real(poles_iir), imag(poles_iir), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

hold off;
xlabel('Real Part', 'FontSize', 11);
ylabel('Imaginary Part', 'FontSize', 11);
title('Pole-Zero Plot (Poles Must Be Inside Unit Circle for Stability)', 'FontSize', 12);
legend('Unit Circle', 'Zeros', 'Poles', 'Location', 'best');
grid on;
axis equal;
xlim([-1.5, 1.5]);
ylim([-1.5, 1.5]);

% Check stability
pole_magnitudes = abs(poles_iir);
max_pole_magnitude = max(pole_magnitudes);

fprintf('\nSTABILITY ANALYSIS:\n');
fprintf('  Pole magnitudes:\n');
for i = 1:length(poles_iir)
    fprintf('    Pole %d: %.6f + %.6fj, |pole| = %.6f\n', i, real(poles_iir(i)), imag(poles_iir(i)), abs(poles_iir(i)));
end
fprintf('  Maximum pole magnitude:  %.6f\n', max_pole_magnitude);

if max_pole_magnitude < 1
    fprintf('  Filter stability:        STABLE (all poles inside unit circle)\n');
else
    fprintf('  Filter stability:        UNSTABLE - REDESIGN REQUIRED\n');
end
%% Sub-task 4.3: Custom IIR Filter Verification

fprintf('       CUSTOM IIR FILTER VERIFICATION      \n');

%% Test 1: Compare with MATLAB's filter() function

fprintf('\nTest 1: Comparing custom_iir_filter() with MATLAB filter()\n');
fprintf('------------------------------------------------------------\n');

% Create test signal (use a portion of the mixed signal)
test_length = 10000;
test_signal = x_mixed(1:test_length);

% Filter using MATLAB's built-in function
tic;
y_matlab = filter(b_iir, a_iir, test_signal);
time_matlab = toc;

% Filter using our custom function
tic;
y_custom = custom_iir_filter(b_iir, a_iir, test_signal);
time_custom = toc;

% Calculate error
error_iir = y_matlab - y_custom;
max_error_iir = max(abs(error_iir));
mean_error_iir = mean(abs(error_iir));
rms_error_iir = sqrt(mean(error_iir.^2));

fprintf('  Test signal length:      %d samples\n', test_length);
fprintf('  Maximum absolute error:  %.2e\n', max_error_iir);
fprintf('  Mean absolute error:     %.2e\n', mean_error_iir);
fprintf('  RMS error:               %.2e\n', rms_error_iir);
fprintf('  MATLAB filter() time:    %.4f seconds\n', time_matlab);
fprintf('  Custom function time:    %.4f seconds\n', time_custom);
fprintf('  Speed ratio:             %.1fx slower\n', time_custom/time_matlab);

if max_error_iir < 1e-10
    fprintf('  Result:                  PASS (negligible numerical error)\n');
else
    fprintf('  Result:                  CHECK - Error may be significant\n');
end

%% Test 2: Verify with known input (impulse response)

fprintf('\nTest 2: Impulse Response Verification\n');
fprintf('--------------------------------------\n');

% Create impulse signal
impulse_length = 500;
impulse = [1, zeros(1, impulse_length - 1)];

% Get impulse response using both methods
h_matlab = filter(b_iir, a_iir, impulse);
h_custom = custom_iir_filter(b_iir, a_iir, impulse);

% Compare
impulse_error = max(abs(h_matlab - h_custom));
fprintf('  Impulse signal length:   %d samples\n', impulse_length);
fprintf('  Maximum impulse error:   %.2e\n', impulse_error);

if impulse_error < 1e-10
    fprintf('  Result:                  PASS\n');
else
    fprintf('  Result:                  CHECK\n');
end

%% Test 3: Verify with step response

fprintf('\nTest 3: Step Response Verification\n');
fprintf('-----------------------------------\n');

% Create step signal
step_length = 500;
step_signal = ones(1, step_length);

% Get step response using both methods
step_matlab = filter(b_iir, a_iir, step_signal);
step_custom = custom_iir_filter(b_iir, a_iir, step_signal);

% Compare
step_error = max(abs(step_matlab - step_custom));
fprintf('  Step signal length:      %d samples\n', step_length);
fprintf('  Maximum step error:      %.2e\n', step_error);
fprintf('  Final value (MATLAB):    %.6f\n', step_matlab(end));
fprintf('  Final value (custom):    %.6f\n', step_custom(end));
fprintf('  Expected DC gain:        %.6f\n', sum(b_iir)/sum(a_iir));

if step_error < 1e-10
    fprintf('  Result:                  PASS\n');
else
    fprintf('  Result:                  CHECK\n');
end

%% Plot comparison

figure('Name', 'Custom IIR Filter Verification', 'Position', [100, 100, 1200, 800]);

% Plot 1: Output comparison (first 500 samples)
subplot(2,2,1);
plot_samples = min(500, test_length);
plot(1:plot_samples, y_matlab(1:plot_samples), 'b-', 'LineWidth', 1.5);
hold on;
plot(1:plot_samples, y_custom(1:plot_samples), 'r--', 'LineWidth', 1);
hold off;
xlabel('Sample', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Output Comparison (First 500 Samples)', 'FontSize', 12);
legend('MATLAB filter()', 'custom\_iir\_filter()', 'Location', 'best');
grid on;

% Plot 2: Error signal
subplot(2,2,2);
plot(1:test_length, error_iir, 'r-', 'LineWidth', 1);
xlabel('Sample', 'FontSize', 11);
ylabel('Error', 'FontSize', 11);
title(sprintf('Error Signal (Max = %.2e)', max_error_iir), 'FontSize', 12);
grid on;

% Plot 3: Impulse response comparison
subplot(2,2,3);
plot(1:impulse_length, h_matlab, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:impulse_length, h_custom, 'r--', 'LineWidth', 1);
hold off;
xlabel('Sample', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Impulse Response Comparison', 'FontSize', 12);
legend('MATLAB', 'Custom', 'Location', 'best');
grid on;

% Plot 4: Step response comparison
subplot(2,2,4);
plot(1:step_length, step_matlab, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:step_length, step_custom, 'r--', 'LineWidth', 1);
yline(sum(b_iir)/sum(a_iir), 'g--', 'LineWidth', 1.5);
hold off;
xlabel('Sample', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Step Response Comparison', 'FontSize', 12);
legend('MATLAB', 'Custom', 'DC Gain', 'Location', 'best');
grid on;

fprintf('\nCustom IIR filter implementation verified.\n');

%% Sub-task 4.4: Applying IIR Lowpass Filter to Mixed Signal
fprintf('       APPLYING IIR LOWPASS FILTER         \n');

%% Apply lowpass filter using custom implementation

fprintf('\nFiltering mixed signal with custom IIR filter...\n');

tic;
x_demodulated = custom_iir_filter(b_iir, a_iir, x_mixed);
time_iir_filter = toc;

fprintf('  Input signal length:     %d samples (%.2f seconds)\n', length(x_mixed), length(x_mixed)/fs);
fprintf('  Output signal length:    %d samples (%.2f seconds)\n', length(x_demodulated), length(x_demodulated)/fs);
fprintf('  Filtering time:          %.2f seconds\n', time_iir_filter);

%% Time domain analysis

fprintf('\nTIME DOMAIN ANALYSIS:\n');
fprintf('---------------------\n');

% Calculate statistics
mixed_max = max(abs(x_mixed));
mixed_rms = sqrt(mean(x_mixed.^2));
demod_max = max(abs(x_demodulated));
demod_rms = sqrt(mean(x_demodulated.^2));

fprintf('  Mixed signal:\n');
fprintf('    Maximum amplitude:     %.6f\n', mixed_max);
fprintf('    RMS amplitude:         %.6f\n', mixed_rms);
fprintf('  Demodulated signal:\n');
fprintf('    Maximum amplitude:     %.6f\n', demod_max);
fprintf('    RMS amplitude:         %.6f\n', demod_rms);
fprintf('  Amplitude ratio:         %.4f\n', demod_rms/mixed_rms);

%% Plot time domain comparison

figure('Name', 'IIR Lowpass Filter - Time Domain', 'Position', [100, 100, 1200, 800]);

% Plot 1: Mixed signal (input to lowpass filter)
subplot(3,1,1);
plot(t, x_mixed, 'b', 'LineWidth', 0.5);
xlabel('Time (seconds)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Mixed Signal (Input to Lowpass Filter)', 'FontSize', 12);
grid on;
xlim([0, t(end)]);

% Plot 2: Demodulated signal (output of lowpass filter)
subplot(3,1,2);
plot(t, x_demodulated, 'r', 'LineWidth', 0.5);
xlabel('Time (seconds)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Demodulated Signal (Output of Lowpass Filter)', 'FontSize', 12);
grid on;
xlim([0, t(end)]);

% Plot 3: Overlay comparison (portion of signal)
subplot(3,1,3);
% Show a portion where we can see detail
plot_duration = 0.05;  % 50 ms
plot_samples = round(plot_duration * fs);
plot_start = round(length(t) / 2);  % Middle of signal
plot_end = min(plot_start + plot_samples, length(t));
t_portion = t(plot_start:plot_end);

plot(t_portion, x_mixed(plot_start:plot_end), 'b', 'LineWidth', 0.5);
hold on;
plot(t_portion, x_demodulated(plot_start:plot_end), 'r', 'LineWidth', 1.5);
hold off;
xlabel('Time (seconds)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Comparison (50 ms Portion from Middle of Signal)', 'FontSize', 12);
legend('Mixed', 'Demodulated', 'Location', 'best');
grid on;

%% Frequency domain analysis

fprintf('\nFREQUENCY DOMAIN ANALYSIS:\n');
fprintf('--------------------------\n');

% Compute spectrum of demodulated signal
window_demod = hamming(length(x_demodulated))';
x_demod_windowed = x_demodulated .* window_demod;
X_demod = fft(x_demod_windowed);
X_demod_magnitude = abs(X_demod) / length(X_demod);

% Single-sided spectrum
N_demod = length(X_demod);
X_demod_single = X_demod_magnitude(1:floor(N_demod/2)+1);
X_demod_single(2:end-1) = 2 * X_demod_single(2:end-1);
f_demod = (0:floor(N_demod/2)) * fs / N_demod;

% Convert to dB
X_demod_dB = 20 * log10(X_demod_single + eps);

% Also compute spectrum of mixed signal for comparison
window_mixed = hamming(length(x_mixed))';
x_mixed_windowed = x_mixed .* window_mixed;
X_mixed = fft(x_mixed_windowed);
X_mixed_magnitude = abs(X_mixed) / length(X_mixed);

N_mixed = length(X_mixed);
X_mixed_single = X_mixed_magnitude(1:floor(N_mixed/2)+1);
X_mixed_single(2:end-1) = 2 * X_mixed_single(2:end-1);
f_mixed = (0:floor(N_mixed/2)) * fs / N_mixed;
X_mixed_dB = 20 * log10(X_mixed_single + eps);

%% Plot frequency domain comparison

figure('Name', 'IIR Lowpass Filter - Frequency Domain', 'Position', [100, 100, 1200, 800]);

% Plot 1: Mixed signal spectrum (full range)
subplot(2,2,1);
plot(f_mixed/1000, X_mixed_dB, 'b', 'LineWidth', 1);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Mixed Signal Spectrum (Before Lowpass)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_mixed_dB)+10]);

hold on;
xline(fc_lowpass/1000, 'r--', 'LineWidth', 1.5);
xline(2*fc_final/1000, 'g--', 'LineWidth', 1.5);
legend('Spectrum', 'Lowpass cutoff (4 kHz)', sprintf('2f_c (%.0f kHz)', 2*fc_final/1000), 'Location', 'northeast');
hold off;

% Plot 2: Demodulated signal spectrum (full range)
subplot(2,2,2);
plot(f_demod/1000, X_demod_dB, 'r', 'LineWidth', 1);
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Demodulated Signal Spectrum (After Lowpass)', 'FontSize', 12);
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_demod_dB)+10]);

hold on;
xline(fc_lowpass/1000, 'r--', 'LineWidth', 1.5);
legend('Spectrum', 'Lowpass cutoff (4 kHz)', 'Location', 'northeast');
hold off;

% Plot 3: Baseband detail (0-10 kHz)
subplot(2,2,3);
plot(f_mixed/1000, X_mixed_dB, 'b', 'LineWidth', 1);
hold on;
plot(f_demod/1000, X_demod_dB, 'r', 'LineWidth', 1.5);
xline(fc_lowpass/1000, 'k--', 'LineWidth', 1.5);
hold off;
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Baseband Comparison (0-10 kHz)', 'FontSize', 12);
legend('Before Lowpass', 'After Lowpass', 'Cutoff (4 kHz)', 'Location', 'northeast');
grid on;
xlim([0, 10]);
ylim([-120, max(X_mixed_dB)+10]);

% Plot 4: Comparison around 2fc region
subplot(2,2,4);
plot(f_mixed/1000, X_mixed_dB, 'b', 'LineWidth', 1);
hold on;
plot(f_demod/1000, X_demod_dB, 'r', 'LineWidth', 1.5);
xline(2*fc_final/1000, 'g--', 'LineWidth', 1.5);
hold off;
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title(sprintf('2f_c Region (%.0f kHz)', 2*fc_final/1000), 'FontSize', 12);
legend('Before Lowpass', 'After Lowpass', '2f_c', 'Location', 'northeast');
grid on;
xlim([2*fc_final/1000 - 10, min(2*fc_final/1000 + 10, fs/2000)]);
ylim([-120, max(X_mixed_dB)+10]);

%% Quantitative analysis of filtering effect

fprintf('\nFILTERING EFFECTIVENESS:\n');
fprintf('------------------------\n');

% Calculate power in different frequency bands
% Baseband (0 to 4 kHz) - message region
baseband_idx_mixed = find(f_mixed <= fc_lowpass);
baseband_idx_demod = find(f_demod <= fc_lowpass);

baseband_power_before = sum(X_mixed_single(baseband_idx_mixed).^2);
baseband_power_after = sum(X_demod_single(baseband_idx_demod).^2);

% Stopband (above 4 kHz) - should be attenuated
stopband_idx_mixed = find(f_mixed > fc_lowpass);
stopband_idx_demod = find(f_demod > fc_lowpass);

stopband_power_before = sum(X_mixed_single(stopband_idx_mixed).^2);
stopband_power_after = sum(X_demod_single(stopband_idx_demod).^2);

% 2fc region (2fc ± 4 kHz)
fc2_low = 2*fc_final - 4000;
fc2_high = 2*fc_final + 4000;
fc2_idx_mixed = find(f_mixed >= fc2_low & f_mixed <= fc2_high);
fc2_idx_demod = find(f_demod >= fc2_low & f_demod <= fc2_high);

fc2_power_before = sum(X_mixed_single(fc2_idx_mixed).^2);
fc2_power_after = sum(X_demod_single(fc2_idx_demod).^2);

% Calculate attenuation
baseband_change_dB = 10 * log10(baseband_power_after / baseband_power_before);
stopband_attenuation_dB = 10 * log10(stopband_power_before / stopband_power_after);
fc2_attenuation_dB = 10 * log10(fc2_power_before / fc2_power_after);

fprintf('  Baseband (0-4 kHz):\n');
fprintf('    Power before:          %.6e\n', baseband_power_before);
fprintf('    Power after:           %.6e\n', baseband_power_after);
fprintf('    Change:                %.2f dB\n', baseband_change_dB);
fprintf('\n');
fprintf('  Stopband (>4 kHz):\n');
fprintf('    Power before:          %.6e\n', stopband_power_before);
fprintf('    Power after:           %.6e\n', stopband_power_after);
fprintf('    Attenuation:           %.2f dB\n', stopband_attenuation_dB);
fprintf('\n');
fprintf('  2f_c region (%.0f-%.0f kHz):\n', fc2_low/1000, fc2_high/1000);
fprintf('    Power before:          %.6e\n', fc2_power_before);
fprintf('    Power after:           %.6e\n', fc2_power_after);
fprintf('    Attenuation:           %.2f dB\n', fc2_attenuation_dB);

%% SNR estimation

fprintf('\nSIGNAL-TO-NOISE RATIO ESTIMATION:\n');
fprintf('----------------------------------\n');

% Estimate SNR as ratio of baseband power to remaining stopband power
snr_before = 10 * log10(baseband_power_before / stopband_power_before);
snr_after = 10 * log10(baseband_power_after / stopband_power_after);
snr_improvement = snr_after - snr_before;

fprintf('  SNR before lowpass:      %.2f dB\n', snr_before);
fprintf('  SNR after lowpass:       %.2f dB\n', snr_after);
fprintf('  SNR improvement:         %.2f dB\n', snr_improvement);

%% Summary

fprintf('\nTASK 4 SUMMARY:\n');
fprintf('---------------\n');
fprintf('  IIR Filter:              4th order Butterworth lowpass\n');
fprintf('  Cutoff frequency:        %d Hz\n', fc_lowpass);
fprintf('  Baseband preserved:      %.2f dB change\n', baseband_change_dB);
fprintf('  2f_c component removed:  %.2f dB attenuation\n', fc2_attenuation_dB);
fprintf('  Demodulated signal ready for phase optimisation (Task 5)\n');

%% Store demodulated signal for Task 5

% The signal x_demodulated is now ready for Task 5
% Current phase is phi = 0, will be optimised in Task 5
fprintf('\nSignal x_demodulated stored for Task 5.\n');
fprintf('Current carrier phase: phi = %.4f radians (%.2f degrees)\n', phi, phi*180/pi);

%% ========================================================================
%                           TASK 5: AUDIO OUTPUT
%  ========================================================================

fprintf('\n');
fprintf('#                                                            #\n');
fprintf('#                 TASK 5: AUDIO SIGNAL OUTPUT                #\n');
fprintf('#                                                            #\n');

%% Sub-task 5.1: Phase Optimisation

fprintf('\n============================================\n');
fprintf('       PHASE OPTIMISATION                  \n');

%% Technique 1: Coarse Grid Search

fprintf('\nTechnique 1: Coarse Grid Search\n');
fprintf('--------------------------------\n');

% Define coarse phase grid (0 to pi, since cos(phi) = cos(-phi) and we 
% care about |cos(phi)|, and cos(phi + pi) = -cos(phi) just inverts)
n_coarse = 37;  % 37 points gives 5-degree steps from 0 to 180 degrees
phi_coarse = linspace(0, pi, n_coarse);

% Preallocate arrays for results
rms_coarse = zeros(1, n_coarse);
max_amp_coarse = zeros(1, n_coarse);

fprintf('  Testing %d phase values from 0 to %.0f degrees...\n', n_coarse, 180);

% Test each phase value
tic;
for i = 1:n_coarse
    % Generate carrier with this phase
    carrier_test = cos(2*pi*fc_final*t + phi_coarse(i));
    
    % Mix with bandpass filtered signal
    x_mixed_test = x_filtered .* carrier_test;
    
    % Apply lowpass filter
    x_demod_test = custom_iir_filter(b_iir, a_iir, x_mixed_test);
    
    % Calculate metrics
    rms_coarse(i) = sqrt(mean(x_demod_test.^2));
    max_amp_coarse(i) = max(abs(x_demod_test));
end
time_coarse = toc;

% Find best phase from coarse search
[best_rms_coarse, best_idx_coarse] = max(rms_coarse);
phi_best_coarse = phi_coarse(best_idx_coarse);

fprintf('  Search completed in %.2f seconds\n', time_coarse);
fprintf('  Best phase (coarse):     %.4f rad (%.2f degrees)\n', phi_best_coarse, phi_best_coarse*180/pi);
fprintf('  Best RMS amplitude:      %.6f\n', best_rms_coarse);

%% Technique 2: Fine Grid Search (around coarse optimum)

fprintf('\nTechnique 2: Fine Grid Search\n');
fprintf('------------------------------\n');

% Define fine search range around coarse optimum
phi_range = pi/18;  % +/- 10 degrees
phi_fine_min = max(0, phi_best_coarse - phi_range);
phi_fine_max = min(pi, phi_best_coarse + phi_range);
n_fine = 41;  % 41 points gives 0.5-degree steps over 20-degree range
phi_fine = linspace(phi_fine_min, phi_fine_max, n_fine);

% Preallocate arrays
rms_fine = zeros(1, n_fine);
max_amp_fine = zeros(1, n_fine);

fprintf('  Refining search from %.2f to %.2f degrees...\n', phi_fine_min*180/pi, phi_fine_max*180/pi);

% Test each fine phase value
tic;
for i = 1:n_fine
    % Generate carrier with this phase
    carrier_test = cos(2*pi*fc_final*t + phi_fine(i));
    
    % Mix with bandpass filtered signal
    x_mixed_test = x_filtered .* carrier_test;
    
    % Apply lowpass filter
    x_demod_test = custom_iir_filter(b_iir, a_iir, x_mixed_test);
    
    % Calculate metrics
    rms_fine(i) = sqrt(mean(x_demod_test.^2));
    max_amp_fine(i) = max(abs(x_demod_test));
end
time_fine = toc;

% Find best phase from fine search
[best_rms_fine, best_idx_fine] = max(rms_fine);
phi_best_fine = phi_fine(best_idx_fine);

fprintf('  Search completed in %.2f seconds\n', time_fine);
fprintf('  Best phase (fine):       %.4f rad (%.4f degrees)\n', phi_best_fine, phi_best_fine*180/pi);
fprintf('  Best RMS amplitude:      %.6f\n', best_rms_fine);
fprintf('  Improvement over coarse: %.4f%%\n', (best_rms_fine/best_rms_coarse - 1)*100);

%% Technique 3: Golden Section Search (efficient optimisation)

fprintf('\nTechnique 3: Golden Section Search\n');
fprintf('------------------------------------\n');

% Golden ratio
golden_ratio = (1 + sqrt(5)) / 2;

% Define objective function (negative because we want to maximise)
function rms_val = calculate_rms_for_phase(phi_test, x_filt, fc, t_vec, b_coef, a_coef)
    carrier = cos(2*pi*fc*t_vec + phi_test);
    x_mix = x_filt .* carrier;
    x_dem = custom_iir_filter(b_coef, a_coef, x_mix);
    rms_val = sqrt(mean(x_dem.^2));
end

% Golden section search parameters
a_gs = 0;           % Lower bound
b_gs = pi;          % Upper bound
tol_gs = 1e-6;      % Tolerance (radians)
max_iter = 50;      % Maximum iterations

% Initial interior points
c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
d_gs = a_gs + (b_gs - a_gs) / golden_ratio;

% Evaluate at initial points
tic;
fc_val = calculate_rms_for_phase(c_gs, x_filtered, fc_final, t, b_iir, a_iir);
fd_val = calculate_rms_for_phase(d_gs, x_filtered, fc_final, t, b_iir, a_iir);

iter_count = 0;
convergence_history = [];

fprintf('  Initial bounds: [%.4f, %.4f] rad\n', a_gs, b_gs);

while (b_gs - a_gs) > tol_gs && iter_count < max_iter
    iter_count = iter_count + 1;
    
    if fc_val > fd_val
        % Maximum is in [a, d]
        b_gs = d_gs;
        d_gs = c_gs;
        fd_val = fc_val;
        c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
        fc_val = calculate_rms_for_phase(c_gs, x_filtered, fc_final, t, b_iir, a_iir);
    else
        % Maximum is in [c, b]
        a_gs = c_gs;
        c_gs = d_gs;
        fc_val = fd_val;
        d_gs = a_gs + (b_gs - a_gs) / golden_ratio;
        fd_val = calculate_rms_for_phase(d_gs, x_filtered, fc_final, t, b_iir, a_iir);
    end
    
    % Store convergence history
    convergence_history(iter_count) = (a_gs + b_gs) / 2;
end

time_golden = toc;

% Final optimum
phi_best_golden = (a_gs + b_gs) / 2;
best_rms_golden = calculate_rms_for_phase(phi_best_golden, x_filtered, fc_final, t, b_iir, a_iir);

fprintf('  Iterations:              %d\n', iter_count);
fprintf('  Search completed in:     %.2f seconds\n', time_golden);
fprintf('  Best phase (golden):     %.6f rad (%.4f degrees)\n', phi_best_golden, phi_best_golden*180/pi);
fprintf('  Best RMS amplitude:      %.6f\n', best_rms_golden);
fprintf('  Final interval width:    %.2e rad\n', b_gs - a_gs);

%% Compare all techniques

fprintf('\nPHASE OPTIMISATION COMPARISON:\n');
fprintf('-------------------------------\n');
fprintf('  Technique          Phase (deg)    RMS Amplitude    Time (s)\n');
fprintf('  ---------          -----------    -------------    --------\n');
fprintf('  Coarse Grid        %8.4f       %.6f         %.2f\n', phi_best_coarse*180/pi, best_rms_coarse, time_coarse);
fprintf('  Fine Grid          %8.4f       %.6f         %.2f\n', phi_best_fine*180/pi, best_rms_fine, time_fine);
fprintf('  Golden Section     %8.4f       %.6f         %.2f\n', phi_best_golden*180/pi, best_rms_golden, time_golden);

% Select the best overall phase
[~, best_technique] = max([best_rms_coarse, best_rms_fine, best_rms_golden]);
phi_values = [phi_best_coarse, phi_best_fine, phi_best_golden];
rms_values = [best_rms_coarse, best_rms_fine, best_rms_golden];
technique_names = {'Coarse Grid', 'Fine Grid', 'Golden Section'};

phi_optimal = phi_values(best_technique);
rms_optimal = rms_values(best_technique);

fprintf('\nOPTIMAL PHASE SELECTED:\n');
fprintf('  Technique:               %s\n', technique_names{best_technique});
fprintf('  Optimal phase:           %.6f rad (%.4f degrees)\n', phi_optimal, phi_optimal*180/pi);
fprintf('  cos(phi_optimal):        %.6f\n', cos(phi_optimal));
fprintf('  Expected amplitude scale: %.4f (= 0.5 * |cos(phi)|)\n', 0.5 * abs(cos(phi_optimal)));

%% Plot optimisation results

figure('Name', 'Phase Optimisation Results', 'Position', [100, 100, 1200, 800]);

% Plot 1: Coarse grid search results
subplot(2,2,1);
plot(phi_coarse*180/pi, rms_coarse, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 10);
hold on;
plot(phi_best_coarse*180/pi, best_rms_coarse, 'ro', 'MarkerSize', 15, 'LineWidth', 2);
hold off;
xlabel('Phase (degrees)', 'FontSize', 11);
ylabel('RMS Amplitude', 'FontSize', 11);
title('Coarse Grid Search (5° steps)', 'FontSize', 12);
legend('RMS vs Phase', sprintf('Optimum: %.1f°', phi_best_coarse*180/pi), 'Location', 'best');
grid on;
xlim([0, 180]);

% Plot 2: Fine grid search results
subplot(2,2,2);
plot(phi_fine*180/pi, rms_fine, 'g.-', 'LineWidth', 1.5, 'MarkerSize', 10);
hold on;
plot(phi_best_fine*180/pi, best_rms_fine, 'ro', 'MarkerSize', 15, 'LineWidth', 2);
hold off;
xlabel('Phase (degrees)', 'FontSize', 11);
ylabel('RMS Amplitude', 'FontSize', 11);
title('Fine Grid Search (0.5° steps)', 'FontSize', 12);
legend('RMS vs Phase', sprintf('Optimum: %.2f°', phi_best_fine*180/pi), 'Location', 'best');
grid on;

% Plot 3: Golden section convergence
subplot(2,2,3);
plot(1:iter_count, convergence_history*180/pi, 'm.-', 'LineWidth', 1.5, 'MarkerSize', 10);
hold on;
yline(phi_best_golden*180/pi, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('Iteration', 'FontSize', 11);
ylabel('Phase Estimate (degrees)', 'FontSize', 11);
title('Golden Section Search Convergence', 'FontSize', 12);
legend('Phase estimate', sprintf('Final: %.4f°', phi_best_golden*180/pi), 'Location', 'best');
grid on;

% Plot 4: Theoretical cos(phi) curve vs measured
subplot(2,2,4);
phi_theory = linspace(0, pi, 100);
cos_theory = abs(cos(phi_theory));
% Normalise measured RMS to compare shape
rms_normalised = rms_coarse / max(rms_coarse);

plot(phi_theory*180/pi, cos_theory, 'k-', 'LineWidth', 2);
hold on;
plot(phi_coarse*180/pi, rms_normalised, 'b.', 'MarkerSize', 15);
plot(phi_optimal*180/pi, abs(cos(phi_optimal)), 'ro', 'MarkerSize', 15, 'LineWidth', 2);
hold off;
xlabel('Phase (degrees)', 'FontSize', 11);
ylabel('Normalised Amplitude', 'FontSize', 11);
title('Measured vs Theoretical |cos(\phi)|', 'FontSize', 12);
legend('Theoretical |cos(\phi)|', 'Measured (normalised)', sprintf('Optimal: %.1f°', phi_optimal*180/pi), 'Location', 'best');
grid on;
xlim([0, 180]);

%% Sub-task 5.2: Generate Optimally Demodulated Signal
fprintf('       FINAL DEMODULATION                  \n');

% Generate carrier with optimal phase
carrier_optimal = cos(2*pi*fc_final*t + phi_optimal);

% Mix with bandpass filtered signal
x_mixed_optimal = x_filtered .* carrier_optimal;

% Apply lowpass filter
x_final = custom_iir_filter(b_iir, a_iir, x_mixed_optimal);

fprintf('  Carrier frequency:       %d Hz\n', fc_final);
fprintf('  Optimal phase:           %.4f rad (%.2f degrees)\n', phi_optimal, phi_optimal*180/pi);
fprintf('  Final signal length:     %d samples (%.2f seconds)\n', length(x_final), length(x_final)/fs);

%% Sub-task 5.3: SNR Measurement
fprintf('       SNR MEASUREMENT                     \n');

% Compute spectrum of final demodulated signal
window_final = hamming(length(x_final))';
x_final_windowed = x_final .* window_final;
X_final = fft(x_final_windowed);
X_final_magnitude = abs(X_final) / length(X_final);

N_final = length(X_final);
X_final_single = X_final_magnitude(1:floor(N_final/2)+1);
X_final_single(2:end-1) = 2 * X_final_single(2:end-1);
f_final = (0:floor(N_final/2)) * fs / N_final;
X_final_dB = 20 * log10(X_final_single + eps);

%% Method 1: Speech band vs out-of-band noise

fprintf('\nMethod 1: Speech Band Power vs Out-of-Band Noise\n');
fprintf('-------------------------------------------------\n');

% Define frequency bands
speech_low = 300;       % Speech starts around 300 Hz
speech_high = 3400;     % Speech ends around 3400 Hz (telephone quality)
noise_low = 3800;       % Noise region above speech
noise_high = 4000;      % Up to filter cutoff

% Calculate power in speech band
speech_idx = find(f_final >= speech_low & f_final <= speech_high);
speech_power = sum(X_final_single(speech_idx).^2);

% Calculate power in noise region (just above speech)
noise_idx = find(f_final >= noise_low & f_final <= noise_high);
if ~isempty(noise_idx)
    noise_power_method1 = sum(X_final_single(noise_idx).^2);
    % Scale to equivalent bandwidth
    noise_bw = noise_high - noise_low;
    speech_bw = speech_high - speech_low;
    noise_power_scaled = noise_power_method1 * (speech_bw / noise_bw);
    snr_method1 = 10 * log10(speech_power / noise_power_scaled);
else
    snr_method1 = NaN;
end

fprintf('  Speech band:             %d - %d Hz\n', speech_low, speech_high);
fprintf('  Speech power:            %.6e\n', speech_power);
fprintf('  Noise band:              %d - %d Hz\n', noise_low, noise_high);
fprintf('  Noise power (scaled):    %.6e\n', noise_power_scaled);
fprintf('  SNR (Method 1):          %.2f dB\n', snr_method1);

%% Method 2: Signal peaks vs RMS noise floor

fprintf('\nMethod 2: Peak Signal vs RMS Noise Floor\n');
fprintf('------------------------------------------\n');

% Find spectral peaks (signal components)
[peaks, peak_locs] = findpeaks(X_final_single, 'MinPeakHeight', max(X_final_single)*0.1);
peak_freqs = f_final(peak_locs);

% Only consider peaks in speech band
speech_peak_idx = find(peak_freqs >= speech_low & peak_freqs <= speech_high);
if ~isempty(speech_peak_idx)
    signal_power_peaks = sum(peaks(speech_peak_idx).^2);
else
    signal_power_peaks = speech_power;
end

% Estimate noise floor as median of spectrum (robust to peaks)
noise_floor = median(X_final_single(speech_idx));
noise_power_method2 = noise_floor^2 * length(speech_idx);

snr_method2 = 10 * log10(signal_power_peaks / noise_power_method2);

fprintf('  Number of peaks found:   %d\n', length(speech_peak_idx));
fprintf('  Signal power (peaks):    %.6e\n', signal_power_peaks);
fprintf('  Noise floor (median):    %.6e\n', noise_floor);
fprintf('  Noise power:             %.6e\n', noise_power_method2);
fprintf('  SNR (Method 2):          %.2f dB\n', snr_method2);

%% Method 3: Time-domain SNR estimation using signal envelope

fprintf('\nMethod 3: Time-Domain Envelope Analysis\n');
fprintf('-----------------------------------------\n');

% Compute signal envelope using Hilbert transform
signal_envelope = abs(hilbert(x_final));

% Estimate signal power from envelope peaks
envelope_threshold = 0.3 * max(signal_envelope);
signal_samples = signal_envelope > envelope_threshold;
signal_power_time = mean(x_final(signal_samples).^2);

% Estimate noise power from quiet regions
noise_samples = signal_envelope < 0.1 * max(signal_envelope);
if sum(noise_samples) > 100
    noise_power_time = mean(x_final(noise_samples).^2);
else
    % If no quiet regions, use overall variance minus signal
    noise_power_time = var(x_final) - signal_power_time * mean(signal_samples);
    noise_power_time = max(noise_power_time, eps);  % Ensure positive
end

snr_method3 = 10 * log10(signal_power_time / noise_power_time);

fprintf('  Signal samples:          %d (%.1f%% of total)\n', sum(signal_samples), 100*mean(signal_samples));
fprintf('  Signal power:            %.6e\n', signal_power_time);
fprintf('  Noise samples:           %d (%.1f%% of total)\n', sum(noise_samples), 100*mean(noise_samples));
fprintf('  Noise power:             %.6e\n', noise_power_time);
fprintf('  SNR (Method 3):          %.2f dB\n', snr_method3);

%% Method 4: Compare with phi = 0 (unoptimised)

fprintf('\nMethod 4: Improvement from Phase Optimisation\n');
fprintf('-----------------------------------------------\n');

% Calculate RMS with phi = 0
carrier_zero = cos(2*pi*fc_final*t);
x_mixed_zero = x_filtered .* carrier_zero;
x_demod_zero = custom_iir_filter(b_iir, a_iir, x_mixed_zero);
rms_zero = sqrt(mean(x_demod_zero.^2));

% Calculate RMS with optimal phase
rms_final = sqrt(mean(x_final.^2));

% Calculate improvement
amplitude_improvement = rms_final / rms_zero;
power_improvement_dB = 20 * log10(amplitude_improvement);

fprintf('  RMS with phi = 0:        %.6f\n', rms_zero);
fprintf('  RMS with phi optimal:    %.6f\n', rms_final);
fprintf('  Amplitude improvement:   %.4fx\n', amplitude_improvement);
fprintf('  Power improvement:       %.2f dB\n', power_improvement_dB);

%% SNR Summary

fprintf('\nSNR MEASUREMENT SUMMARY:\n');
fprintf('-------------------------\n');
fprintf('  Method 1 (Frequency bands):     %.2f dB\n', snr_method1);
fprintf('  Method 2 (Peaks vs floor):      %.2f dB\n', snr_method2);
fprintf('  Method 3 (Time envelope):       %.2f dB\n', snr_method3);
fprintf('  Average SNR:                    %.2f dB\n', mean([snr_method1, snr_method2, snr_method3], 'omitnan'));

%% Plot SNR analysis

figure('Name', 'SNR Analysis', 'Position', [100, 100, 1200, 800]);

% Plot 1: Spectrum with frequency bands marked
subplot(2,2,1);
plot(f_final/1000, X_final_dB, 'b', 'LineWidth', 1);
hold on;
% Mark speech band
xline(speech_low/1000, 'g--', 'LineWidth', 1.5);
xline(speech_high/1000, 'g--', 'LineWidth', 1.5);
% Mark noise band
xline(noise_low/1000, 'r--', 'LineWidth', 1.5);
xline(noise_high/1000, 'r--', 'LineWidth', 1.5);
% Mark noise floor
yline(20*log10(noise_floor + eps), 'm--', 'LineWidth', 1.5);
hold off;
xlabel('Frequency (kHz)', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('Demodulated Signal Spectrum with Analysis Bands', 'FontSize', 12);
legend('Spectrum', 'Speech band', '', 'Noise band', '', 'Noise floor', 'Location', 'northeast');
grid on;
xlim([0, 5]);

% Plot 2: Time-domain envelope
subplot(2,2,2);
plot(t, x_final, 'b', 'LineWidth', 0.5);
hold on;
plot(t, signal_envelope, 'r', 'LineWidth', 1);
yline(envelope_threshold, 'g--', 'LineWidth', 1.5);
yline(0.1*max(signal_envelope), 'm--', 'LineWidth', 1.5);
hold off;
xlabel('Time (seconds)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Signal and Envelope for SNR Estimation', 'FontSize', 12);
legend('Signal', 'Envelope', 'Signal threshold', 'Noise threshold', 'Location', 'best');
grid on;

% Plot 3: Comparison phi=0 vs phi optimal
subplot(2,2,3);
plot_duration = 0.1;  % 100 ms
plot_samples = round(plot_duration * fs);
plot_start = round(length(t) / 2);
plot_end = min(plot_start + plot_samples, length(t));
t_portion = t(plot_start:plot_end);

plot(t_portion, x_demod_zero(plot_start:plot_end), 'b', 'LineWidth', 1);
hold on;
plot(t_portion, x_final(plot_start:plot_end), 'r', 'LineWidth', 1.5);
hold off;
xlabel('Time (seconds)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Phase Optimisation Effect (100 ms portion)', 'FontSize', 12);
legend(sprintf('\\phi = 0° (RMS = %.4f)', rms_zero), ...
       sprintf('\\phi = %.1f° (RMS = %.4f)', phi_optimal*180/pi, rms_final), 'Location', 'best');
grid on;

% Plot 4: SNR comparison bar chart
subplot(2,2,4);
snr_values = [snr_method1, snr_method2, snr_method3];
bar_colors = [0.2 0.6 0.8; 0.8 0.4 0.2; 0.4 0.8 0.4];
b = bar(1:3, snr_values);
b.FaceColor = 'flat';
b.CData = bar_colors;
hold on;
yline(mean(snr_values, 'omitnan'), 'k--', 'LineWidth', 2);
hold off;
set(gca, 'XTickLabel', {'Freq. Bands', 'Peaks/Floor', 'Time Envelope'});
xlabel('SNR Estimation Method', 'FontSize', 11);
ylabel('SNR (dB)', 'FontSize', 11);
title('SNR Estimates by Different Methods', 'FontSize', 12);
legend('SNR', sprintf('Average: %.1f dB', mean(snr_values, 'omitnan')), 'Location', 'best');
grid on;

%% Sub-task 5.4: Audio Playback and Message Identification

fprintf('       AUDIO PLAYBACK                      \n');

% Normalise for playback (prevent clipping)
x_playback = x_final / max(abs(x_final)) * 0.9;

fprintf('  Playing demodulated audio...\n');
fprintf('  Listen carefully for the 3-letter message!\n');

% Play the audio
sound(x_playback, fs);

% Wait for playback to complete
pause(length(x_playback)/fs + 0.5);

fprintf('\n  Playback complete.\n');

%% Save audio file for reference

output_filename = 'demodulated_message.wav';
audiowrite(output_filename, x_playback, fs);
fprintf('\n  Audio saved to: %s\n', output_filename);

%% Final message identification prompt

fprintf('#                 MESSAGE IDENTIFICATION                     #\n');
fprintf('\n');
fprintf('  Listen to the audio output and identify the 3-letter message.\n');
fprintf('\n');
fprintf('  Enter the message you heard: ');
message = input('', 's');
fprintf('\n');
fprintf('  You identified the message as: %s\n', upper(message));
fprintf('\n');

%% Complete summary
fprintf('\n');
fprintf('#                 DEMODULATION COMPLETE                      #\n');
fprintf('\n');
fprintf('SIGNAL PROCESSING CHAIN SUMMARY:\n');
fprintf('  Task 1: Time/Frequency Analysis\n');
fprintf('    - Carrier frequency identified: %d Hz\n', fc_final);
fprintf('    - AM band: %d - %d Hz\n', fmin, fmax);
fprintf('\n');
fprintf('  Task 2: FIR Bandpass Filter\n');
fprintf('    - Filter order: %d taps\n', length(h_bp));
fprintf('    - Passband: %d - %d Hz\n', fmin, fmax);
fprintf('    - Window: Hamming\n');
fprintf('\n');
fprintf('  Task 3: Carrier Recovery and Mixing\n');
fprintf('    - Carrier recovered via square law at 2f_c = %d Hz\n', 2*fc_final);
fprintf('    - Mixed to baseband (0-4 kHz)\n');
fprintf('\n');
fprintf('  Task 4: IIR Lowpass Filter\n');
fprintf('    - 4th order Butterworth\n');
fprintf('    - Cutoff frequency: %d Hz\n', fc_lowpass);
fprintf('\n');
fprintf('  Task 5: Phase Optimisation and Audio Output\n');
fprintf('    - Optimal phase: %.2f degrees\n', phi_optimal*180/pi);
fprintf('    - Estimated SNR: %.1f dB (average)\n', mean([snr_method1, snr_method2, snr_method3], 'omitnan'));
fprintf('    - Message identified: %s\n', upper(message));
fprintf('\n');