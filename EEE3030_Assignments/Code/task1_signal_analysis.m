function [signal, params] = task1_signal_analysis(filename)
% TASK1_SIGNAL_ANALYSIS - Read and analyse AM signal in time and frequency domain
%
% Usage:
%   [signal, params] = task1_signal_analysis(filename)
%
% Inputs:
%   filename - Path to the .wav audio file
%
% Outputs:
%   signal   - Structure containing signal data:
%              .x        - Raw signal (row vector)
%              .t        - Time vector
%              .X_dB     - Single-sided magnitude spectrum (dB)
%              .f        - Frequency vector for spectrum
%   params   - Structure containing signal parameters:
%              .fs       - Sampling frequency
%              .N        - Number of samples
%              .duration - Signal duration (seconds)
%              .fmin     - Lower AM band edge (Hz)
%              .fmax     - Upper AM band edge (Hz)
%              .fc       - Estimated carrier frequency (Hz)
%
% EEE3030 DSP Assignment - Task 1

%% Read the audio file
fprintf('Reading audio file: %s\n', filename);
[x, fs] = audioread(filename);

% Convert stereo to mono if necessary
if size(x, 2) > 1
    x = x(:, 1);
    fprintf('Note: Stereo file detected, using first channel only.\n');
end

% Ensure row vector
if iscolumn(x)
    x = x';
end

%% Calculate basic signal properties
N = length(x);
duration = N / fs;
freq_resolution = fs / N;
nyquist_freq = fs / 2;

fprintf('\nSIGNAL PROPERTIES:\n');
fprintf('  Sampling frequency:    %d Hz\n', fs);
fprintf('  Number of samples:     %d\n', N);
fprintf('  Signal duration:       %.4f seconds\n', duration);
fprintf('  Frequency resolution:  %.4f Hz\n', freq_resolution);
fprintf('  Nyquist frequency:     %d Hz\n', nyquist_freq);

%% Amplitude statistics
fprintf('\nAMPLITUDE STATISTICS:\n');
fprintf('  Maximum amplitude:     %.6f\n', max(x));
fprintf('  Minimum amplitude:     %.6f\n', min(x));
fprintf('  Peak-to-peak:          %.6f\n', max(x) - min(x));
fprintf('  RMS amplitude:         %.6f\n', sqrt(mean(x.^2)));

%% Time domain visualisation
t = (0:N-1) / fs;

figure('Name', 'Task 1 - Time Domain', 'Position', [100, 100, 1200, 400]);
plot(t, x, 'b', 'LineWidth', 0.3);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Time Domain Signal - Full Duration');
grid on;
xlim([0, duration]);

%% Spectrogram
figure('Name', 'Task 1 - Spectrogram', 'Position', [100, 100, 800, 600]);
spectrogram(x, 1024, 512, 1024, fs, 'yaxis');
title('Spectrogram of Input Signal');
xlabel('Time (s)');
ylabel('Frequency (kHz)');

%% Frequency domain analysis - FFT
X = fft(x);
X_magnitude = abs(X);
X_normalised = X_magnitude / N;

% Single-sided spectrum
num_bins = floor(N/2) + 1;
X_single = X_normalised(1:num_bins);
X_single(2:end-1) = 2 * X_single(2:end-1);

% Frequency vector
f = (0:num_bins-1) * fs / N;

% Convert to dB
X_dB = 20 * log10(X_single + eps);

%% Apply Hamming window for better spectral analysis
hamming_window = 0.54 - 0.46 * cos(2 * pi * (0:N-1) / (N - 1));
CG_hamming = sum(hamming_window) / N;  % Coherent gain

x_hamming = x .* hamming_window;
X_hamming = fft(x_hamming);
X_hamming_magnitude = abs(X_hamming);
X_hamming_normalised = X_hamming_magnitude / N / CG_hamming;
X_hamming_single = X_hamming_normalised(1:num_bins);
X_hamming_single(2:end-1) = 2 * X_hamming_single(2:end-1);
X_hamming_dB = 20 * log10(X_hamming_single + eps);

%% Plot frequency spectrum
figure('Name', 'Task 1 - Frequency Spectrum', 'Position', [100, 100, 1200, 600]);

subplot(2,1,1);
plot(f/1000, X_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Frequency Spectrum - Rectangular Window');
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_dB) + 10]);

subplot(2,1,2);
plot(f/1000, X_hamming_dB, 'r', 'LineWidth', 0.3);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Frequency Spectrum - Hamming Window (53 dB Stopband Attenuation)');
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_hamming_dB) + 10]);

%% Identify AM signal band
% These values should be determined by visual inspection of the spectrum
% Default values based on typical assignment parameters
fmin_measured = 12000;  % Hz - lower edge of AM signal band
fmax_measured = 20000;  % Hz - upper edge of AM signal band

% Calculate derived parameters
fc_estimated = (fmin_measured + fmax_measured) / 2;

% Round carrier to nearest 1 kHz (as per assignment specification)
fc_rounded = round(fc_estimated / 1000) * 1000;

% Recalculate fmin and fmax based on rounded carrier and 8 kHz bandwidth
fmin = fc_rounded - 4000;
fmax = fc_rounded + 4000;

fprintf('\nAM SIGNAL BAND IDENTIFICATION:\n');
fprintf('  Estimated carrier:     %.1f Hz\n', fc_estimated);
fprintf('  Rounded carrier:       %d Hz\n', fc_rounded);
fprintf('  Final f_min:           %d Hz\n', fmin);
fprintf('  Final f_max:           %d Hz\n', fmax);
fprintf('  Bandwidth:             %d Hz\n', fmax - fmin);

%% Plot with identified band marked
figure('Name', 'Task 1 - AM Band Identification', 'Position', [100, 100, 1200, 500]);
plot(f/1000, X_hamming_dB, 'b', 'LineWidth', 0.3);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('AM Signal Spectrum with Identified Band Boundaries');
grid on;
xlim([0, fs/2000]);
ylim([-120, max(X_hamming_dB) + 10]);

hold on;
xline(fmin/1000, 'r-', 'LineWidth', 2);
xline(fmax/1000, 'r-', 'LineWidth', 2);
xline(fc_rounded/1000, 'g--', 'LineWidth', 2);

% Add shaded region
y_limits = ylim;
patch([fmin/1000, fmax/1000, fmax/1000, fmin/1000], ...
      [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
      'green', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

legend('Spectrum', 'f_{min}', 'f_{max}', 'f_c (carrier)', 'Signal Band', 'Location', 'northeast');
hold off;

%% Prepare output structures
signal.x = x;
signal.t = t;
signal.X_dB = X_hamming_dB;
signal.X_single = X_hamming_single;
signal.f = f;
signal.hamming_window = hamming_window;
signal.CG_hamming = CG_hamming;

params.fs = fs;
params.N = N;
params.duration = duration;
params.fmin = fmin;
params.fmax = fmax;
params.fc = fc_rounded;
params.num_bins = num_bins;

fprintf('\nTask 1 complete. Signal and parameters ready for Task 2.\n');

end