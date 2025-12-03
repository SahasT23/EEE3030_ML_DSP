%% EEE3030 Signal Processing Assignment - AM Demodulator (Improved Version)
% 
% Purpose: Demodulate an AM signal to extract a 3-letter spoken message
%
% Signal Specifications:
%   Sampling frequency: 96 kHz
%   Message bandwidth: 4 kHz
%   Modulation type: DSB-SC
%
% Tasks:
%   Task 1: Time and frequency domain analysis
%   Task 2: Bandpass FIR filter design and application
%   Task 3: Carrier recovery and mixing
%   Task 4: Lowpass IIR filter design and application
%   Task 5: Phase optimization and audio extraction

clear all;
close all;
clc;

%% ========================================================================
%  TASK 1: TIME AND FREQUENCY DOMAIN ANALYSIS
%  ========================================================================

fprintf('========================================\n');
fprintf('TASK 1: Signal Analysis\n');
fprintf('========================================\n\n');

% Load the audio signal
% audioread returns the signal vector and sampling frequency
[x, fs] = audioread('Sahas_Talasila.wav');

% Convert stereo to mono if necessary
% We only need one channel for processing
if size(x, 2) > 1
    x = x(:, 1);
end

% Get signal properties
N = length(x);              % Total number of samples
duration = N/fs;            % Duration in seconds
t = (0:N-1)/fs;             % Time vector

% Display basic signal information
fprintf('Signal Properties:\n');
fprintf('  Sampling frequency: %d Hz\n', fs);
fprintf('  Number of samples: %d\n', N);
fprintf('  Signal duration: %.3f seconds\n', duration);
fprintf('  Signal range: [%.4f, %.4f]\n', min(x), max(x));

% Calculate frequency resolution
% Frequency resolution = sampling frequency / FFT length
% This tells us the minimum frequency separation we can distinguish
freq_resolution = fs/N;
fprintf('  Frequency resolution: %.2f Hz\n\n', freq_resolution);

%% Calculate Initial SNR (Input Signal)
% Estimate SNR of input signal before any processing
% Method: Calculate signal power and estimate noise from high-frequency content

fprintf('Initial Signal Quality Assessment:\n');

% Calculate total signal power
signal_power_initial = 0;
for i = 1:N
    signal_power_initial = signal_power_initial + x(i)^2;
end
signal_power_initial = signal_power_initial / N;

% Estimate noise by assuming high-frequency content above expected signal band is noise
% Use simple highpass filter to isolate high-frequency noise
% First-order highpass: y[n] = alpha*(y[n-1] + x[n] - x[n-1])
alpha_hp = 0.95;  % Highpass filter coefficient (closer to 1 = higher cutoff)
noise_estimate_initial = zeros(N, 1);
for i = 2:N
    noise_estimate_initial(i) = alpha_hp * (noise_estimate_initial(i-1) + x(i) - x(i-1));
end

% Calculate noise power
noise_power_initial = 0;
for i = 1:N
    noise_power_initial = noise_power_initial + noise_estimate_initial(i)^2;
end
noise_power_initial = noise_power_initial / N;

% Calculate SNR in dB
% SNR_dB = 10*log10(signal_power / noise_power)
if noise_power_initial > 0
    SNR_initial_dB = 10*log10(signal_power_initial / noise_power_initial);
else
    SNR_initial_dB = Inf;  % Infinite SNR if no noise detected
end

fprintf('  Total signal power: %.6f\n', signal_power_initial);
fprintf('  Estimated noise power: %.6f\n', noise_power_initial);
fprintf('  Estimated input SNR: %.2f dB\n\n', SNR_initial_dB);

%% Time Domain Plot
% Plot the original signal to observe amplitude modulation envelope
figure('Name', 'Task 1 - Time Domain Analysis');
subplot(2,1,1);
plot(t, x, 'b', 'LineWidth', 0.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original AM Signal - Time Domain');
grid on;
xlim([0 duration]);

% Zoomed view to see carrier and envelope more clearly
subplot(2,1,2);
zoom_start = 0.5;
zoom_duration = 0.01;
zoom_idx = round(zoom_start*fs):round((zoom_start+zoom_duration)*fs);
plot(t(zoom_idx), x(zoom_idx), 'b', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Zoomed View (10 ms) - Shows Carrier and Envelope');
grid on;

%% Frequency Domain Analysis with Manual Window Creation
% Create Hamming window manually to avoid built-in hamming() function
% Hamming window formula: w[n] = 0.54 - 0.46*cos(2*pi*n/(N-1))

fprintf('Creating Hamming window manually...\n');

window_hamming = zeros(N, 1);
for n = 1:N
    window_hamming(n) = 0.54 - 0.46*cos(2*pi*(n-1)/(N-1));
end

% Apply window to signal
x_windowed = x .* window_hamming;

% Compute FFT (using built-in fft as manual DFT would be too slow for N>10000)
X_fft = fft(x_windowed);

% Calculate single-sided spectrum manually
% For real signals, FFT is symmetric, so we only need positive frequencies
X_single = X_fft(1:N/2+1);

% Normalize amplitude
% Multiply by 2 (except DC and Nyquist) to account for negative frequencies
X_single(2:end-1) = 2*X_single(2:end-1);
X_magnitude = abs(X_single)/N;

% Convert to dB manually: dB = 20*log10(magnitude)
% Add small epsilon to avoid log(0)
X_dB = zeros(length(X_magnitude), 1);
eps_val = 1e-10;
for i = 1:length(X_magnitude)
    if X_magnitude(i) > eps_val
        X_dB(i) = 20*log10(X_magnitude(i));
    else
        X_dB(i) = 20*log10(eps_val);
    end
end

% Create frequency vector manually
f = zeros(length(X_single), 1);
for i = 1:length(X_single)
    f(i) = (i-1) * fs / (2 * (length(X_single)-1));
end

fprintf('  Hamming window applied\n');
fprintf('  FFT computed\n');
fprintf('  Number of frequency bins: %d\n\n', length(X_single));

%% Identify AM Signal Bandwidth
% Find carrier frequency and sidebands
% AM signal spectrum has peaks at fc-B, fc, and fc+B

fprintf('Identifying Signal Bandwidth:\n');

% Find peaks above threshold
threshold_dB = -40;  % 40 dB below maximum

% Find indices where spectrum exceeds threshold
peak_indices = [];
for i = 1:length(X_dB)
    if X_dB(i) > threshold_dB
        peak_indices = [peak_indices; i];
    end
end

% Get corresponding frequencies
peak_freqs = f(peak_indices);

% Find frequency band limits
fmin_initial = min(peak_freqs);
fmax_initial = max(peak_freqs);

fprintf('  Initial frequency range: %.2f - %.2f Hz\n', fmin_initial, fmax_initial);

% Refine using known message bandwidth B = 4 kHz
% Signal occupies fc +/- B
B = 4000;  % Message bandwidth in Hz
fc_estimate = (fmin_initial + fmax_initial) / 2;
fmin = fc_estimate - B;
fmax = fc_estimate + B;

fprintf('  Estimated carrier frequency: %.2f Hz\n', fc_estimate);
fprintf('  Refined bandwidth: %.2f - %.2f Hz\n', fmin, fmax);
fprintf('  Total bandwidth: %.2f Hz\n\n', fmax - fmin);

% Plot spectrum with bandwidth annotation
figure('Name', 'Task 1 - Frequency Spectrum');
plot(f/1000, X_dB, 'b', 'LineWidth', 1);
hold on;
xline(fmin/1000, 'r--', 'LineWidth', 2);
xline(fmax/1000, 'r--', 'LineWidth', 2);
xline(fc_estimate/1000, 'g--', 'LineWidth', 2);
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Signal Spectrum (Hamming Window)');
grid on;
xlim([0 fs/2000]);
ylim([-120 0]);
legend('Spectrum', 'fmin', 'fmax', 'fc', 'Location', 'best');
hold off;

%% ========================================================================
%  TASK 2: BANDPASS FIR FILTER DESIGN
%  ========================================================================

fprintf('========================================\n');
fprintf('TASK 2: Bandpass FIR Filter Design\n');
fprintf('========================================\n\n');

% Filter specifications from assignment
fp1 = fmin;                  % Lower passband edge
fp2 = fmax;                  % Upper passband edge
fstop1 = fmin - 2000;        % Lower stopband edge
fstop2 = fmax + 2000;        % Upper stopband edge
transition_bw = 2000;        % Transition bandwidth

fprintf('Filter Specifications:\n');
fprintf('  Passband: %.2f - %.2f Hz\n', fp1, fp2);
fprintf('  Stopband: DC-%.2f Hz and %.2f-Nyquist Hz\n', fstop1, fstop2);
fprintf('  Transition bandwidth: %.2f Hz\n', transition_bw);
fprintf('  Required attenuation: > 50 dB\n\n');

%% Calculate FIR Filter Order
% For Hamming window: transition_width = 3.3/N (normalized)
% N = 3.3 / (transition_bw / fs)

delta_F = transition_bw / fs;  % Normalized transition width
N_fir = ceil(3.3 / delta_F);

% Ensure odd number of taps for symmetric linear-phase filter
if mod(N_fir, 2) == 0
    N_fir = N_fir + 1;
end

M = (N_fir - 1) / 2;

fprintf('FIR Filter Design:\n');
fprintf('  Calculated filter order: %d taps\n', N_fir);
fprintf('  M (half-length): %d\n\n', M);

%% Design Bandpass FIR Filter Using Impulse Response Truncation
% Bandpass = Lowpass(fc2) - Lowpass(fc1)
% Generate ideal impulse response then apply window

% Normalize cutoff frequencies to sampling frequency
Fc1 = fp1 / fs;
Fc2 = fp2 / fs;

fprintf('  Normalized cutoffs: Fc1 = %.4f, Fc2 = %.4f\n', Fc1, Fc2);

% Generate sample indices centered at zero
n_vals = -M:M;

% Calculate ideal bandpass impulse response manually
h_ideal = zeros(1, N_fir);
for i = 1:N_fir
    n_val = n_vals(i);
    
    if n_val == 0
        % At n=0, use limit formula
        h_ideal(i) = 2*(Fc2 - Fc1);
    else
        % For n≠0, use sinc formula
        h_ideal(i) = (sin(2*pi*Fc2*n_val) - sin(2*pi*Fc1*n_val)) / (pi*n_val);
    end
end

fprintf('  Ideal impulse response calculated\n');

% Create Hamming window for FIR filter manually
w_hamming_fir = zeros(1, N_fir);
for i = 1:N_fir
    w_hamming_fir(i) = 0.54 - 0.46*cos(2*pi*(i-1)/(N_fir-1));
end

% Apply window to ideal response
h_bp = h_ideal .* w_hamming_fir;

fprintf('  Hamming window applied to impulse response\n');

% Normalize filter to have unity gain at center frequency
% Calculate frequency response at center frequency manually
fc_center = (Fc1 + Fc2) / 2;
H_center_real = 0;
H_center_imag = 0;
for i = 1:N_fir
    angle = -2*pi*fc_center*(i-1);
    H_center_real = H_center_real + h_bp(i)*cos(angle);
    H_center_imag = H_center_imag + h_bp(i)*sin(angle);
end
H_center_mag = sqrt(H_center_real^2 + H_center_imag^2);

% Normalize coefficients
h_bp = h_bp / H_center_mag;

fprintf('  Filter normalized (unity gain at %.2f Hz)\n\n', fc_center*fs);

%% Apply Bandpass Filter Using Custom Convolution
fprintf('Applying bandpass filter to signal...\n');

% Use custom convolution function
x_bp = custom_conv(x, h_bp);

fprintf('  Filtering complete (output length: %d samples)\n\n', length(x_bp));

%% Calculate SNR After Bandpass Filtering
fprintf('Signal Quality After Bandpass Filtering:\n');

% Calculate signal power after filtering
signal_power_bp = 0;
for i = 1:length(x_bp)
    signal_power_bp = signal_power_bp + x_bp(i)^2;
end
signal_power_bp = signal_power_bp / length(x_bp);

% Estimate noise using highpass filter on filtered signal
noise_estimate_bp = zeros(length(x_bp), 1);
for i = 2:length(x_bp)
    noise_estimate_bp(i) = alpha_hp * (noise_estimate_bp(i-1) + x_bp(i) - x_bp(i-1));
end

% Calculate noise power
noise_power_bp = 0;
for i = 1:length(noise_estimate_bp)
    noise_power_bp = noise_power_bp + noise_estimate_bp(i)^2;
end
noise_power_bp = noise_power_bp / length(noise_estimate_bp);

% Calculate SNR
if noise_power_bp > 0
    SNR_bp_dB = 10*log10(signal_power_bp / noise_power_bp);
else
    SNR_bp_dB = Inf;
end

fprintf('  Signal power: %.6f\n', signal_power_bp);
fprintf('  Noise power: %.6f\n', noise_power_bp);
fprintf('  SNR after bandpass: %.2f dB\n', SNR_bp_dB);
fprintf('  SNR improvement: %.2f dB\n\n', SNR_bp_dB - SNR_initial_dB);

%% ========================================================================
%  TASK 3: CARRIER RECOVERY AND MIXING
%  ========================================================================

fprintf('========================================\n');
fprintf('TASK 3: Carrier Recovery and Mixing\n');
fprintf('========================================\n\n');

%% Carrier Recovery Using Squaring Method
fprintf('Recovering carrier frequency...\n');

% Square the bandpass filtered signal
% This creates component at 2*fc
x_squared = x_bp .^ 2;

% Window and compute FFT
N_squared = length(x_squared);
window_squared = zeros(N_squared, 1);
for n = 1:N_squared
    window_squared(n) = 0.54 - 0.46*cos(2*pi*(n-1)/(N_squared-1));
end

x_squared_windowed = x_squared .* window_squared;
X_squared_fft = fft(x_squared_windowed);

% Single-sided spectrum
X_squared_single = X_squared_fft(1:N_squared/2+1);
X_squared_single(2:end-1) = 2*X_squared_single(2:end-1);
X_squared_mag = abs(X_squared_single)/N_squared;

% Create frequency vector
f_squared = zeros(length(X_squared_single), 1);
for i = 1:length(X_squared_single)
    f_squared(i) = (i-1) * fs / (2 * (length(X_squared_single)-1));
end

% Find 2*fc peak (search in reasonable range)
search_min = 10000;  % 10 kHz
search_max = 40000;  % 40 kHz

max_magnitude = 0;
max_freq_idx = 1;

for i = 1:length(f_squared)
    if f_squared(i) >= search_min && f_squared(i) <= search_max
        if X_squared_mag(i) > max_magnitude
            max_magnitude = X_squared_mag(i);
            max_freq_idx = i;
        end
    end
end

fc_2x = f_squared(max_freq_idx);
fc = fc_2x / 2;

% Round to nearest kHz (carrier frequencies are typically at exact kHz)
fc = round(fc/1000) * 1000;

fprintf('  Found 2*fc peak at: %.2f Hz\n', fc_2x);
fprintf('  Recovered carrier frequency: %.2f Hz\n\n', fc);

%% Mix Signal with Local Carrier
fprintf('Mixing signal with local carrier...\n');

% Initial phase (will optimize in Task 5)
phi = 0;

% Generate local carrier signal manually
carrier = zeros(length(x_bp), 1);
for i = 1:length(x_bp)
    carrier(i) = cos(2*pi*fc*t(i) + phi);
end

% Multiply (mix) signals
x_mixed = x_bp .* carrier;

fprintf('  Mixing complete\n');
fprintf('  Mixed signal contains baseband message + 2*fc component\n\n');

%% ========================================================================
%  TASK 4: LOWPASS IIR FILTER DESIGN
%  ========================================================================

fprintf('========================================\n');
fprintf('TASK 4: Lowpass IIR Filter Design\n');
fprintf('========================================\n\n');

% IIR filter specifications from assignment
iir_order = 4;
fc_lp = 4000;  % 4 kHz cutoff
Wn = fc_lp / (fs/2);  % Normalized cutoff frequency

fprintf('IIR Filter Specifications:\n');
fprintf('  Type: Butterworth (maximally flat passband)\n');
fprintf('  Order: %d\n', iir_order);
fprintf('  Cutoff frequency: %.0f Hz\n', fc_lp);
fprintf('  Normalized cutoff: %.4f\n\n', Wn);

%% Design Butterworth Filter
% Use built-in butter() function as manual Butterworth design is complex
% This calculates the required bilinear transform coefficients

[b_iir, a_iir] = butter(iir_order, Wn, 'low');

fprintf('  Butterworth coefficients calculated\n');
fprintf('  b (numerator) length: %d\n', length(b_iir));
fprintf('  a (denominator) length: %d\n\n', length(a_iir));

%% Apply IIR Filter Using Custom Implementation
fprintf('Applying lowpass filter to remove 2*fc component...\n');

% Use custom IIR filter function
x_demod = custom_iir_filter(b_iir, a_iir, x_mixed);

fprintf('  Lowpass filtering complete\n');
fprintf('  Output length: %d samples\n\n', length(x_demod));

%% Calculate SNR After Initial Demodulation (phi=0)
fprintf('Signal Quality After Demodulation (phi=0):\n');

% Calculate signal power
signal_power_demod = 0;
for i = 1:length(x_demod)
    signal_power_demod = signal_power_demod + x_demod(i)^2;
end
signal_power_demod = signal_power_demod / length(x_demod);

% Estimate noise
noise_estimate_demod = zeros(length(x_demod), 1);
for i = 2:length(x_demod)
    noise_estimate_demod(i) = alpha_hp * (noise_estimate_demod(i-1) + x_demod(i) - x_demod(i-1));
end

noise_power_demod = 0;
for i = 1:length(noise_estimate_demod)
    noise_power_demod = noise_power_demod + noise_estimate_demod(i)^2;
end
noise_power_demod = noise_power_demod / length(noise_estimate_demod);

% Calculate SNR
if noise_power_demod > 0
    SNR_demod_initial_dB = 10*log10(signal_power_demod / noise_power_demod);
else
    SNR_demod_initial_dB = Inf;
end

fprintf('  Signal power: %.6f\n', signal_power_demod);
fprintf('  Noise power: %.6f\n', noise_power_demod);
fprintf('  SNR (phi=0): %.2f dB\n\n', SNR_demod_initial_dB);

%% ========================================================================
%  TASK 5: PHASE OPTIMIZATION
%  ========================================================================

fprintf('========================================\n');
fprintf('TASK 5: Phase Optimization\n');
fprintf('========================================\n\n');

%% Phase Optimization to Maximize SNR
% Test multiple phase values from 0 to pi
% Find phase that maximizes SNR

fprintf('Optimizing carrier phase for maximum SNR...\n');

num_phases = 100;
phi_test = zeros(1, num_phases);
for i = 1:num_phases
    phi_test(i) = (i-1) * pi / (num_phases-1);
end

% Storage for results
amplitude_max = zeros(1, num_phases);
snr_values = zeros(1, num_phases);

fprintf('  Testing %d phase values from 0 to pi...\n', num_phases);

% Test each phase value
for idx = 1:num_phases
    phi_current = phi_test(idx);
    
    % Generate carrier with current phase
    carrier_test = zeros(length(x_bp), 1);
    for i = 1:length(x_bp)
        carrier_test(i) = cos(2*pi*fc*t(i) + phi_current);
    end
    
    % Mix
    x_mixed_test = x_bp .* carrier_test;
    
    % Apply lowpass filter
    x_demod_test = custom_iir_filter(b_iir, a_iir, x_mixed_test);
    
    % Calculate maximum amplitude
    max_amp = 0;
    for i = 1:length(x_demod_test)
        if abs(x_demod_test(i)) > max_amp
            max_amp = abs(x_demod_test(i));
        end
    end
    amplitude_max(idx) = max_amp;
    
    % Calculate SNR for this phase
    sig_pow = 0;
    for i = 1:length(x_demod_test)
        sig_pow = sig_pow + x_demod_test(i)^2;
    end
    sig_pow = sig_pow / length(x_demod_test);
    
    % Estimate noise
    noise_est = zeros(length(x_demod_test), 1);
    for i = 2:length(x_demod_test)
        noise_est(i) = alpha_hp * (noise_est(i-1) + x_demod_test(i) - x_demod_test(i-1));
    end
    
    noise_pow = 0;
    for i = 1:length(noise_est)
        noise_pow = noise_pow + noise_est(i)^2;
    end
    noise_pow = noise_pow / length(noise_est);
    
    % Calculate SNR
    if noise_pow > 0
        snr_values(idx) = 10*log10(sig_pow / noise_pow);
    else
        snr_values(idx) = 0;
    end
    
    % Progress indicator
    if mod(idx, 20) == 0
        fprintf('    Progress: %d/%d phases tested\n', idx, num_phases);
    end
end

fprintf('  Phase optimization complete\n\n');

%% Find Optimal Phase
% Find phase with maximum SNR
max_snr = snr_values(1);
max_snr_idx = 1;
for i = 2:num_phases
    if snr_values(i) > max_snr
        max_snr = snr_values(i);
        max_snr_idx = i;
    end
end

phi_optimal = phi_test(max_snr_idx);

fprintf('Optimization Results:\n');
fprintf('  Optimal phase: %.4f rad (%.2f degrees)\n', phi_optimal, phi_optimal*180/pi);
fprintf('  Maximum SNR achieved: %.2f dB\n', max_snr);
fprintf('  SNR improvement from phi=0: %.2f dB\n\n', max_snr - SNR_demod_initial_dB);

%% Generate Final Demodulated Signal with Optimal Phase
fprintf('Generating final demodulated signal with optimal phase...\n');

% Generate optimal carrier
carrier_optimal = zeros(length(x_bp), 1);
for i = 1:length(x_bp)
    carrier_optimal(i) = cos(2*pi*fc*t(i) + phi_optimal);
end

% Mix and filter
x_mixed_optimal = x_bp .* carrier_optimal;
x_demod_final = custom_iir_filter(b_iir, a_iir, x_mixed_optimal);

fprintf('  Final signal generated\n\n');

%% Calculate Final SNR
fprintf('Final Signal Quality:\n');

% Calculate final signal power
signal_power_final = 0;
for i = 1:length(x_demod_final)
    signal_power_final = signal_power_final + x_demod_final(i)^2;
end
signal_power_final = signal_power_final / length(x_demod_final);

% Estimate final noise
noise_estimate_final = zeros(length(x_demod_final), 1);
for i = 2:length(x_demod_final)
    noise_estimate_final(i) = alpha_hp * (noise_estimate_final(i-1) + x_demod_final(i) - x_demod_final(i-1));
end

noise_power_final = 0;
for i = 1:length(noise_estimate_final)
    noise_power_final = noise_power_final + noise_estimate_final(i)^2;
end
noise_power_final = noise_power_final / length(noise_estimate_final);

% Calculate final SNR
if noise_power_final > 0
    SNR_final_dB = 10*log10(signal_power_final / noise_power_final);
else
    SNR_final_dB = Inf;
end

fprintf('  Final signal power: %.6f\n', signal_power_final);
fprintf('  Final noise power: %.6f\n', noise_power_final);
fprintf('  Final SNR: %.2f dB\n\n', SNR_final_dB);

%% Normalize Audio for Playback
fprintf('Preparing audio for playback...\n');

% Find maximum absolute value manually
max_val = 0;
for i = 1:length(x_demod_final)
    if abs(x_demod_final(i)) > max_val
        max_val = abs(x_demod_final(i));
    end
end

% Normalize to 0.9 to prevent clipping
audio_output = (x_demod_final / max_val) * 0.9;

fprintf('  Audio normalized to range [%.3f, %.3f]\n\n', min(audio_output), max(audio_output));

%% Play and Save Audio
fprintf('Playing demodulated audio...\n');
sound(audio_output, fs);
fprintf('  Playback started\n\n');

% Save to file
output_filename = 'demodulated_message.wav';
audiowrite(output_filename, audio_output, fs);
fprintf('Audio saved to: %s\n\n', output_filename);

%% ========================================================================
%  SNR PERFORMANCE SUMMARY
%  ========================================================================

fprintf('========================================\n');
fprintf('SNR PERFORMANCE SUMMARY\n');
fprintf('========================================\n\n');
fprintf('Processing Stage                SNR (dB)   Improvement (dB)\n');
fprintf('----------------------------------------------------------------\n');
fprintf('1. Input Signal                  %7.2f      baseline\n', SNR_initial_dB);
fprintf('2. After Bandpass Filter         %7.2f      %+7.2f\n', SNR_bp_dB, SNR_bp_dB - SNR_initial_dB);
fprintf('3. After Demodulation (phi=0)    %7.2f      %+7.2f\n', SNR_demod_initial_dB, SNR_demod_initial_dB - SNR_initial_dB);
fprintf('4. After Phase Optimization      %7.2f      %+7.2f\n', SNR_final_dB, SNR_final_dB - SNR_initial_dB);
fprintf('----------------------------------------------------------------\n');
fprintf('Total SNR Improvement:           %7.2f dB\n\n', SNR_final_dB - SNR_initial_dB);

fprintf('========================================\n');
fprintf('Key Results:\n');
fprintf('  Carrier frequency: %.2f kHz\n', fc/1000);
fprintf('  Optimal phase: %.4f rad (%.2f deg)\n', phi_optimal, phi_optimal*180/pi);
fprintf('  Final SNR: %.2f dB\n', SNR_final_dB);
fprintf('========================================\n');
fprintf('DEMODULATION COMPLETE\n');
fprintf('========================================\n');