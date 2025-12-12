function [x_final, phi_optimal, message] = task5_phase_optimisation(x_filtered, params, iir_params, carrier_params)
% TASK5_PHASE_OPTIMISATION - Optimise carrier phase and output audio
%
% Usage:
%   [x_final, phi_optimal, message] = task5_phase_optimisation(x_filtered, params, iir_params, carrier_params)
%
% Inputs:
%   x_filtered    - Bandpass filtered signal from Task 2
%   params        - Parameters structure from Task 1
%   iir_params    - IIR filter parameters from Task 4
%   carrier_params - Carrier parameters from Task 3
%
% Outputs:
%   x_final     - Final demodulated signal with optimal phase
%   phi_optimal - Optimal carrier phase (radians)
%   message     - Identified 3-letter message
%
% Method:
%   1. Coarse grid search (0 to pi, 5-degree steps)
%   2. Fine grid search (around coarse optimum)
%   3. Golden section search for refinement
%   4. Apply optimal phase and output audio
%
% EEE3030 DSP Assignment - Task 5

%% Extract parameters
fs = params.fs;
N = params.N;
t = (0:N-1) / fs;
fc = carrier_params.fc;
b_iir = iir_params.b_iir;
a_iir = iir_params.a_iir;

%% Technique 1: Coarse Grid Search
fprintf('PHASE OPTIMISATION\n\n');
fprintf('Technique 1: Coarse Grid Search\n');
fprintf('--------------------------------\n');

n_coarse = 37;  % 5-degree steps from 0 to 180 degrees
phi_coarse = linspace(0, pi, n_coarse);
rms_coarse = zeros(1, n_coarse);

fprintf('  Testing %d phase values (0 to 180 deg)...\n', n_coarse);

tic;
for i = 1:n_coarse
    carrier_test = cos(2*pi*fc*t + phi_coarse(i));
    x_mixed_test = x_filtered .* carrier_test;
    x_demod_test = custom_iir_filter(b_iir, a_iir, x_mixed_test);
    rms_coarse(i) = sqrt(mean(x_demod_test.^2));
end
time_coarse = toc;

[best_rms_coarse, best_idx_coarse] = max(rms_coarse);
phi_best_coarse = phi_coarse(best_idx_coarse);

fprintf('  Time: %.2f seconds\n', time_coarse);
fprintf('  Best phase: %.2f deg (%.4f rad)\n', phi_best_coarse*180/pi, phi_best_coarse);
fprintf('  Best RMS:   %.6f\n', best_rms_coarse);

%% Technique 2: Fine Grid Search
fprintf('\nTechnique 2: Fine Grid Search\n');
fprintf('------------------------------\n');

phi_range = pi/18;  % +/- 10 degrees
phi_fine_min = max(0, phi_best_coarse - phi_range);
phi_fine_max = min(pi, phi_best_coarse + phi_range);
n_fine = 41;  % 0.5-degree steps
phi_fine = linspace(phi_fine_min, phi_fine_max, n_fine);
rms_fine = zeros(1, n_fine);

fprintf('  Refining search (%.1f to %.1f deg)...\n', phi_fine_min*180/pi, phi_fine_max*180/pi);

tic;
for i = 1:n_fine
    carrier_test = cos(2*pi*fc*t + phi_fine(i));
    x_mixed_test = x_filtered .* carrier_test;
    x_demod_test = custom_iir_filter(b_iir, a_iir, x_mixed_test);
    rms_fine(i) = sqrt(mean(x_demod_test.^2));
end
time_fine = toc;

[best_rms_fine, best_idx_fine] = max(rms_fine);
phi_best_fine = phi_fine(best_idx_fine);

fprintf('  Time: %.2f seconds\n', time_fine);
fprintf('  Best phase: %.4f deg (%.6f rad)\n', phi_best_fine*180/pi, phi_best_fine);
fprintf('  Best RMS:   %.6f\n', best_rms_fine);
fprintf('  Improvement: %.4f%%\n', (best_rms_fine/best_rms_coarse - 1)*100);

%% Technique 3: Golden Section Search
fprintf('\nTechnique 3: Golden Section Search\n');
fprintf('------------------------------------\n');

golden_ratio = (1 + sqrt(5)) / 2;
a_gs = 0;
b_gs = pi;
tol_gs = 1e-6;
max_iter = 50;

c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
d_gs = a_gs + (b_gs - a_gs) / golden_ratio;

% Evaluate at initial points
fc_val = calculate_rms(c_gs, x_filtered, fc, t, b_iir, a_iir);
fd_val = calculate_rms(d_gs, x_filtered, fc, t, b_iir, a_iir);

iter_count = 0;
tic;

while (b_gs - a_gs) > tol_gs && iter_count < max_iter
    iter_count = iter_count + 1;
    
    if fc_val > fd_val
        b_gs = d_gs;
        d_gs = c_gs;
        fd_val = fc_val;
        c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
        fc_val = calculate_rms(c_gs, x_filtered, fc, t, b_iir, a_iir);
    else
        a_gs = c_gs;
        c_gs = d_gs;
        fc_val = fd_val;
        d_gs = a_gs + (b_gs - a_gs) / golden_ratio;
        fd_val = calculate_rms(d_gs, x_filtered, fc, t, b_iir, a_iir);
    end
end

time_golden = toc;
phi_golden = (a_gs + b_gs) / 2;
rms_golden = calculate_rms(phi_golden, x_filtered, fc, t, b_iir, a_iir);

fprintf('  Iterations: %d\n', iter_count);
fprintf('  Time: %.2f seconds\n', time_golden);
fprintf('  Best phase: %.6f deg (%.8f rad)\n', phi_golden*180/pi, phi_golden);
fprintf('  Best RMS:   %.6f\n', rms_golden);

%% Select optimal phase
% Use golden section result as it's most precise
phi_optimal = phi_golden;

fprintf('\nOPTIMAL PHASE SELECTED:\n');
fprintf('  Phase: %.4f degrees (%.6f radians)\n', phi_optimal*180/pi, phi_optimal);

%% Plot optimisation results
figure('Name', 'Task 5 - Phase Optimisation', 'Position', [100, 100, 1200, 600]);

subplot(1,2,1);
plot(phi_coarse*180/pi, rms_coarse, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;
plot(phi_fine*180/pi, rms_fine, 'r.-', 'LineWidth', 1);
plot(phi_optimal*180/pi, rms_golden, 'gp', 'MarkerSize', 15, 'LineWidth', 2);
hold off;
xlabel('Phase (degrees)');
ylabel('RMS Amplitude');
title('Phase Optimisation Results');
legend('Coarse Search', 'Fine Search', 'Optimal', 'Location', 'best');
grid on;
xlim([0, 180]);

subplot(1,2,2);
plot(phi_fine*180/pi, rms_fine, 'r.-', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
plot(phi_optimal*180/pi, rms_golden, 'gp', 'MarkerSize', 15, 'LineWidth', 2);
hold off;
xlabel('Phase (degrees)');
ylabel('RMS Amplitude');
title('Fine Search Detail');
legend('Fine Search', 'Optimal', 'Location', 'best');
grid on;

%% Generate final demodulated signal with optimal phase
fprintf('\nGENERATING FINAL OUTPUT:\n');

carrier_optimal = cos(2*pi*fc*t + phi_optimal);
x_mixed_optimal = x_filtered .* carrier_optimal;
x_final = custom_iir_filter(b_iir, a_iir, x_mixed_optimal);

fprintf('  Final signal RMS: %.6f\n', sqrt(mean(x_final.^2)));
fprintf('  Final signal max: %.6f\n', max(abs(x_final)));

%% Compare with phi = 0
carrier_zero = cos(2*pi*fc*t);
x_mixed_zero = x_filtered .* carrier_zero;
x_demod_zero = custom_iir_filter(b_iir, a_iir, x_mixed_zero);

rms_zero = sqrt(mean(x_demod_zero.^2));
rms_optimal = sqrt(mean(x_final.^2));
amplitude_improvement = rms_optimal / rms_zero;
power_improvement_dB = 20 * log10(amplitude_improvement);

fprintf('\nPHASE OPTIMISATION IMPROVEMENT:\n');
fprintf('  RMS with phi = 0:        %.6f\n', rms_zero);
fprintf('  RMS with phi optimal:    %.6f\n', rms_optimal);
fprintf('  Amplitude improvement:   %.4fx\n', amplitude_improvement);
fprintf('  Power improvement:       %.2f dB\n', power_improvement_dB);

%% Plot comparison
figure('Name', 'Task 5 - Phase Comparison', 'Position', [100, 100, 1200, 600]);

subplot(2,1,1);
plot(t, x_demod_zero, 'b', 'LineWidth', 0.5);
xlabel('Time (seconds)');
ylabel('Amplitude');
title(sprintf('Demodulated Signal with \\phi = 0° (RMS = %.4f)', rms_zero));
grid on;

subplot(2,1,2);
plot(t, x_final, 'r', 'LineWidth', 0.5);
xlabel('Time (seconds)');
ylabel('Amplitude');
title(sprintf('Demodulated Signal with \\phi = %.1f° (RMS = %.4f)', phi_optimal*180/pi, rms_optimal));
grid on;

%% SNR estimation
fprintf('\nSNR ESTIMATION:\n');

% Method: Time-domain envelope analysis
signal_envelope = abs(hilbert(x_final));
envelope_threshold = 0.3 * max(signal_envelope);
signal_samples = signal_envelope > envelope_threshold;
signal_power = mean(x_final(signal_samples).^2);

noise_samples = signal_envelope < 0.1 * max(signal_envelope);
if sum(noise_samples) > 100
    noise_power = mean(x_final(noise_samples).^2);
else
    noise_power = var(x_final) - signal_power * mean(signal_samples);
    noise_power = max(noise_power, eps);
end

snr_estimate = 10 * log10(signal_power / noise_power);
fprintf('  Estimated SNR: %.2f dB\n', snr_estimate);

%% Audio playback
fprintf('\nAUDIO OUTPUT:\n');

% Normalise for playback
x_playback = x_final / max(abs(x_final)) * 0.9;

fprintf('  Playing demodulated audio...\n');
fprintf('  Listen for the 3-letter message!\n');

sound(x_playback, fs);
pause(length(x_playback)/fs + 0.5);

fprintf('  Playback complete.\n');

%% Save audio file
output_filename = 'demodulated_message.wav';
audiowrite(output_filename, x_playback, fs);
fprintf('  Audio saved to: %s\n', output_filename);

%% Message identification
fprintf('\nMESSAGE IDENTIFICATION:\n');
fprintf('  Enter the 3-letter message you heard: ');
message = input('', 's');
fprintf('  Message identified as: %s\n', upper(message));

fprintf('\nTask 5 complete.\n');

end

%% Helper function for RMS calculation
function rms_val = calculate_rms(phi_test, x_filt, fc, t_vec, b_coef, a_coef)
    carrier = cos(2*pi*fc*t_vec + phi_test);
    x_mix = x_filt .* carrier;
    x_dem = custom_iir_filter(b_coef, a_coef, x_mix);
    rms_val = sqrt(mean(x_dem.^2));
end