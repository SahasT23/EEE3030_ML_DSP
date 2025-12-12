%% main.m - DSB-SC AM Signal Demodulator
% EEE3030 DSP Assignment - Main Entry Point
% 
% This script orchestrates the complete demodulation pipeline:
%   Task 1: Signal analysis (time & frequency domain)
%   Task 2: FIR bandpass filter design and application
%   Task 3: Carrier recovery and mixing
%   Task 4: IIR lowpass filter design and application
%   Task 5: Phase optimisation and audio output
%
% Usage: Simply run this script in MATLAB
%
% Required files in same directory:
%   - custom_conv.m
%   - custom_iir_filter.m
%   - task1_signal_analysis.m
%   - task2_fir_filter.m
%   - task3_carrier_recovery.m
%   - task4_iir_filter.m
%   - task5_phase_optimisation.m
%   - Sahas_Talasila.wav (input audio file)

clear all;
close all;
clc;

fprintf('=============================================================\n');
fprintf('       DSB-SC AM SIGNAL DEMODULATOR                          \n');
fprintf('       EEE3030 Digital Signal Processing Assignment          \n');
fprintf('=============================================================\n\n');

%% Configuration
config.filename = 'Sahas_Talasila.wav';
config.fs = 96000;  % Expected sampling frequency

%% Task 1: Signal Analysis
fprintf('=============================================================\n');
fprintf('                    TASK 1: SIGNAL ANALYSIS                  \n');
fprintf('=============================================================\n\n');

[signal, params] = task1_signal_analysis(config.filename);

%% Task 2: FIR Bandpass Filter
fprintf('\n=============================================================\n');
fprintf('                    TASK 2: FIR BANDPASS FILTER              \n');
fprintf('=============================================================\n\n');

[x_filtered, h_bp, fir_params] = task2_fir_filter(signal, params);

%% Task 3: Carrier Recovery and Mixing
fprintf('\n=============================================================\n');
fprintf('                    TASK 3: CARRIER RECOVERY                 \n');
fprintf('=============================================================\n\n');

[x_mixed, carrier_params] = task3_carrier_recovery(x_filtered, params);

%% Task 4: IIR Lowpass Filter
fprintf('\n=============================================================\n');
fprintf('                    TASK 4: IIR LOWPASS FILTER               \n');
fprintf('=============================================================\n\n');

[x_demodulated, iir_params] = task4_iir_filter(x_mixed, params);

%% Task 5: Phase Optimisation and Audio Output
fprintf('\n=============================================================\n');
fprintf('                    TASK 5: PHASE OPTIMISATION               \n');
fprintf('=============================================================\n\n');

[x_final, phi_optimal, message] = task5_phase_optimisation(x_filtered, params, iir_params, carrier_params);

%% Final Summary
fprintf('\n=============================================================\n');
fprintf('                    DEMODULATION COMPLETE                    \n');
fprintf('=============================================================\n\n');

fprintf('SIGNAL PROCESSING CHAIN SUMMARY:\n');
fprintf('---------------------------------\n');
fprintf('  Task 1: Time/Frequency Analysis\n');
fprintf('    - Carrier frequency:     %d Hz\n', carrier_params.fc);
fprintf('    - AM band:               %d - %d Hz\n', params.fmin, params.fmax);
fprintf('\n');
fprintf('  Task 2: FIR Bandpass Filter\n');
fprintf('    - Filter order:          %d taps\n', length(h_bp));
fprintf('    - Passband:              %d - %d Hz\n', params.fmin, params.fmax);
fprintf('    - Window:                Hamming\n');
fprintf('\n');
fprintf('  Task 3: Carrier Recovery and Mixing\n');
fprintf('    - Carrier recovered at:  2f_c = %d Hz\n', 2*carrier_params.fc);
fprintf('    - Mixed to baseband:     0-4 kHz\n');
fprintf('\n');
fprintf('  Task 4: IIR Lowpass Filter\n');
fprintf('    - Filter type:           4th order Butterworth\n');
fprintf('    - Cutoff frequency:      %d Hz\n', iir_params.fc_lowpass);
fprintf('\n');
fprintf('  Task 5: Phase Optimisation and Audio Output\n');
fprintf('    - Optimal phase:         %.2f degrees\n', phi_optimal*180/pi);
fprintf('    - Message identified:    %s\n', upper(message));
fprintf('\n');
fprintf('=============================================================\n');