% Demonstration of FIR filter design by DFT method
% in MATLAB
% J Neasham, Newcastle University, 2020
clc;
close all;
clear all;
fs = 72000; %sampling frequency
Ts = 1/fs; %sample period
fc = 1100; %cut off frequency in Hz
Fc = fc/fs; %normalised cut off frequency
m = 593; %number of taps (N = 2m+1)
N = 2*m+1; % total number of filter taps
for n = 1:m
h(n) = 2*Fc*sin(n*2*pi*Fc)/(n*2*pi*Fc); %calculate truncated impulse response
for LP filter (+ve n)
end
h = [fliplr(h) 2*Fc h]; %construct filter (add n = 0 coefficient for LP and -ve
half)
w = hamming(N)'; %generate N point hamming window
h = h.*w; %apply window to filter coefficients
x = randn(1,10000000)*sqrt(512); %generate white noise signal (normalise for
default 512 point fft in pspectrum)
tic
xf = conv(h,x); %calculate filter output
toc
figure;
pspectrum(xf,fs) % spectrum of output signal (frequency response)
ylabel('Gain (dB)');
%legend('Rectangular window','Hamming window');
title('Filter Frequency Response');
%multirate implementation
fc1 = 2500; %1st stage cut off frequency in Hz
Fc1 = fc1/fs; %1st stage normalised cut off frequency
m1 = 39; %number of taps for first stage filter (N = 2m+1)
N1 = 2*m1+1; %total number of filter taps for 1st stage filter
D1 = 9; %decimation factor for 1st stage
fc2 = 1100; %2nd stage cut off frequency in Hz
Fc2 = fc2/(fs/D1); %2nd stage normalised cut off frequency
m2 = 66; %number of taps for 2nd stage filter (N = 2m+1)
N2 = 2*m2+1; %total number of filter taps for 2nd stage filter
D2 = 2; %decimation factor for 2nd stage
for n = 1:m1
h1(n) = 2*Fc1*sin(n*2*pi*Fc1)/(n*2*pi*Fc1); %calculate truncated impulse
response for LP filter (+ve n)
end
h1 = [fliplr(h1) 2*Fc1 h1]; %construct filter 1 (add n = 0 coefficient for LP
and -ve half)
w = hamming(N1)'; %generate N point hamming window
h1 = h1.*w; %apply window to filter coefficients
for n = 1:m2
h2(n) = 2*Fc2*sin(n*2*pi*Fc2)/(n*2*pi*Fc2); %calculate truncated impulse
response for LP filter (+ve n)
end
h2 = [fliplr(h2) 2*Fc2 h2]; %construct filter 2 (add n = 0 coefficient for LP
and -ve half)
w = hamming(N2)'; %generate N point hanning window
h2 = h2.*w; %apply window to filter coefficients
tic
xf1 = conv(h1,x); %calculate filter 1 output
xd1 = xf1(1:D1:end)*sqrt(D1); %decimate (and compensate for reduced signal
energy)
xf2 = conv(h2,xd1); %calculate filter 2 output
xd2 = xf2(1:D2:end)*sqrt(D2); %decimate (and compensate for reduced signal
energy)
toc
figure;
pspectrum(xf1,fs) % spectrum of output signal (frequency response)
ylabel('Gain (dB)');
title('1st stage filter Frequency Response');
figure;
pspectrum(xd1,fs/D1);
title('1st stage decimated signal spectrum');
figure;
pspectrum(xf2,fs/D1) % spectrum of output signal (frequency
response)
ylabel('Gain (dB)');
title('2nd stage filter Frequency Response');
figure;
pspectrum(xd2,fs/(D1*D2));
title('Final decimated signal spectrum');
