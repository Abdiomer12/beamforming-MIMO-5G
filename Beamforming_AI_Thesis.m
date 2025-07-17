%% Part 1: Training Data Generation
% This section generates simulated data that will be used to train the neural network.
% This data represents channel state information (CSI) and optimal beamforming weights.

clear; clc; close all; % Clear workspace, command window, and close all figures

% System Parameters
numAntennas = 64;   % Number of antennas at the Base Station - Massive MIMO setup
numUsers = 4;       % Number of simultaneous users
snr_db = 20;        % Signal-to-Noise Ratio (dB)
snr_linear = 10^(snr_db/10); % SNR in linear scale

numSamples = 10000; % Number of data samples to generate

% Pre-allocate memory for storing data
H_data = zeros(numUsers, numAntennas, numSamples); % Channel data (Input for NN)
W_optimal_data = zeros(numAntennas, numUsers, numSamples); % Desired beamforming weights (Output/Target for NN)

fprintf('Generating training data...\n');
for i = 1:numSamples
    % Channel Generation - Rayleigh fading model
    % H is the channel matrix
    H = (randn(numUsers, numAntennas) + 1j*randn(numUsers, numAntennas)) / sqrt(2);
    H_data(:,:,i) = H; % Store the channel data

    % Calculate Optimal Beamforming Weights
    % Using Zero-Forcing (ZF) beamforming as an example
    % Utilize the pseudo-inverse for weight calculation.
    W_zf = H' / (H * H'); % Initial ZF weights
    
    % Normalization (Power constraint)
    % Ensure the transmit power is constrained
    for u = 1:numUsers
        W_zf(:,u) = W_zf(:,u) / norm(W_zf(:,u)); % Normalize each column (each user's beamformer)
    end
    
    % For MMSE (Minimum Mean Square Error), you would replace the ZF calculation with:
    % W_mmse = inv(H'*H + (numUsers/snr_linear)*eye(numAntennas)) * H';
    
    W_optimal_data(:,:,i) = W_zf; % Store the desired optimal beamforming weights
end
fprintf('Data generation complete.\n\n');

%% Part 2: Data Preparation for Neural Network
% MATLAB's neural network toolbox typically requires data to be in a specific format (vector).
% Here, we unroll the 2D matrices into 1D vectors and separate real and imaginary parts.

% Unroll matrices into vectors. MATLAB works in column-major order for reshape.
% H_reshaped: (numUsers * numAntennas * 2) x numSamples (Real and Imaginary parts)
% W_reshaped: (numAntennas * numUsers * 2) x numSamples (Real and Imaginary parts)

% Input Data for the Neural Network
H_real_vec = reshape(real(H_data), numUsers * numAntennas, numSamples);
H_imag_vec = reshape(imag(H_data), numUsers * numAntennas, numSamples);
X = [H_real_vec; H_imag_vec]; % Concatenate real and imaginary parts to form the input vector

% Output Data (Target) for the Neural Network
W_real_vec = reshape(real(W_optimal_data), numAntennas * numUsers, numSamples);
W_imag_vec = reshape(imag(W_optimal_data), numAntennas * numUsers, numSamples);
Y = [W_real_vec; W_imag_vec]; % Concatenate real and imaginary parts to form the target output vector

%% Part 3: Neural Network Construction and Training
% We will use MATLAB's 'Deep Learning Toolbox'.
% Ensure you have this toolbox installed.

% Feedforward Neural Network for regression task
% We've chosen a simple network with one hidden layer.
% You can experiment with more layers or more neurons to see the impact.

hiddenLayerSize = 256; % Number of neurons in the hidden layer. This is a tunable parameter.

net = fitnet(hiddenLayerSize); % Create a feedforward neural network
net.trainFcn = 'trainscg'; % Training algorithm (Scaled Conjugate Gradient)
net.divideParam.trainRatio = 0.7; % 70% of data for training
net.divideParam.valRatio = 0.15; % 15% of data for validation
net.divideParam.testRatio = 0.15; % 15% of data for testing

fprintf('Training the neural network, please wait...\n');
[net, tr] = train(net, X, Y); % Start the network training process

fprintf('Training complete.\n\n');

%% Part 4: Evaluation and Visualization of Results
% This section evaluates the performance of the trained neural network.

% Use the trained network to predict beamforming weights for the test data.
Y_pred_vec = net(X(:, tr.testInd)); % Predictions for the test set inputs only

Y_target_vec = Y(:, tr.testInd);    % Actual (target) weights for the test data

% Reshape the predicted and target results back into complex matrices for comparison
numTestSamples = length(tr.testInd);
W_pred_real = reshape(Y_pred_vec(1:numAntennas*numUsers, :), numAntennas, numUsers, numTestSamples);
W_pred_imag = reshape(Y_pred_vec(numAntennas*numUsers+1:end, :), numAntennas, numUsers, numTestSamples);
W_predicted = W_pred_real + 1j*W_pred_imag;

W_target_real = reshape(Y_target_vec(1:numAntennas*numUsers, :), numAntennas, numUsers, numTestSamples);
W_target_imag = reshape(Y_target_vec(numAntennas*numUsers+1:end, :), numAntennas, numUsers, numTestSamples);
W_target = W_target_real + 1j*W_target_imag;

% Calculate Error (Mean Squared Error - MSE)
% We are looking at the difference between the predicted and actual weights.
mse_beamformer = sum(abs(W_predicted(:) - W_target(:)).^2) / numel(W_predicted);
fprintf('Mean Squared Error (MSE) for Beamformer Weights: %.4f\n', mse_beamformer);

% Visualize some results
% We will plot one example of predicted vs. actual weights.

sampleIndex = 1; % Choose a sample to display

figure;
subplot(2,2,1);
stem(abs(W_target(:,1,sampleIndex)));
title(sprintf('Target Weight (Magnitude) for User 1 - Sample %d', sampleIndex));
xlabel('Antenna Index'); ylabel('Magnitude');
grid on;

subplot(2,2,2);
stem(abs(W_predicted(:,1,sampleIndex)));
title(sprintf('Predicted Weight (Magnitude) for User 1 - Sample %d', sampleIndex));
xlabel('Antenna Index'); ylabel('Magnitude');
grid on;

subplot(2,2,3);
plot(angle(W_target(:,1,sampleIndex)));
title(sprintf('Target Weight (Phase) for User 1 - Sample %d', sampleIndex));
xlabel('Antenna Index'); ylabel('Phase (Radians)');
grid on;

subplot(2,2,4);
plot(angle(W_predicted(:,1,sampleIndex)));
title(sprintf('Predicted Weight (Phase) for User 1 - Sample %d', sampleIndex));
xlabel('Antenna Index'); ylabel('Phase (Radians)');
grid on;

sgtitle('Comparison of Beamforming Weights: Target vs. Predicted');

% Training performance plots from the Neural Network Toolbox
figure, plotperform(tr); % Displays the performance curve during training
figure, plottrainstate(tr); % Displays training state, e.g., gradient, validation checks

% === Custom Histogram for Prediction Errors (Robust solution) ===
% Directly calculate the errors for the test set
test_outputs = net(X(:, tr.testInd));
test_targets = Y(:, tr.testInd);
errors_complex = test_targets - test_outputs; % Calculate the complex errors

% Display histogram of the magnitude of errors
figure;
histogram(abs(errors_complex(:)), 'Normalization', 'probability', 'BinWidth', 0.01);
title('Histogram of Prediction Error Magnitudes');
xlabel('Error Magnitude');
ylabel('Probability');
grid on;
% ======================================================

fprintf('\nCode execution complete.\n');