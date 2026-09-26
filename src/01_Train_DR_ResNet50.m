clc;
clear;
close all;

%% =========================================================
% PIXELFORGE
% RESNET-50 TRAINING FOR DIABETIC RETINOPATHY CLASSIFICATION
%% =========================================================

rng(42);

disp("======================================================");
disp("PIXELFORGE - DR MODEL TRAINING");
disp("======================================================");

%% 1. Dataset Configuration
%
% Expected folder structure:
%
% dataset/
%   gaussian_filtered_images/
%       Mild/
%       Moderate/
%       No_DR/
%       Proliferate_DR/
%       Severe/
%

datasetPath = fullfile("dataset", "gaussian_filtered_images");

%% 2. Check Dataset

if ~isfolder(datasetPath)

    error("Dataset folder not found. Place the dataset inside dataset/gaussian_filtered_images.");

end

%% 3. Load Dataset

disp("Loading retinal fundus dataset...");

imds = imageDatastore(datasetPath, "IncludeSubfolders", true, "LabelSource", "foldernames");

disp("Dataset loaded successfully.");

%% 4. Display Dataset Information

disp(" ");
disp("Dataset Distribution:");

disp(countEachLabel(imds));

fprintf("Total Images: %d\n", numel(imds.Files));

%% 5. Detect DR Classes

classNames = categories(imds.Labels);

numClasses = numel(classNames);

disp(" ");
disp("Detected DR Classes:");

disp(classNames);

fprintf("Number of Classes: %d\n", numClasses);

%% 6. Verify Number of Classes

if numClasses ~= 5

    error("Expected 5 DR classes, but MATLAB detected %d classes.", numClasses);

end

%% 7. Split Dataset
%
% Training   : 70 percent
% Validation : 15 percent
% Testing    : 15 percent
%

disp(" ");
disp("Creating training, validation and test splits...");

[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, 0.70, 0.15, 0.15, "randomized");

%% 8. Display Dataset Split

disp(" ");
disp("Training Dataset:");

disp(countEachLabel(imdsTrain));

disp("Validation Dataset:");

disp(countEachLabel(imdsValidation));

disp("Testing Dataset:");

disp(countEachLabel(imdsTest));

fprintf("Training Images   : %d\n", numel(imdsTrain.Files));

fprintf("Validation Images : %d\n", numel(imdsValidation.Files));

fprintf("Testing Images    : %d\n", numel(imdsTest.Files));

%% 9. Save Dataset Split

save("DR_DataSplit.mat", "imdsTrain", "imdsValidation", "imdsTest", "classNames", "-v7.3");

disp(" ");
disp("Dataset split saved as DR_DataSplit.mat");

%% 10. Configure Input Size

inputSize = [224 224 3];

%% 11. Configure Training Augmentation

augmenter = imageDataAugmenter("RandRotation", [-10 10], "RandXReflection", true, "RandXTranslation", [-5 5], "RandYTranslation", [-5 5]);

%% 12. Prepare Training Dataset

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, "DataAugmentation", augmenter, "ColorPreprocessing", "gray2rgb");

%% 13. Prepare Validation Dataset

augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation, "ColorPreprocessing", "gray2rgb");

%% 14. Prepare Test Dataset

augimdsTest = augmentedImageDatastore(inputSize, imdsTest, "ColorPreprocessing", "gray2rgb");

disp("Image preprocessing and augmentation configured successfully.");

%% 15. Load Pretrained ResNet-50

disp(" ");
disp("Loading pretrained ResNet-50...");

net5 = imagePretrainedNetwork("resnet50", NumClasses=5);

disp("ResNet-50 loaded successfully.");

%% 16. Configure Training Options

options = trainingOptions("adam", "MiniBatchSize", 32, "MaxEpochs", 20, "InitialLearnRate", 1e-4, "Shuffle", "every-epoch", "ValidationData", augimdsValidation, "ValidationFrequency", 20, "Verbose", true, "Plots", "training-progress", "ExecutionEnvironment", "cpu");

%% 17. Display Training Configuration

disp(" ");
disp("======================================================");
disp("PIXELFORGE TRAINING CONFIGURATION");
disp("======================================================");

fprintf("Model             : ResNet-50\n");
fprintf("Number of Classes : %d\n", numClasses);
fprintf("Input Size        : 224 x 224 x 3\n");
fprintf("Epochs            : 20\n");
fprintf("Mini Batch Size   : 32\n");
fprintf("Learning Rate     : 0.0001\n");
fprintf("Optimizer         : Adam\n");
fprintf("Loss Function     : Cross-Entropy\n");
fprintf("Execution         : CPU\n");

disp("======================================================");

%% 18. Start Model Training

disp(" ");
disp("Starting ResNet-50 training...");

trainingStart = tic;

[net5, info] = trainnet(augimdsTrain, net5, "crossentropy", options);

trainingTime = toc(trainingStart);

disp("Training completed successfully.");

%% 19. Calculate Training Time

trainingTimeSeconds = trainingTime;

trainingTimeMinutes = trainingTime / 60;

trainingTimeHours = trainingTime / 3600;

fprintf("\n======================================================\n");
fprintf("TRAINING TIME\n");
fprintf("======================================================\n");

fprintf("Seconds : %.2f\n", trainingTimeSeconds);
fprintf("Minutes : %.2f\n", trainingTimeMinutes);
fprintf("Hours   : %.2f\n", trainingTimeHours);

fprintf("======================================================\n");

%% 20. Save Trained Model

disp("Saving trained model...");

save("DR_ResNet50_Final.mat", "net5", "classNames", "info", "trainingTimeSeconds", "trainingTimeMinutes", "trainingTimeHours", "-v7.3");

%% 21. Verify Saved Model

if ~isfile("DR_ResNet50_Final.mat")

    error("Trained model could not be saved.");

end

savedVariables = whos("-file", "DR_ResNet50_Final.mat");

savedNames = string({savedVariables.name});

if ~any(savedNames == "net5")

    error("net5 is missing from the saved model.");

end

if ~any(savedNames == "classNames")

    error("classNames is missing from the saved model.");

end

%% 22. Final Confirmation

disp(" ");
disp("======================================================");
disp("PIXELFORGE MODEL TRAINING COMPLETED");
disp("======================================================");

disp("Generated Files:");

disp("1. DR_DataSplit.mat");
disp("2. DR_ResNet50_Final.mat");

disp(" ");
disp("Next Step: Model Evaluation");

disp("======================================================");
