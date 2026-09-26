clc;
clear;
close all;

%% =========================================================
% PIXELFORGE
% GROUND TRUTH VS MODEL PREDICTION
%% =========================================================

disp("======================================================");
disp("PIXELFORGE - GROUND TRUTH COMPARISON");
disp("======================================================");

%% 1. Check Required Files

if ~isfile("DR_ResNet50_Final.mat")

    error("DR_ResNet50_Final.mat not found. Train the model first.");

end

if ~isfile("DR_DataSplit.mat")

    error("DR_DataSplit.mat not found. Saved test split is required.");

end

%% 2. Load Trained Model

disp("Loading trained model...");

modelData = load("DR_ResNet50_Final.mat");

net5 = modelData.net5;

classNames = modelData.classNames;

disp("Model loaded successfully.");

%% 3. Load Held-Out Test Dataset

disp("Loading held-out test dataset...");

splitData = load("DR_DataSplit.mat");

imdsTest = splitData.imdsTest;

numTestImages = numel(imdsTest.Files);

fprintf("Number of held-out test images: %d\n", numTestImages);

%% 4. Select Random Test Image

rng("shuffle");

randomIndex = randi(numTestImages);

imagePath = imdsTest.Files{randomIndex};

groundTruth = string(imdsTest.Labels(randomIndex));

[~, imageName, imageExtension] = fileparts(imagePath);

fprintf("\nSelected Test Image : %s%s\n", imageName, imageExtension);

fprintf("Ground Truth Label  : %s\n", groundTruth);

%% 5. Prepare Image for Prediction

singleImds = imageDatastore(imagePath);

inputSize = [224 224 3];

augSingleImage = augmentedImageDatastore(inputSize, singleImds, "ColorPreprocessing", "gray2rgb");

%% 6. Run Prediction

disp(" ");
disp("Running model prediction...");

predictionStart = tic;

scores = minibatchpredict(net5, augSingleImage, "ExecutionEnvironment", "cpu");

inferenceTime = toc(predictionStart);

%% 7. Convert Prediction Scores

if isa(scores, "dlarray")

    scores = extractdata(scores);

end

if isa(scores, "gpuArray")

    scores = gather(scores);

end

scores = double(scores);

numClasses = numel(classNames);

if size(scores, 2) ~= numClasses && size(scores, 1) == numClasses

    scores = scores';

end

%% 8. Obtain Prediction

[confidence, predictedIndex] = max(scores, [], 2);

predictedClass = string(classNames{predictedIndex});

%% 9. Compare Prediction With Ground Truth

isCorrect = predictedClass == groundTruth;

if isCorrect

    comparisonStatus = "CORRECT PREDICTION";

else

    comparisonStatus = "INCORRECT PREDICTION";

end

%% 10. Display Comparison

fprintf("\n======================================================\n");
fprintf("GROUND TRUTH COMPARISON\n");
fprintf("======================================================\n");

fprintf("Ground Truth       : %s\n", groundTruth);

fprintf("Model Prediction   : %s\n", predictedClass);

fprintf("Confidence         : %.2f %%\n", confidence * 100);

fprintf("Result             : %s\n", comparisonStatus);

fprintf("Inference Time     : %.4f seconds\n", inferenceTime);

fprintf("======================================================\n");

%% 11. Display All Class Probabilities

fprintf("\nCLASS PROBABILITIES\n");
fprintf("------------------------------------------------------\n");

for i = 1:numClasses

    fprintf("%-20s : %.2f %%\n", string(classNames{i}), scores(i) * 100);

end

fprintf("------------------------------------------------------\n");

%% 12. Read Original Image

Ioriginal = imread(imagePath);

%% 13. Display Ground Truth Comparison

figure("Name", "PixelForge - Ground Truth Comparison", "NumberTitle", "off", "Position", [150 100 1100 600]);

comparisonLayout = tiledlayout(1, 2, "TileSpacing", "compact", "Padding", "compact");

%% Original Test Image

nexttile;

imshow(Ioriginal);

title({
    "Held-Out Test Image"
    sprintf("Ground Truth: %s", groundTruth)
}, "FontSize", 12, "FontWeight", "bold");

%% Probability Distribution

nexttile;

classCategories = categorical(string(classNames), string(classNames));

bar(classCategories, scores * 100);

ylim([0 100]);

xlabel("DR Grade");

ylabel("Probability (%)");

title({
    sprintf("Prediction: %s", predictedClass)
    sprintf("Confidence: %.2f%%", confidence * 100)
    comparisonStatus
}, "FontSize", 12, "FontWeight", "bold");

grid on;

title(comparisonLayout, "PixelForge - Held-Out Test Prediction Comparison", "FontSize", 15, "FontWeight", "bold");

%% 14. Determine Review Priority

if predictedClass == "No_DR"

    screeningFlag = "ROUTINE REVIEW";

else

    screeningFlag = "PRIORITY REVIEW";

end

%% 15. Final Summary

disp(" ");

disp("======================================================");
disp("PIXELFORGE VALIDATION SUMMARY");
disp("======================================================");

fprintf("Ground Truth     : %s\n", groundTruth);

fprintf("Prediction       : %s\n", predictedClass);

fprintf("Confidence       : %.2f %%\n", confidence * 100);

fprintf("Comparison       : %s\n", comparisonStatus);

fprintf("Review Priority  : %s\n", screeningFlag);

disp("======================================================");
