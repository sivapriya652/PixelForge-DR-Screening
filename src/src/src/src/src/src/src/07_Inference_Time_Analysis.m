clc;
clear;
close all;

%% =========================================================
% PIXELFORGE
% RESNET-50 INFERENCE TIME ANALYSIS
%% =========================================================

disp("======================================================");
disp("PIXELFORGE - INFERENCE TIME ANALYSIS");
disp("======================================================");

%% 1. Check Required Files

if ~isfile("DR_ResNet50_Final.mat")

    error("DR_ResNet50_Final.mat not found. Train the model first.");

end

if ~isfile("DR_DataSplit.mat")

    error("DR_DataSplit.mat not found. Saved test split is required.");

end

%% 2. Load Trained Model

disp("Loading trained ResNet-50 model...");

modelData = load("DR_ResNet50_Final.mat");

net5 = modelData.net5;

classNames = modelData.classNames;

disp("Model loaded successfully.");

%% 3. Load Held-Out Test Dataset

disp("Loading held-out test dataset...");

splitData = load("DR_DataSplit.mat");

imdsTest = splitData.imdsTest;

totalTestImages = numel(imdsTest.Files);

fprintf("Total held-out test images: %d\n", totalTestImages);

%% 4. Configure Number of Images

requestedImages = 100;

numImages = min(requestedImages, totalTestImages);

fprintf("Images selected for timing analysis: %d\n", numImages);

%% 5. Configure Input Size

inputSize = [224 224 3];

%% =========================================================
% WARM-UP RUN
%% =========================================================

%% 6. Prepare Warm-Up Image

disp(" ");
disp("Performing model warm-up...");

warmupPath = imdsTest.Files{1};

warmupImds = imageDatastore(warmupPath);

warmupAug = augmentedImageDatastore(inputSize, warmupImds, "ColorPreprocessing", "gray2rgb");

%% 7. Warm-Up Prediction

warmupScores = minibatchpredict(net5, warmupAug, "ExecutionEnvironment", "cpu");

clear warmupScores;

disp("Warm-up completed.");

%% =========================================================
% INFERENCE TIME MEASUREMENT
%% =========================================================

%% 8. Initialize Timing Array

inferenceTimes = zeros(numImages, 1);

predictedClasses = strings(numImages, 1);

confidences = zeros(numImages, 1);

%% 9. Run Timing Analysis

disp(" ");
disp("Starting inference time analysis...");

for i = 1:numImages

    imagePath = imdsTest.Files{i};

    singleImds = imageDatastore(imagePath);

    augSingleImage = augmentedImageDatastore(inputSize, singleImds, "ColorPreprocessing", "gray2rgb");

    predictionStart = tic;

    scores = minibatchpredict(net5, augSingleImage, "ExecutionEnvironment", "cpu");

    inferenceTimes(i) = toc(predictionStart);

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

    [confidence, predictedIndex] = max(scores, [], 2);

    predictedClasses(i) = string(classNames{predictedIndex});

    confidences(i) = confidence;

    fprintf("Image %3d/%3d | Time: %.4f sec | Prediction: %-18s | Confidence: %.2f %%\n", i, numImages, inferenceTimes(i), predictedClasses(i), confidences(i) * 100);

end

disp("Inference timing completed.");

%% =========================================================
% CALCULATE TIMING STATISTICS
%% =========================================================

%% 10. Calculate Statistics

averageInferenceTime = mean(inferenceTimes);

medianInferenceTime = median(inferenceTimes);

minimumInferenceTime = min(inferenceTimes);

maximumInferenceTime = max(inferenceTimes);

standardDeviation = std(inferenceTimes);

imagesPerSecond = 1 / averageInferenceTime;

totalInferenceTime = sum(inferenceTimes);

%% 11. Display Timing Results

fprintf("\n======================================================\n");
fprintf("PIXELFORGE INFERENCE TIME RESULTS\n");
fprintf("======================================================\n");

fprintf("Images Tested          : %d\n", numImages);

fprintf("Average Inference Time : %.4f seconds/image\n", averageInferenceTime);

fprintf("Median Inference Time  : %.4f seconds/image\n", medianInferenceTime);

fprintf("Minimum Inference Time : %.4f seconds\n", minimumInferenceTime);

fprintf("Maximum Inference Time : %.4f seconds\n", maximumInferenceTime);

fprintf("Standard Deviation     : %.4f seconds\n", standardDeviation);

fprintf("Approx. Images/Second  : %.2f\n", imagesPerSecond);

fprintf("Total Inference Time   : %.2f seconds\n", totalInferenceTime);

fprintf("Execution Environment  : CPU\n");

fprintf("======================================================\n");

%% =========================================================
% CREATE RESULTS TABLE
%% =========================================================

%% 12. Prepare Ground Truth Labels

groundTruthLabels = string(imdsTest.Labels(1:numImages));

%% 13. Create Detailed Timing Table

imageNumber = (1:numImages)';

timingTable = table(imageNumber, groundTruthLabels, predictedClasses, confidences * 100, inferenceTimes, "VariableNames", {"Image_Number", "Ground_Truth", "Prediction", "Confidence_Percent", "Inference_Time_Seconds"});

%% 14. Display First Results

disp(" ");
disp("Sample Timing Results:");

disp(timingTable(1:min(10, height(timingTable)), :));

%% =========================================================
% VISUALIZATION
%% =========================================================

%% 15. Per-Image Inference Time Graph

figure("Name", "PixelForge - Inference Time Analysis", "NumberTitle", "off");

plot(imageNumber, inferenceTimes, "LineWidth", 1.5);

hold on;

yline(averageInferenceTime, "--", sprintf("Average = %.4f s", averageInferenceTime));

hold off;

xlabel("Test Image Number");

ylabel("Inference Time (seconds)");

title("PixelForge - ResNet-50 CPU Inference Time");

grid on;

%% 16. Inference Time Distribution

figure("Name", "PixelForge - Inference Time Distribution", "NumberTitle", "off");

histogram(inferenceTimes);

xlabel("Inference Time (seconds)");

ylabel("Number of Images");

title("PixelForge - Inference Time Distribution");

grid on;

%% =========================================================
% SAVE RESULTS
%% =========================================================

%% 17. Save CSV Results

writetable(timingTable, "DR_Inference_Times.csv");

%% 18. Save MATLAB Results

save("DR_Inference_Time_Analysis.mat", "inferenceTimes", "averageInferenceTime", "medianInferenceTime", "minimumInferenceTime", "maximumInferenceTime", "standardDeviation", "imagesPerSecond", "totalInferenceTime", "numImages");

%% =========================================================
% FINAL CONFIRMATION
%% =========================================================

disp(" ");

disp("======================================================");
disp("INFERENCE TIME ANALYSIS COMPLETED");
disp("======================================================");

disp("Generated Files:");

disp("1. DR_Inference_Times.csv");

disp("2. DR_Inference_Time_Analysis.mat");

disp(" ");

disp("Generated Figures:");

disp("1. Per-Image Inference Time");

disp("2. Inference Time Distribution");

disp("======================================================");
