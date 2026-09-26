clc;
clear;
close all;

%% =========================================================
% PIXELFORGE
% TOTAL DR SCREENING PIPELINE TIME ANALYSIS
%% =========================================================

disp("======================================================");
disp("PIXELFORGE - TOTAL PIPELINE TIME ANALYSIS");
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

%% 4. Configure Analysis

requestedImages = 20;

numImages = min(requestedImages, totalTestImages);

inputSize = [224 224 3];

fprintf("Images selected for pipeline analysis: %d\n", numImages);

%% 5. Prototype Image Quality Thresholds

minimumFocus = 20;
minimumIllumination = 0.15;
maximumIllumination = 0.90;
minimumContrast = 0.08;

%% 6. Initialize Timing Arrays

qualityCheckTimes = zeros(numImages, 1);
preprocessingTimes = zeros(numImages, 1);
predictionTimes = zeros(numImages, 1);
gradcamTimes = zeros(numImages, 1);
totalPipelineTimes = zeros(numImages, 1);

predictedClasses = strings(numImages, 1);
confidences = zeros(numImages, 1);
qualityStatus = strings(numImages, 1);

%% =========================================================
% MODEL WARM-UP
%% =========================================================

disp(" ");
disp("Performing model warm-up...");

warmupPath = imdsTest.Files{1};

warmupImds = imageDatastore(warmupPath);

warmupAug = augmentedImageDatastore(inputSize, warmupImds, "ColorPreprocessing", "gray2rgb");

warmupScores = minibatchpredict(net5, warmupAug, "ExecutionEnvironment", "cpu");

clear warmupScores;

disp("Warm-up completed.");

%% =========================================================
% TOTAL PIPELINE ANALYSIS
%% =========================================================

disp(" ");
disp("Starting total pipeline timing analysis...");

for i = 1:numImages

    imagePath = imdsTest.Files{i};

    totalStart = tic;

    %% -----------------------------------------------------
    % A. IMAGE LOADING
    %% -----------------------------------------------------

    Ioriginal = imread(imagePath);

    if ndims(Ioriginal) == 2
        Irgb = cat(3, Ioriginal, Ioriginal, Ioriginal);
    else
        Irgb = Ioriginal;
    end

    if size(Irgb, 3) == 1
        Irgb = cat(3, Irgb, Irgb, Irgb);
    end

    if size(Irgb, 3) > 3
        Irgb = Irgb(:, :, 1:3);
    end

    %% -----------------------------------------------------
    % B. IMAGE QUALITY ASSESSMENT
    %% -----------------------------------------------------

    qualityStart = tic;

    Igray = rgb2gray(Irgb);

    IgrayDouble = im2double(Igray);

    laplacianKernel = [0 1 0; 1 -4 1; 0 1 0];

    laplacianImage = imfilter(IgrayDouble, laplacianKernel, "replicate");

    focusScore = var(laplacianImage(:)) * 10000;

    illuminationScore = mean(IgrayDouble(:));

    contrastScore = std(IgrayDouble(:));

    focusPass = focusScore >= minimumFocus;

    illuminationPass = illuminationScore >= minimumIllumination && illuminationScore <= maximumIllumination;

    contrastPass = contrastScore >= minimumContrast;

    qualityPass = focusPass && illuminationPass && contrastPass;

    qualityCheckTimes(i) = toc(qualityStart);

    if qualityPass
        qualityStatus(i) = "ACCEPTABLE";
    else
        qualityStatus(i) = "RECAPTURE";
    end

    %% -----------------------------------------------------
    % C. PREPROCESSING
    %% -----------------------------------------------------

    preprocessingStart = tic;

    singleImds = imageDatastore(imagePath);

    augSingleImage = augmentedImageDatastore(inputSize, singleImds, "ColorPreprocessing", "gray2rgb");

    Iresized = imresize(Irgb, [224 224]);

    Igrad = single(Iresized);

    preprocessingTimes(i) = toc(preprocessingStart);

    %% -----------------------------------------------------
    % D. RESNET-50 PREDICTION
    %% -----------------------------------------------------

    predictionStart = tic;

    scores = minibatchpredict(net5, augSingleImage, "ExecutionEnvironment", "cpu");

    predictionTimes(i) = toc(predictionStart);

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

    %% -----------------------------------------------------
    % E. GRAD-CAM
    %% -----------------------------------------------------

    gradcamStart = tic;

    scoreMap = gradCAM(net5, Igrad, predictedIndex);

    if isa(scoreMap, "dlarray")
        scoreMap = extractdata(scoreMap);
    end

    if isa(scoreMap, "gpuArray")
        scoreMap = gather(scoreMap);
    end

    scoreMap = double(scoreMap);

    scoreMap = squeeze(scoreMap);

    minimumValue = min(scoreMap(:));

    maximumValue = max(scoreMap(:));

    if maximumValue > minimumValue
        scoreMap = (scoreMap - minimumValue) / (maximumValue - minimumValue);
    end

    scoreMap = imresize(scoreMap, [224 224]);

    gradcamTimes(i) = toc(gradcamStart);

    %% -----------------------------------------------------
    % F. TOTAL PIPELINE TIME
    %% -----------------------------------------------------

    totalPipelineTimes(i) = toc(totalStart);

    fprintf("Image %2d/%2d | Quality: %-10s | Prediction: %-18s | Total: %.4f sec\n", i, numImages, qualityStatus(i), predictedClasses(i), totalPipelineTimes(i));

end

disp("Pipeline timing analysis completed.");

%% =========================================================
% CALCULATE AVERAGE COMPONENT TIMES
%% =========================================================

averageQualityTime = mean(qualityCheckTimes);

averagePreprocessingTime = mean(preprocessingTimes);

averagePredictionTime = mean(predictionTimes);

averageGradcamTime = mean(gradcamTimes);

averageTotalTime = mean(totalPipelineTimes);

medianTotalTime = median(totalPipelineTimes);

minimumTotalTime = min(totalPipelineTimes);

maximumTotalTime = max(totalPipelineTimes);

standardDeviationTotal = std(totalPipelineTimes);

%% =========================================================
% DISPLAY RESULTS
%% =========================================================

fprintf("\n======================================================\n");
fprintf("PIXELFORGE TOTAL PIPELINE TIME RESULTS\n");
fprintf("======================================================\n");

fprintf("Images Tested               : %d\n", numImages);

fprintf("\nAVERAGE COMPONENT TIMES\n");

fprintf("Quality Assessment          : %.4f seconds\n", averageQualityTime);

fprintf("Preprocessing               : %.4f seconds\n", averagePreprocessingTime);

fprintf("ResNet-50 Prediction        : %.4f seconds\n", averagePredictionTime);

fprintf("Grad-CAM                    : %.4f seconds\n", averageGradcamTime);

fprintf("\nTOTAL PIPELINE PERFORMANCE\n");

fprintf("Average Total Pipeline Time : %.4f seconds\n", averageTotalTime);

fprintf("Median Total Pipeline Time  : %.4f seconds\n", medianTotalTime);

fprintf("Minimum Total Pipeline Time : %.4f seconds\n", minimumTotalTime);

fprintf("Maximum Total Pipeline Time : %.4f seconds\n", maximumTotalTime);

fprintf("Standard Deviation          : %.4f seconds\n", standardDeviationTotal);

fprintf("Execution Environment       : CPU\n");

fprintf("======================================================\n");

%% =========================================================
% CREATE RESULTS TABLE
%% =========================================================

imageNumber = (1:numImages)';

groundTruthLabels = string(imdsTest.Labels(1:numImages));

pipelineTable = table(imageNumber, groundTruthLabels, qualityStatus, predictedClasses, confidences * 100, qualityCheckTimes, preprocessingTimes, predictionTimes, gradcamTimes, totalPipelineTimes, "VariableNames", {"Image_Number", "Ground_Truth", "Quality_Status", "Prediction", "Confidence_Percent", "Quality_Time", "Preprocessing_Time", "Prediction_Time", "GradCAM_Time", "Total_Pipeline_Time"});

%% =========================================================
% GRAPH 1 - TOTAL TIME PER IMAGE
%% =========================================================

figure("Name", "PixelForge - Total Pipeline Time", "NumberTitle", "off");

plot(imageNumber, totalPipelineTimes, "o-", "LineWidth", 1.5);

hold on;

yline(averageTotalTime, "--", sprintf("Average = %.4f s", averageTotalTime));

hold off;

xlabel("Test Image Number");

ylabel("Total Pipeline Time (seconds)");

title("PixelForge - Total Screening Pipeline Time");

grid on;

%% =========================================================
% GRAPH 2 - COMPONENT TIME COMPARISON
%% =========================================================

figure("Name", "PixelForge - Pipeline Component Times", "NumberTitle", "off");

componentNames = categorical(["Quality Check", "Preprocessing", "Prediction", "Grad-CAM"]);

componentTimes = [averageQualityTime averagePreprocessingTime averagePredictionTime averageGradcamTime];

bar(componentNames, componentTimes);

ylabel("Average Processing Time (seconds)");

title("PixelForge - Average Pipeline Component Time");

grid on;

%% =========================================================
% SAVE RESULTS
%% =========================================================

writetable(pipelineTable, "DR_Total_Pipeline_Times.csv");

save("DR_Total_Pipeline_Time_Analysis.mat", "qualityCheckTimes", "preprocessingTimes", "predictionTimes", "gradcamTimes", "totalPipelineTimes", "averageQualityTime", "averagePreprocessingTime", "averagePredictionTime", "averageGradcamTime", "averageTotalTime", "medianTotalTime", "minimumTotalTime", "maximumTotalTime", "standardDeviationTotal");

%% =========================================================
% FINAL CONFIRMATION
%% =========================================================

disp(" ");

disp("======================================================");
disp("TOTAL PIPELINE TIME ANALYSIS COMPLETED");
disp("======================================================");

disp("Generated Files:");

disp("1. DR_Total_Pipeline_Times.csv");

disp("2. DR_Total_Pipeline_Time_Analysis.mat");

disp(" ");

disp("Generated Figures:");

disp("1. Total Pipeline Time Per Image");

disp("2. Average Pipeline Component Time");

disp("======================================================");
