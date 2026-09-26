clc;
clear;
close all;

%% =========================================================
% PIXELFORGE
% COMPLETE AI-ASSISTED DIABETIC RETINOPATHY SCREENING DEMO
%% =========================================================

disp("======================================================");
disp("PIXELFORGE");
disp("AI-ASSISTED DIABETIC RETINOPATHY SCREENING");
disp("======================================================");

%% 1. Check Trained Model

if ~isfile("DR_ResNet50_Final.mat")
    error("DR_ResNet50_Final.mat not found. Train the model first.");
end

%% 2. Load Model

disp("Loading trained ResNet-50 model...");

modelData = load("DR_ResNet50_Final.mat");

net5 = modelData.net5;
classNames = modelData.classNames;

disp("Model loaded successfully.");

%% 3. Select Fundus Image

disp(" ");
disp("Select a fundus image for screening...");

[file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff', 'Fundus Images'}, 'PixelForge - Select Fundus Image');

if isequal(file, 0)
    disp("No image selected.");
    return;
end

imagePath = fullfile(path, file);

fprintf("Selected Image : %s\n", file);

%% 4. Generate Demo Sample ID

sampleNumber = randi([10000 99999]);

sampleID = sprintf("PF-%05d", sampleNumber);

screeningDate = string(datetime("now", "Format", "dd-MMM-yyyy HH:mm:ss"));

fprintf("Sample ID      : %s\n", sampleID);
fprintf("Screening Time : %s\n", screeningDate);

%% 5. Start Total Pipeline Timer

totalPipelineStart = tic;

%% =========================================================
% IMAGE LOADING
%% =========================================================

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

%% =========================================================
% STAGE 1 - IMAGE QUALITY ASSESSMENT
%% =========================================================

disp(" ");
disp("STAGE 1: IMAGE QUALITY ASSESSMENT");

qualityStart = tic;

Igray = rgb2gray(Irgb);

IgrayDouble = im2double(Igray);

laplacianKernel = [0 1 0; 1 -4 1; 0 1 0];

laplacianImage = imfilter(IgrayDouble, laplacianKernel, "replicate");

focusScore = var(laplacianImage(:)) * 10000;

illuminationScore = mean(IgrayDouble(:));

contrastScore = std(IgrayDouble(:));

minimumFocus = 20;
minimumIllumination = 0.15;
maximumIllumination = 0.90;
minimumContrast = 0.08;

focusPass = focusScore >= minimumFocus;

illuminationPass = illuminationScore >= minimumIllumination && illuminationScore <= maximumIllumination;

contrastPass = contrastScore >= minimumContrast;

qualityPass = focusPass && illuminationPass && contrastPass;

qualityCheckTime = toc(qualityStart);

fprintf("\nFocus Score  : %.2f\n", focusScore);
fprintf("Illumination : %.3f\n", illuminationScore);
fprintf("Contrast     : %.3f\n", contrastScore);

%% 6. Stop Pipeline if Image Quality Fails

if ~qualityPass

    recaptureReason = "";

    if ~focusPass
        recaptureReason = recaptureReason + "Poor focus / blur. ";
    end

    if illuminationScore < minimumIllumination
        recaptureReason = recaptureReason + "Image too dark. ";
    end

    if illuminationScore > maximumIllumination
        recaptureReason = recaptureReason + "Image too bright. ";
    end

    if ~contrastPass
        recaptureReason = recaptureReason + "Low contrast. ";
    end

    totalPipelineTime = toc(totalPipelineStart);

    fprintf("\n======================================================\n");
    fprintf("IMAGE QUALITY: RECAPTURE REQUIRED\n");
    fprintf("======================================================\n");

    fprintf("Reason             : %s\n", recaptureReason);
    fprintf("Quality Check Time : %.4f seconds\n", qualityCheckTime);
    fprintf("Total Time         : %.4f seconds\n", totalPipelineTime);

    fprintf("------------------------------------------------------\n");
    fprintf("DR classification was NOT performed.\n");
    fprintf("Please recapture the fundus image.\n");
    fprintf("======================================================\n");

    figure("Name", "PixelForge - Recapture Required", "NumberTitle", "off", "Position", [250 120 900 650]);

    imshow(Irgb);

    title({
        "PIXELFORGE"
        "IMAGE QUALITY: RECAPTURE REQUIRED"
        char(recaptureReason)
        sprintf("Focus: %.2f | Illumination: %.3f | Contrast: %.3f", focusScore, illuminationScore, contrastScore)
    }, "FontSize", 12, "FontWeight", "bold");

    return;
end

disp("IMAGE QUALITY: ACCEPTABLE");
disp("Proceeding to DR screening...");

%% =========================================================
% STAGE 2 - PREPROCESSING
%% =========================================================

disp(" ");
disp("STAGE 2: IMAGE PREPARATION");

preprocessingStart = tic;

inputSize = [224 224 3];

singleImds = imageDatastore(imagePath);

augSingleImage = augmentedImageDatastore(inputSize, singleImds, "ColorPreprocessing", "gray2rgb");

Iresized = imresize(Irgb, [224 224]);

Igrad = single(Iresized);

preprocessingTime = toc(preprocessingStart);

fprintf("Preprocessing Time : %.4f seconds\n", preprocessingTime);

%% =========================================================
% STAGE 3 - RESNET-50 DR PREDICTION
%% =========================================================

disp(" ");
disp("STAGE 3: RESNET-50 DR SCREENING");

predictionStart = tic;

scores = minibatchpredict(net5, augSingleImage, "ExecutionEnvironment", "cpu");

predictionTime = toc(predictionStart);

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

predictedClass = string(classNames{predictedIndex});

fprintf("Predicted DR Grade : %s\n", predictedClass);
fprintf("Confidence         : %.2f %%\n", confidence * 100);
fprintf("Prediction Time    : %.4f seconds\n", predictionTime);

%% =========================================================
% STAGE 4 - CONFIDENCE DISPLAY CATEGORY
%% =========================================================

if confidence >= 0.80
    confidenceLevel = "HIGH";
elseif confidence >= 0.60
    confidenceLevel = "MODERATE";
else
    confidenceLevel = "LOW";
end

%% =========================================================
% STAGE 5 - REVIEW PRIORITY
%% =========================================================

if predictedClass == "No_DR"
    reviewPriority = "ROUTINE REVIEW";
else
    reviewPriority = "PRIORITY REVIEW";
end

%% =========================================================
% STAGE 6 - GRAD-CAM EXPLAINABILITY
%% =========================================================

disp(" ");
disp("STAGE 4: GENERATING GRAD-CAM");

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

gradcamTime = toc(gradcamStart);

fprintf("Grad-CAM Time : %.4f seconds\n", gradcamTime);

%% =========================================================
% STAGE 7 - TOTAL PIPELINE TIME
%% =========================================================

totalPipelineTime = toc(totalPipelineStart);

%% =========================================================
% COMMAND WINDOW REPORT
%% =========================================================

fprintf("\n======================================================\n");
fprintf("PIXELFORGE - FINAL SCREENING REPORT\n");
fprintf("======================================================\n");

fprintf("Sample ID          : %s\n", sampleID);
fprintf("Date / Time        : %s\n", screeningDate);
fprintf("Image Quality      : ACCEPTABLE\n");

fprintf("\nAI SCREENING RESULT\n");
fprintf("------------------------------------------------------\n");

fprintf("Predicted DR Grade : %s\n", predictedClass);
fprintf("Confidence         : %.2f %%\n", confidence * 100);
fprintf("Confidence Level   : %s\n", confidenceLevel);
fprintf("Review Priority    : %s\n", reviewPriority);

fprintf("\nCLASS PROBABILITIES\n");
fprintf("------------------------------------------------------\n");

for i = 1:numClasses
    fprintf("%-20s : %.2f %%\n", string(classNames{i}), scores(i) * 100);
end

fprintf("\nIMAGE QUALITY METRICS\n");
fprintf("------------------------------------------------------\n");

fprintf("Focus Score        : %.2f\n", focusScore);
fprintf("Illumination       : %.3f\n", illuminationScore);
fprintf("Contrast           : %.3f\n", contrastScore);

fprintf("\nPROCESSING TIME\n");
fprintf("------------------------------------------------------\n");

fprintf("Quality Check      : %.4f seconds\n", qualityCheckTime);
fprintf("Preprocessing      : %.4f seconds\n", preprocessingTime);
fprintf("Prediction         : %.4f seconds\n", predictionTime);
fprintf("Grad-CAM           : %.4f seconds\n", gradcamTime);
fprintf("Total Pipeline     : %.4f seconds\n", totalPipelineTime);

fprintf("\nMODEL INFORMATION\n");
fprintf("------------------------------------------------------\n");

fprintf("Architecture       : ResNet-50\n");
fprintf("Input Size         : 224 x 224 x 3\n");
fprintf("Classes            : 5\n");
fprintf("Execution          : Local CPU\n");
fprintf("Explainability     : Grad-CAM\n");

fprintf("\n======================================================\n");
fprintf("AI screening support - clinician review required.\n");
fprintf("======================================================\n");

%% =========================================================
% FINAL DOCTOR-FACING DASHBOARD
%% =========================================================

figure("Name", "PixelForge - Doctor Screening Dashboard", "NumberTitle", "off", "Position", [50 50 1450 820]);

dashboard = tiledlayout(2, 3, "TileSpacing", "compact", "Padding", "compact");

%% TILE 1 - ORIGINAL FUNDUS IMAGE

nexttile;

imshow(Irgb);

title({
    "Fundus Image"
    sprintf("Sample ID: %s", sampleID)
}, "FontSize", 11, "FontWeight", "bold");

%% TILE 2 - GRAD-CAM

nexttile;

imshow(Iresized);

hold on;

heatmapHandle = imagesc(scoreMap);

set(heatmapHandle, "AlphaData", 0.45);

colormap jet;

colorbar;

axis image;
axis off;

title("Grad-CAM Explanation", "FontSize", 11, "FontWeight", "bold");

hold off;

%% TILE 3 - AI SCREENING RESULT

nexttile;

axis off;

resultText = {
    "AI SCREENING RESULT"
    ""
    sprintf("DR Grade: %s", predictedClass)
    sprintf("Confidence: %.2f%%", confidence * 100)
    sprintf("Confidence Level: %s", confidenceLevel)
    ""
    sprintf("Review Priority:")
    sprintf("%s", reviewPriority)
    ""
    "Clinician review required"
};

text(0.05, 0.95, resultText, "Units", "normalized", "VerticalAlignment", "top", "FontSize", 12, "FontWeight", "bold");

%% TILES 4 AND 5 - CLASS PROBABILITIES

nexttile([1 2]);

classCategories = categorical(string(classNames), string(classNames));

bar(classCategories, scores * 100);

ylim([0 100]);

xlabel("DR Grade");

ylabel("Probability (%)");

title("Five-Class DR Probability Distribution", "FontSize", 11, "FontWeight", "bold");

grid on;

%% TILE 6 - TECHNICAL INFORMATION

nexttile;

axis off;

technicalText = {
    "SCREENING INFORMATION"
    ""
    "Image Quality: ACCEPTABLE"
    sprintf("Focus: %.2f", focusScore)
    sprintf("Illumination: %.3f", illuminationScore)
    sprintf("Contrast: %.3f", contrastScore)
    ""
    "Model: ResNet-50"
    "Execution: Local CPU"
    "Explainability: Grad-CAM"
    ""
    sprintf("Prediction Time: %.3f s", predictionTime)
    sprintf("Total Pipeline: %.3f s", totalPipelineTime)
};

text(0.05, 0.95, technicalText, "Units", "normalized", "VerticalAlignment", "top", "FontSize", 10.5);

title(dashboard, "PIXELFORGE - AI-ASSISTED DIABETIC RETINOPATHY SCREENING", "FontSize", 16, "FontWeight", "bold");

%% =========================================================
% PROCESSING TIME GRAPH
%% =========================================================

figure("Name", "PixelForge - Processing Time", "NumberTitle", "off");

processingStages = categorical(["Quality Check", "Preprocessing", "Prediction", "Grad-CAM", "Total Pipeline"]);

processingTimes = [qualityCheckTime preprocessingTime predictionTime gradcamTime totalPipelineTime];

bar(processingStages, processingTimes);

ylabel("Processing Time (seconds)");

title("PixelForge - Screening Pipeline Processing Time");

grid on;

%% =========================================================
% FINAL MESSAGE
%% =========================================================

disp(" ");
disp("======================================================");
disp("PIXELFORGE COMPLETE SCREENING DEMO FINISHED");
disp("======================================================");

fprintf("Sample ID       : %s\n", sampleID);
fprintf("DR Grade        : %s\n", predictedClass);
fprintf("Confidence      : %.2f %%\n", confidence * 100);
fprintf("Review Priority : %s\n", reviewPriority);
fprintf("Total Time      : %.4f seconds\n", totalPipelineTime);

disp(" ");
disp("Grad-CAM shows regions that influenced the model prediction.");
disp("It is not a lesion segmentation map.");
disp("Final interpretation requires clinician review.");

disp("======================================================");
