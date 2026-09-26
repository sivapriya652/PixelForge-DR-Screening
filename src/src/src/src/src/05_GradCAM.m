clc;
clear;
close all;

%% =========================================================
% PIXELFORGE
% GRAD-CAM EXPLAINABILITY FOR DR SCREENING
%% =========================================================

disp("======================================================");
disp("PIXELFORGE - GRAD-CAM EXPLAINABILITY");
disp("======================================================");

%% 1. Check Trained Model

if ~isfile("DR_ResNet50_Final.mat")

    error("DR_ResNet50_Final.mat not found. Train the model first.");

end

%% 2. Load Trained Model

disp("Loading trained ResNet-50 model...");

modelData = load("DR_ResNet50_Final.mat");

net5 = modelData.net5;

classNames = modelData.classNames;

disp("Model loaded successfully.");

fprintf("Network Type      : %s\n", class(net5));

fprintf("Number of Classes : %d\n", numel(classNames));

%% 3. Select Fundus Image

disp(" ");
disp("Select a fundus image...");

[file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff', 'Fundus Images'}, 'PixelForge - Select Fundus Image');

if isequal(file, 0)

    disp("No image selected.");
    return;

end

imagePath = fullfile(path, file);

fprintf("Selected Image : %s\n", file);

%% =========================================================
% DR PREDICTION
%% =========================================================

%% 4. Create Single Image Datastore

singleImds = imageDatastore(imagePath);

inputSize = [224 224 3];

augSingleImage = augmentedImageDatastore(inputSize, singleImds, "ColorPreprocessing", "gray2rgb");

%% 5. Run Prediction

disp(" ");
disp("Running DR prediction...");

predictionStart = tic;

scores = minibatchpredict(net5, augSingleImage, "ExecutionEnvironment", "cpu");

inferenceTime = toc(predictionStart);

disp("Prediction completed.");

%% 6. Convert Prediction Scores

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

%% 7. Obtain Predicted Class

[confidence, predictedIndex] = max(scores, [], 2);

predictedClass = string(classNames{predictedIndex});

fprintf("\nPredicted DR Grade : %s\n", predictedClass);

fprintf("Confidence         : %.2f %%\n", confidence * 100);

fprintf("Inference Time     : %.4f seconds\n", inferenceTime);

%% =========================================================
% PREPARE IMAGE FOR GRAD-CAM
%% =========================================================

%% 8. Read Original Image

Ioriginal = imread(imagePath);

%% 9. Ensure RGB Image

if ndims(Ioriginal) == 2

    Ioriginal = cat(3, Ioriginal, Ioriginal, Ioriginal);

end

if size(Ioriginal, 3) == 1

    Ioriginal = cat(3, Ioriginal, Ioriginal, Ioriginal);

end

if size(Ioriginal, 3) > 3

    Ioriginal = Ioriginal(:, :, 1:3);

end

%% 10. Resize Image

I = imresize(Ioriginal, [224 224]);

%% 11. Convert Image to Single Precision

Igrad = single(I);

%% =========================================================
% GENERATE GRAD-CAM
%% =========================================================

%% 12. Generate Grad-CAM Heatmap

disp(" ");
disp("Generating Grad-CAM...");

gradcamStart = tic;

scoreMap = gradCAM(net5, Igrad, predictedIndex);

gradcamTime = toc(gradcamStart);

disp("Grad-CAM generated successfully.");

%% 13. Convert Grad-CAM Output

if isa(scoreMap, "dlarray")

    scoreMap = extractdata(scoreMap);

end

if isa(scoreMap, "gpuArray")

    scoreMap = gather(scoreMap);

end

scoreMap = double(scoreMap);

scoreMap = squeeze(scoreMap);

%% 14. Normalize Heatmap

minimumValue = min(scoreMap(:));

maximumValue = max(scoreMap(:));

if maximumValue > minimumValue

    scoreMap = (scoreMap - minimumValue) / (maximumValue - minimumValue);

end

%% 15. Resize Heatmap

scoreMap = imresize(scoreMap, [224 224]);

%% =========================================================
% DISPLAY GRAD-CAM RESULT
%% =========================================================

%% 16. Create Explainability Figure

figure("Name", "PixelForge - Grad-CAM Explainability", "NumberTitle", "off", "Position", [150 150 1100 500]);

resultLayout = tiledlayout(1, 2, "TileSpacing", "compact", "Padding", "compact");

%% Original Fundus Image

nexttile;

imshow(I);

title("Original Fundus Image", "FontSize", 12, "FontWeight", "bold");

%% Grad-CAM Overlay

nexttile;

imshow(I);

hold on;

heatmapHandle = imagesc(scoreMap);

set(heatmapHandle, "AlphaData", 0.45);

colormap jet;

colorbar;

axis image;

axis off;

title({
    "Grad-CAM Explanation"
    sprintf("Prediction: %s | Confidence: %.2f%%", predictedClass, confidence * 100)
}, "FontSize", 12, "FontWeight", "bold");

hold off;

title(resultLayout, "PixelForge - Explainable AI for DR Screening", "FontSize", 15, "FontWeight", "bold");

%% =========================================================
% DISPLAY CLASS PROBABILITIES
%% =========================================================

%% 17. Command Window Probabilities

fprintf("\n======================================================\n");

fprintf("CLASS PROBABILITIES\n");

fprintf("======================================================\n");

for i = 1:numClasses

    fprintf("%-20s : %.2f %%\n", string(classNames{i}), scores(i) * 100);

end

fprintf("======================================================\n");

%% =========================================================
% SCREENING PRIORITY
%% =========================================================

%% 18. Determine Review Priority

if predictedClass == "No_DR"

    screeningFlag = "ROUTINE REVIEW";

else

    screeningFlag = "PRIORITY REVIEW";

end

%% =========================================================
% FINAL OUTPUT
%% =========================================================

fprintf("\n======================================================\n");

fprintf("PIXELFORGE EXPLAINABLE AI OUTPUT\n");

fprintf("======================================================\n");

fprintf("DR Grade          : %s\n", predictedClass);

fprintf("Confidence        : %.2f %%\n", confidence * 100);

fprintf("Review Priority   : %s\n", screeningFlag);

fprintf("Prediction Time   : %.4f seconds\n", inferenceTime);

fprintf("Grad-CAM Time     : %.4f seconds\n", gradcamTime);

fprintf("Explainability    : Grad-CAM\n");

fprintf("------------------------------------------------------\n");

fprintf("Grad-CAM visualizes regions that influenced the model prediction.\n");

fprintf("It should not be interpreted as lesion segmentation.\n");

fprintf("Clinician review is required for final interpretation.\n");

fprintf("======================================================\n");
