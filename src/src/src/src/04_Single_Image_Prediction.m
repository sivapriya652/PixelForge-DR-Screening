clc;
clear;
close all;

%% =========================================================
% PIXELFORGE
% SINGLE FUNDUS IMAGE DR PREDICTION
%% =========================================================

disp("======================================================");
disp("PIXELFORGE - SINGLE IMAGE DR SCREENING");
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

fprintf("Model          : ResNet-50\n");
fprintf("Number Classes : %d\n", numel(classNames));

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

%% 4. Create Single Image Datastore

singleImds = imageDatastore(imagePath);

disp("Single image datastore created successfully.");

%% 5. Prepare Image for ResNet-50

inputSize = [224 224 3];

augSingleImage = augmentedImageDatastore(inputSize, singleImds, "ColorPreprocessing", "gray2rgb");

disp("Image prepared successfully.");

%% 6. Run Prediction

disp(" ");
disp("Running DR prediction...");

predictionStart = tic;

scores = minibatchpredict(net5, augSingleImage, "ExecutionEnvironment", "cpu");

inferenceTime = toc(predictionStart);

disp("Prediction completed.");

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

%% 8. Obtain Predicted DR Class

[confidence, predictedIndex] = max(scores, [], 2);

predictedClass = string(classNames{predictedIndex});

%% 9. Determine Review Priority

if predictedClass == "No_DR"

    screeningFlag = "ROUTINE REVIEW";

else

    screeningFlag = "PRIORITY REVIEW";

end

%% 10. Display Main Prediction

fprintf("\n======================================================\n");
fprintf("PIXELFORGE DR SCREENING RESULT\n");
fprintf("======================================================\n");

fprintf("Predicted DR Grade : %s\n", predictedClass);

fprintf("Confidence         : %.2f %%\n", confidence * 100);

fprintf("Review Priority    : %s\n", screeningFlag);

fprintf("Inference Time     : %.4f seconds\n", inferenceTime);

fprintf("======================================================\n");

%% 11. Display All Class Probabilities

fprintf("\nCLASS PROBABILITIES\n");

fprintf("------------------------------------------------------\n");

for i = 1:numClasses

    fprintf("%-20s : %.2f %%\n", string(classNames{i}), scores(i) * 100);

end

fprintf("------------------------------------------------------\n");

%% 12. Read Original Fundus Image

Ioriginal = imread(imagePath);

%% 13. Display Fundus Image With Prediction

figure("Name", "PixelForge - DR Prediction", "NumberTitle", "off");

imshow(Ioriginal);

title({
    sprintf("Predicted DR Grade: %s", predictedClass)
    sprintf("Confidence: %.2f%% | %s", confidence * 100, screeningFlag)
}, "FontSize", 12);

%% 14. Display Probability Distribution

figure("Name", "PixelForge - Class Probabilities", "NumberTitle", "off");

classCategories = categorical(string(classNames), string(classNames));

bar(classCategories, scores * 100);

ylim([0 100]);

xlabel("DR Grade");

ylabel("Probability (%)");

title("PixelForge - Five-Class DR Probability Distribution");

grid on;

%% 15. Final Confirmation

disp(" ");

disp("======================================================");
disp("SINGLE IMAGE DR SCREENING COMPLETED");
disp("======================================================");

fprintf("Prediction      : %s\n", predictedClass);

fprintf("Confidence      : %.2f %%\n", confidence * 100);

fprintf("Review Priority : %s\n", screeningFlag);

disp(" ");
disp("AI screening support - clinician review required.");

disp("======================================================");
