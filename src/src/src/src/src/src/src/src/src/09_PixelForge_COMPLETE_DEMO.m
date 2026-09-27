clc;
clear;
close all;

%% =========================================================
% PIXELFORGE
% COMPLETE POC DEMONSTRATION
%
% PART A:
% AI-Assisted Diabetic Retinopathy Screening
%
% PART B:
% District-Level Telemedicine Workflow Simulation in Simulink
%
% Academic prototype for screening support.
% Clinician review is required for final interpretation.
%% =========================================================

disp("======================================================");
disp("                    PIXELFORGE");
disp(" AI-ASSISTED DIABETIC RETINOPATHY SCREENING SYSTEM");
disp("======================================================");

%% =========================================================
% PART A
% AI-ASSISTED FUNDUS IMAGE SCREENING
%% =========================================================

disp(" ");
disp("======================================================");
disp("PART A - AI FUNDUS IMAGE SCREENING");
disp("======================================================");

%% 1. Check Model

if ~isfile("DR_ResNet50_Final.mat")
    error("DR_ResNet50_Final.mat not found. Place the trained model in the MATLAB Current Folder.");
end

%% 2. Load Model

disp("Loading trained ResNet-50 model...");

modelData = load("DR_ResNet50_Final.mat");

net5 = modelData.net5;
classNames = modelData.classNames;

disp("Model loaded successfully.");

%% 3. Select Fundus Image

[file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff', 'Fundus Images'}, 'PixelForge - Select Fundus Image');

if isequal(file, 0)
    disp("No image selected.");
    return;
end

imagePath = fullfile(path, file);

fprintf("Selected Image : %s\n", file);

%% 4. Create Demo Sample ID

sampleNumber = randi([10000 99999]);

sampleID = sprintf("PF-%05d", sampleNumber);

screeningDate = string(datetime("now", "Format", "dd-MMM-yyyy HH:mm:ss"));

fprintf("Sample ID      : %s\n", sampleID);
fprintf("Date / Time    : %s\n", screeningDate);

%% 5. Start Complete AI Pipeline Timer

totalPipelineStart = tic;

%% =========================================================
% STAGE 1 - IMAGE LOADING
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
% STAGE 2 - IMAGE QUALITY ASSESSMENT
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

fprintf("Focus Score       : %.2f\n", focusScore);
fprintf("Illumination      : %.3f\n", illuminationScore);
fprintf("Contrast          : %.3f\n", contrastScore);
fprintf("Quality Check Time: %.4f seconds\n", qualityCheckTime);

%% =========================================================
% QUALITY FAILURE
%% =========================================================

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
    fprintf("Reason : %s\n", recaptureReason);
    fprintf("DR classification was not performed.\n");
    fprintf("Total Processing Time : %.4f seconds\n", totalPipelineTime);
    fprintf("======================================================\n");

    figure("Name", "PixelForge - Recapture Required", "NumberTitle", "off", "Position", [250 100 950 650]);

    imshow(Irgb);

    title({
        "PIXELFORGE"
        "IMAGE QUALITY: RECAPTURE REQUIRED"
        char(recaptureReason)
        sprintf("Focus: %.2f | Illumination: %.3f | Contrast: %.3f", focusScore, illuminationScore, contrastScore)
        "DR Classification Stopped"
    }, "FontSize", 12, "FontWeight", "bold");

    disp(" ");
    disp("Poor-quality image detected.");
    disp("For the complete POC demonstration, rerun the script using an acceptable fundus image.");

    return;
end

disp("IMAGE QUALITY: ACCEPTABLE");
disp("Proceeding to AI screening.");

%% =========================================================
% STAGE 3 - IMAGE PREPARATION
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
% STAGE 4 - RESNET-50 PREDICTION
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
% STAGE 5 - CONFIDENCE DISPLAY CATEGORY
%
% Prototype display category only.
% Not a clinically validated threshold.
%% =========================================================

if confidence >= 0.80
    confidenceLevel = "HIGH";
elseif confidence >= 0.60
    confidenceLevel = "MODERATE";
else
    confidenceLevel = "LOW";
end

%% =========================================================
% STAGE 6 - REVIEW PRIORITY
%% =========================================================

if predictedClass == "No_DR"
    reviewPriority = "ROUTINE REVIEW";
else
    reviewPriority = "PRIORITY REVIEW";
end

%% =========================================================
% STAGE 7 - GRAD-CAM
%% =========================================================

disp(" ");
disp("STAGE 4: GRAD-CAM EXPLAINABILITY");

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
% STAGE 8 - TOTAL AI PIPELINE TIME
%% =========================================================

totalPipelineTime = toc(totalPipelineStart);

%% =========================================================
% COMPLETE SCREENING REPORT
%% =========================================================

fprintf("\n======================================================\n");
fprintf("PIXELFORGE - AI SCREENING REPORT\n");
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

fprintf("\nFIVE-CLASS PROBABILITIES\n");
fprintf("------------------------------------------------------\n");

for i = 1:numClasses
    fprintf("%-20s : %.2f %%\n", string(classNames{i}), scores(i) * 100);
end

fprintf("\nPROCESSING TIMES\n");
fprintf("------------------------------------------------------\n");

fprintf("Quality Assessment : %.4f seconds\n", qualityCheckTime);
fprintf("Preprocessing      : %.4f seconds\n", preprocessingTime);
fprintf("Prediction         : %.4f seconds\n", predictionTime);
fprintf("Grad-CAM           : %.4f seconds\n", gradcamTime);
fprintf("Total AI Pipeline  : %.4f seconds\n", totalPipelineTime);

fprintf("======================================================\n");

%% =========================================================
% FINAL DOCTOR-FACING DASHBOARD
%% =========================================================

figure("Name", "PixelForge - Doctor Screening Dashboard", "NumberTitle", "off", "Position", [40 40 1450 820]);

dashboard = tiledlayout(2, 3, "TileSpacing", "compact", "Padding", "compact");

%% Original Image

nexttile;

imshow(Irgb);

title({
    "Fundus Image"
    sprintf("Sample ID: %s", sampleID)
}, "FontSize", 11, "FontWeight", "bold");

%% Grad-CAM

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

%% AI Result

nexttile;

axis off;

resultText = {
    "AI SCREENING RESULT"
    ""
    sprintf("DR Grade: %s", predictedClass)
    sprintf("Confidence: %.2f%%", confidence * 100)
    sprintf("Confidence Level: %s", confidenceLevel)
    ""
    "Review Priority"
    char(reviewPriority)
    ""
    "Clinician review required"
};

text(0.05, 0.95, resultText, "Units", "normalized", "VerticalAlignment", "top", "FontSize", 12, "FontWeight", "bold");

%% Probability Distribution

nexttile([1 2]);

classCategories = categorical(string(classNames), string(classNames));

bar(classCategories, scores * 100);

ylim([0 100]);

xlabel("DR Grade");

ylabel("Probability (%)");

title("Five-Class DR Probability Distribution", "FontSize", 11, "FontWeight", "bold");

grid on;

%% Technical Information

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
    sprintf("Prediction: %.3f s", predictionTime)
    sprintf("Total Pipeline: %.3f s", totalPipelineTime)
};

text(0.05, 0.95, technicalText, "Units", "normalized", "VerticalAlignment", "top", "FontSize", 10.5);

title(dashboard, "PIXELFORGE - AI-ASSISTED DIABETIC RETINOPATHY SCREENING", "FontSize", 16, "FontWeight", "bold");

%% =========================================================
% AI PIPELINE PROCESSING TIME GRAPH
%% =========================================================

figure("Name", "PixelForge - AI Processing Time", "NumberTitle", "off");

processingStages = categorical(["Quality Check", "Preprocessing", "Prediction", "Grad-CAM", "Total Pipeline"]);

processingTimes = [qualityCheckTime preprocessingTime predictionTime gradcamTime totalPipelineTime];

bar(processingStages, processingTimes);

ylabel("Processing Time (seconds)");

title("PixelForge - AI Screening Pipeline Processing Time");

grid on;

%% =========================================================
% PART B
% DISTRICT-LEVEL SIMULINK WORKFLOW
%% =========================================================

disp(" ");
disp("======================================================");
disp("PART B - SIMULINK TELEMEDICINE WORKFLOW");
disp("======================================================");

disp("Creating district-level workflow simulation...");

%% =========================================================
% OPERATIONAL SIMULATION ASSUMPTIONS
%
% These are prototype scenario assumptions.
% They are NOT measured clinical workflow data.
%% =========================================================

annualPatients = 100000;

workingDaysPerYear = 300;

hoursPerDay = 8;

requiredArrivalRate = annualPatients / workingDaysPerYear / hoursPerDay;

imageAcquisitionCapacity = 45;

bandwidthEfficiency = 0.85;

aiProcessingCapacity = 60;

priorityFraction = 0.30;

routineFraction = 0.70;

priorityReviewCapacity = 18;

routineReviewCapacity = 30;

fprintf("\nDISTRICT WORKFLOW ASSUMPTIONS\n");
fprintf("------------------------------------------------------\n");

fprintf("Annual Screening Demand     : %.0f patients/year\n", annualPatients);
fprintf("Operational Days            : %.0f days/year\n", workingDaysPerYear);
fprintf("Screening Hours             : %.0f hours/day\n", hoursPerDay);
fprintf("Required Average Throughput : %.2f patients/hour\n", requiredArrivalRate);
fprintf("Image Acquisition Capacity  : %.2f patients/hour\n", imageAcquisitionCapacity);
fprintf("Bandwidth Efficiency        : %.0f %%\n", bandwidthEfficiency * 100);
fprintf("AI Processing Capacity      : %.2f patients/hour\n", aiProcessingCapacity);
fprintf("Priority Review Capacity    : %.2f patients/hour\n", priorityReviewCapacity);
fprintf("Routine Review Capacity     : %.2f patients/hour\n", routineReviewCapacity);

fprintf("------------------------------------------------------\n");

%% =========================================================
% CALCULATE WORKFLOW RATES
%% =========================================================

acquiredRate = min(requiredArrivalRate, imageAcquisitionCapacity);

bandwidthAdjustedRate = acquiredRate * bandwidthEfficiency;

aiThroughputRate = min(bandwidthAdjustedRate, aiProcessingCapacity);

priorityDemandRate = aiThroughputRate * priorityFraction;

routineDemandRate = aiThroughputRate * routineFraction;

priorityCompletedRate = min(priorityDemandRate, priorityReviewCapacity);

routineCompletedRate = min(routineDemandRate, routineReviewCapacity);

completedRate = priorityCompletedRate + routineCompletedRate;

backlogRate = max(aiThroughputRate - completedRate, 0);

annualCompletedCapacity = completedRate * workingDaysPerYear * hoursPerDay;

fprintf("\nCALCULATED WORKFLOW PERFORMANCE\n");
fprintf("------------------------------------------------------\n");

fprintf("Acquired Images             : %.2f patients/hour\n", acquiredRate);
fprintf("After Bandwidth Constraint  : %.2f patients/hour\n", bandwidthAdjustedRate);
fprintf("AI Throughput               : %.2f patients/hour\n", aiThroughputRate);
fprintf("Priority Demand             : %.2f patients/hour\n", priorityDemandRate);
fprintf("Routine Demand              : %.2f patients/hour\n", routineDemandRate);
fprintf("Completed Reviews           : %.2f patients/hour\n", completedRate);
fprintf("Backlog Rate                : %.2f patients/hour\n", backlogRate);
fprintf("Annual Completed Capacity   : %.0f patients/year\n", annualCompletedCapacity);

fprintf("------------------------------------------------------\n");

%% =========================================================
% CREATE SIMULINK MODEL
%% =========================================================

modelName = "PixelForge_Telemedicine_Workflow";

if bdIsLoaded(modelName)
    close_system(modelName, 0);
end

modelFile = modelName + ".slx";

if isfile(modelFile)
    delete(modelFile);
end

new_system(modelName);

open_system(modelName);

%% =========================================================
% BLOCK POSITIONS
%% =========================================================

x1 = 50;
x2 = 220;
x3 = 390;
x4 = 560;
x5 = 730;
x6 = 900;
x7 = 1080;

yMain = 180;
yPriority = 80;
yRoutine = 280;

%% =========================================================
% PATIENT ARRIVAL
%% =========================================================

add_block("simulink/Sources/Constant", modelName + "/Patient Arrival");

set_param(modelName + "/Patient Arrival", "Value", num2str(requiredArrivalRate));

set_param(modelName + "/Patient Arrival", "Position", [x1 yMain x1+110 yMain+50]);

%% =========================================================
% IMAGE ACQUISITION
%% =========================================================

add_block("simulink/Discontinuities/Saturation", modelName + "/Image Acquisition");

set_param(modelName + "/Image Acquisition", "UpperLimit", num2str(imageAcquisitionCapacity));

set_param(modelName + "/Image Acquisition", "LowerLimit", "0");

set_param(modelName + "/Image Acquisition", "Position", [x2 yMain x2+110 yMain+50]);

%% =========================================================
% BANDWIDTH
%% =========================================================

add_block("simulink/Math Operations/Gain", modelName + "/Bandwidth Constraint");

set_param(modelName + "/Bandwidth Constraint", "Gain", num2str(bandwidthEfficiency));

set_param(modelName + "/Bandwidth Constraint", "Position", [x3 yMain x3+110 yMain+50]);

%% =========================================================
% AI PROCESSING
%% =========================================================

add_block("simulink/Discontinuities/Saturation", modelName + "/AI Processing");

set_param(modelName + "/AI Processing", "UpperLimit", num2str(aiProcessingCapacity));

set_param(modelName + "/AI Processing", "LowerLimit", "0");

set_param(modelName + "/AI Processing", "Position", [x4 yMain x4+110 yMain+50]);

%% =========================================================
% PRIORITY / ROUTINE SPLIT
%% =========================================================

add_block("simulink/Math Operations/Gain", modelName + "/Priority Cases");

set_param(modelName + "/Priority Cases", "Gain", num2str(priorityFraction));

set_param(modelName + "/Priority Cases", "Position", [x5 yPriority x5+110 yPriority+50]);

add_block("simulink/Math Operations/Gain", modelName + "/Routine Cases");

set_param(modelName + "/Routine Cases", "Gain", num2str(routineFraction));

set_param(modelName + "/Routine Cases", "Position", [x5 yRoutine x5+110 yRoutine+50]);

%% =========================================================
% REVIEW CAPACITY
%% =========================================================

add_block("simulink/Discontinuities/Saturation", modelName + "/Priority Review");

set_param(modelName + "/Priority Review", "UpperLimit", num2str(priorityReviewCapacity));

set_param(modelName + "/Priority Review", "LowerLimit", "0");

set_param(modelName + "/Priority Review", "Position", [x6 yPriority x6+110 yPriority+50]);

add_block("simulink/Discontinuities/Saturation", modelName + "/Routine Review");

set_param(modelName + "/Routine Review", "UpperLimit", num2str(routineReviewCapacity));

set_param(modelName + "/Routine Review", "LowerLimit", "0");

set_param(modelName + "/Routine Review", "Position", [x6 yRoutine x6+110 yRoutine+50]);

%% =========================================================
% COMPLETED SCREENINGS
%% =========================================================

add_block("simulink/Math Operations/Add", modelName + "/Completed Screening");

set_param(modelName + "/Completed Screening", "Inputs", "++");

set_param(modelName + "/Completed Screening", "Position", [x7 yMain x7+110 yMain+60]);

%% =========================================================
% BACKLOG
%% =========================================================

add_block("simulink/Math Operations/Add", modelName + "/Backlog Rate");

set_param(modelName + "/Backlog Rate", "Inputs", "+-");

set_param(modelName + "/Backlog Rate", "Position", [1080 390 1190 450]);

%% =========================================================
% DISPLAYS
%% =========================================================

add_block("simulink/Sinks/Display", modelName + "/Arrival Rate Display");

set_param(modelName + "/Arrival Rate Display", "Position", [50 340 150 390]);

add_block("simulink/Sinks/Display", modelName + "/AI Throughput Display");

set_param(modelName + "/AI Throughput Display", "Position", [560 340 660 390]);

add_block("simulink/Sinks/Display", modelName + "/Completed Display");

set_param(modelName + "/Completed Display", "Position", [1260 180 1370 230]);

add_block("simulink/Sinks/Display", modelName + "/Backlog Display");

set_param(modelName + "/Backlog Display", "Position", [1260 390 1370 440]);

%% =========================================================
% SCOPE
%% =========================================================

add_block("simulink/Signal Routing/Mux", modelName + "/Performance Mux");

set_param(modelName + "/Performance Mux", "Inputs", "4");

set_param(modelName + "/Performance Mux", "Position", [1240 510 1245 620]);

add_block("simulink/Sinks/Scope", modelName + "/Workflow Scope");

set_param(modelName + "/Workflow Scope", "Position", [1330 525 1430 605]);

%% =========================================================
% MAIN CONNECTIONS
%% =========================================================

add_line(modelName, "Patient Arrival/1", "Image Acquisition/1", "autorouting", "on");

add_line(modelName, "Image Acquisition/1", "Bandwidth Constraint/1", "autorouting", "on");

add_line(modelName, "Bandwidth Constraint/1", "AI Processing/1", "autorouting", "on");

%% =========================================================
% SPLIT CONNECTIONS
%% =========================================================

add_line(modelName, "AI Processing/1", "Priority Cases/1", "autorouting", "on");

add_line(modelName, "AI Processing/1", "Routine Cases/1", "autorouting", "on");

%% =========================================================
% REVIEW CONNECTIONS
%% =========================================================

add_line(modelName, "Priority Cases/1", "Priority Review/1", "autorouting", "on");

add_line(modelName, "Routine Cases/1", "Routine Review/1", "autorouting", "on");

add_line(modelName, "Priority Review/1", "Completed Screening/1", "autorouting", "on");

add_line(modelName, "Routine Review/1", "Completed Screening/2", "autorouting", "on");

%% =========================================================
% BACKLOG CONNECTIONS
%% =========================================================

add_line(modelName, "AI Processing/1", "Backlog Rate/1", "autorouting", "on");

add_line(modelName, "Completed Screening/1", "Backlog Rate/2", "autorouting", "on");

%% =========================================================
% DISPLAY CONNECTIONS
%% =========================================================

add_line(modelName, "Patient Arrival/1", "Arrival Rate Display/1", "autorouting", "on");

add_line(modelName, "AI Processing/1", "AI Throughput Display/1", "autorouting", "on");

add_line(modelName, "Completed Screening/1", "Completed Display/1", "autorouting", "on");

add_line(modelName, "Backlog Rate/1", "Backlog Display/1", "autorouting", "on");

%% =========================================================
% SCOPE CONNECTIONS
%% =========================================================

add_line(modelName, "Patient Arrival/1", "Performance Mux/1", "autorouting", "on");

add_line(modelName, "AI Processing/1", "Performance Mux/2", "autorouting", "on");

add_line(modelName, "Completed Screening/1", "Performance Mux/3", "autorouting", "on");

add_line(modelName, "Backlog Rate/1", "Performance Mux/4", "autorouting", "on");

add_line(modelName, "Performance Mux/1", "Workflow Scope/1", "autorouting", "on");

%% =========================================================
% SIMULATION SETTINGS
%% =========================================================

set_param(modelName, "StopTime", "24");

set_param(modelName, "Solver", "ode45");

%% =========================================================
% ANNOTATIONS
%% =========================================================

annotation1 = Simulink.Annotation(modelName, "PIXELFORGE - DISTRICT TELEMEDICINE WORKFLOW");

annotation1.Position = [430 20 930 45];

annotation2 = Simulink.Annotation(modelName, "100,000 Patient/Year Prototype Resource Simulation");

annotation2.Position = [460 50 900 75];

annotation3 = Simulink.Annotation(modelName, "Operational parameters are simulation assumptions, not measured hospital data");

annotation3.Position = [420 650 950 675];

%% =========================================================
% SAVE AND UPDATE MODEL
%% =========================================================

save_system(modelName);

set_param(modelName, "SimulationCommand", "update");

open_system(modelName);

set_param(modelName, "ZoomFactor", "FitSystem");

%% =========================================================
% RUN SIMULINK MODEL
%% =========================================================

disp(" ");
disp("Running Simulink workflow simulation...");

sim(modelName);

disp("Simulink workflow simulation completed.");

%% =========================================================
% RESOURCE ALLOCATION SUMMARY GRAPH
%% =========================================================

figure("Name", "PixelForge - District Workflow Capacity", "NumberTitle", "off");

workflowStages = categorical(["Required Demand", "Image Acquisition", "After Bandwidth", "AI Processing", "Clinical Review"]);

workflowRates = [requiredArrivalRate acquiredRate bandwidthAdjustedRate aiThroughputRate completedRate];

bar(workflowStages, workflowRates);

ylabel("Patients per Hour");

title("PixelForge - District Screening Workflow Capacity");

grid on;

%% =========================================================
% ANNUAL CAPACITY COMPARISON
%% =========================================================

figure("Name", "PixelForge - Annual Screening Capacity", "NumberTitle", "off");

annualCategories = categorical(["Annual Demand", "Simulated Capacity"]);

annualValues = [annualPatients annualCompletedCapacity];

bar(annualCategories, annualValues);

ylabel("Patients per Year");

title("PixelForge - Annual Screening Demand vs Simulated Capacity");

grid on;

%% =========================================================
% FINAL COMPLETE POC SUMMARY
%% =========================================================

disp(" ");
disp("======================================================");
disp("              PIXELFORGE POC COMPLETED");
disp("======================================================");

disp("PART A - AI SCREENING");
fprintf("DR Grade               : %s\n", predictedClass);
fprintf("Prediction Confidence  : %.2f %%\n", confidence * 100);
fprintf("Review Priority        : %s\n", reviewPriority);
fprintf("AI Pipeline Time       : %.4f seconds\n", totalPipelineTime);

disp(" ");

disp("PART B - DISTRICT WORKFLOW");
fprintf("Annual Target          : %.0f patients\n", annualPatients);
fprintf("Required Throughput    : %.2f patients/hour\n", requiredArrivalRate);
fprintf("AI Throughput          : %.2f patients/hour\n", aiThroughputRate);
fprintf("Clinical Completion    : %.2f patients/hour\n", completedRate);
fprintf("Backlog Rate           : %.2f patients/hour\n", backlogRate);
fprintf("Annual Capacity        : %.0f patients/year\n", annualCompletedCapacity);

disp(" ");
disp("Generated Simulink Model:");
disp("PixelForge_Telemedicine_Workflow.slx");

disp(" ");
disp("IMPORTANT:");
disp("AI output is intended for screening support.");
disp("Grad-CAM represents model attention, not lesion segmentation.");
disp("Operational Simulink values are prototype scenario assumptions.");
disp("Clinical validation is required before real-world deployment.");

disp("======================================================");
