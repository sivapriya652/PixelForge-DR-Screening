clc;
clear;
close all;

%% =========================================================
% PIXELFORGE
% RESNET-50 MODEL EVALUATION
%% =========================================================

disp("======================================================");
disp("PIXELFORGE - DR MODEL EVALUATION");
disp("======================================================");

%% 1. Check Required Files

if ~isfile("DR_ResNet50_Final.mat")

    error("DR_ResNet50_Final.mat not found. Train the model first.");

end

if ~isfile("DR_DataSplit.mat")

    error("DR_DataSplit.mat not found. Run the training script first.");

end

%% 2. Load Trained Model

disp("Loading trained ResNet-50 model...");

modelData = load("DR_ResNet50_Final.mat");

net5 = modelData.net5;

classNames = modelData.classNames;

disp("Model loaded successfully.");

%% 3. Load Saved Test Dataset

disp(" ");

disp("Loading held-out test dataset...");

splitData = load("DR_DataSplit.mat");

imdsTest = splitData.imdsTest;

disp("Test Dataset Distribution:");

disp(countEachLabel(imdsTest));

fprintf("Total Test Images: %d\n", numel(imdsTest.Files));

%% 4. Prepare Test Images

inputSize = [224 224 3];

augimdsTest = augmentedImageDatastore(inputSize, imdsTest, "ColorPreprocessing", "gray2rgb");

disp("Test images prepared successfully.");

%% 5. Run Predictions

disp(" ");

disp("Running model evaluation...");

evaluationStart = tic;

scores = minibatchpredict(net5, augimdsTest, "ExecutionEnvironment", "cpu");

evaluationTime = toc(evaluationStart);

disp("Prediction completed.");

fprintf("Evaluation Time: %.2f seconds\n", evaluationTime);

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

%% 7. Obtain Predicted Classes

[confidence, predictedIndex] = max(scores, [], 2);

predictedLabels = strings(numel(predictedIndex), 1);

for i = 1:numel(predictedIndex)

    predictedLabels(i) = string(classNames{predictedIndex(i)});

end

trueLabels = string(imdsTest.Labels);

%% 8. Calculate Overall Accuracy

accuracy = mean(predictedLabels == trueLabels);

fprintf("\n======================================================\n");
fprintf("PIXELFORGE MODEL TEST RESULTS\n");
fprintf("======================================================\n");

fprintf("Test Accuracy : %.2f %%\n", accuracy * 100);

fprintf("======================================================\n");

%% 9. Convert Labels to Categorical

trueCategorical = categorical(trueLabels, string(classNames));

predictedCategorical = categorical(predictedLabels, string(classNames));

%% 10. Generate Confusion Matrix

CM = confusionmat(trueCategorical, predictedCategorical);

disp(" ");

disp("Confusion Matrix:");

disp(CM);

%% 11. Display Confusion Matrix

figure("Name", "PixelForge - Confusion Matrix", "NumberTitle", "off");

confusionchart(trueCategorical, predictedCategorical);

title("PixelForge DR Classification - Confusion Matrix");

%% 12. Initialize Per-Class Metrics

precision = zeros(numClasses, 1);

recall = zeros(numClasses, 1);

specificity = zeros(numClasses, 1);

f1Score = zeros(numClasses, 1);

%% 13. Calculate Per-Class Metrics

for i = 1:numClasses

    TP = CM(i, i);

    FN = sum(CM(i, :)) - TP;

    FP = sum(CM(:, i)) - TP;

    TN = sum(CM(:)) - TP - FN - FP;

    if TP + FP > 0

        precision(i) = TP / (TP + FP);

    end

    if TP + FN > 0

        recall(i) = TP / (TP + FN);

    end

    if TN + FP > 0

        specificity(i) = TN / (TN + FP);

    end

    if precision(i) + recall(i) > 0

        f1Score(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i));

    end

end

%% 14. Calculate Macro Metrics

macroPrecision = mean(precision);

macroRecall = mean(recall);

macroSpecificity = mean(specificity);

macroF1 = mean(f1Score);

%% 15. Create Per-Class Results Table

resultsTable = table(string(classNames), precision * 100, recall * 100, specificity * 100, f1Score * 100, "VariableNames", {"DR_Class", "Precision", "Recall_Sensitivity", "Specificity", "F1_Score"});

%% 16. Display Per-Class Performance

disp(" ");

disp("PER-CLASS PERFORMANCE:");

disp(resultsTable);

%% 17. Display Final Model Performance

fprintf("\n======================================================\n");
fprintf("FINAL PERFORMANCE\n");
fprintf("======================================================\n");

fprintf("Test Accuracy     : %.2f %%\n", accuracy * 100);

fprintf("Macro Precision   : %.2f %%\n", macroPrecision * 100);

fprintf("Macro Recall      : %.2f %%\n", macroRecall * 100);

fprintf("Macro Specificity : %.2f %%\n", macroSpecificity * 100);

fprintf("Macro F1 Score    : %.2f %%\n", macroF1 * 100);

fprintf("======================================================\n");

%% 18. Plot Macro Performance Metrics

figure("Name", "PixelForge - Overall Model Performance", "NumberTitle", "off");

metricNames = categorical(["Accuracy", "Precision", "Recall", "Specificity", "F1 Score"]);

metricValues = [accuracy macroPrecision macroRecall macroSpecificity macroF1] * 100;

bar(metricNames, metricValues);

ylim([0 100]);

ylabel("Performance (%)");

title("PixelForge - Overall Model Performance");

grid on;

%% 19. Plot Per-Class F1 Score

figure("Name", "PixelForge - Per-Class F1 Score", "NumberTitle", "off");

classCategories = categorical(string(classNames), string(classNames));

bar(classCategories, f1Score * 100);

ylim([0 100]);

xlabel("DR Class");

ylabel("F1 Score (%)");

title("PixelForge - Per-Class F1 Score");

grid on;

%% 20. Save Per-Class Results

writetable(resultsTable, "DR_PerClass_Metrics.csv");

%% 21. Save Complete Evaluation Results

save("DR_Evaluation_Results.mat", "accuracy", "macroPrecision", "macroRecall", "macroSpecificity", "macroF1", "precision", "recall", "specificity", "f1Score", "CM", "evaluationTime");

%% 22. Final Confirmation

disp(" ");

disp("======================================================");
disp("PIXELFORGE MODEL EVALUATION COMPLETED");
disp("======================================================");

disp("Generated Files:");

disp("1. DR_PerClass_Metrics.csv");

disp("2. DR_Evaluation_Results.mat");

disp(" ");

disp("Generated Figures:");

disp("1. Confusion Matrix");

disp("2. Overall Performance Graph");

disp("3. Per-Class F1 Score Graph");

disp("======================================================");
