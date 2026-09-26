clc;
clear;
close all;

%% =========================================================
% PIXELFORGE
% FUNDUS IMAGE QUALITY ASSESSMENT
%% =========================================================

disp("======================================================");
disp("PIXELFORGE - FUNDUS IMAGE QUALITY ASSESSMENT");
disp("======================================================");

%% 1. Select Fundus Image

[file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff', 'Fundus Images'}, 'PixelForge - Select Fundus Image');

if isequal(file, 0)

    disp("No image selected.");
    return;

end

imagePath = fullfile(path, file);

fprintf("Selected Image : %s\n", file);

%% 2. Read Fundus Image

I = imread(imagePath);

%% 3. Ensure RGB Format

if ndims(I) == 2

    Irgb = cat(3, I, I, I);

else

    Irgb = I;

end

if size(Irgb, 3) == 1

    Irgb = cat(3, Irgb, Irgb, Irgb);

end

if size(Irgb, 3) > 3

    Irgb = Irgb(:, :, 1:3);

end

%% 4. Convert Image to Grayscale

Igray = rgb2gray(Irgb);

IgrayDouble = im2double(Igray);

%% =========================================================
% FOCUS / BLUR ASSESSMENT
%% =========================================================

%% 5. Calculate Focus Score

laplacianKernel = [0 1 0; 1 -4 1; 0 1 0];

laplacianImage = imfilter(IgrayDouble, laplacianKernel, "replicate");

focusScore = var(laplacianImage(:)) * 10000;

%% =========================================================
% ILLUMINATION ASSESSMENT
%% =========================================================

%% 6. Calculate Mean Illumination

illuminationScore = mean(IgrayDouble(:));

%% =========================================================
% CONTRAST ASSESSMENT
%% =========================================================

%% 7. Calculate Image Contrast

contrastScore = std(IgrayDouble(:));

%% =========================================================
% PROTOTYPE QUALITY THRESHOLDS
%% =========================================================

%% 8. Configure Thresholds
%
% These values are prototype heuristic thresholds.
% They are not clinically validated image-quality criteria.
%

minimumFocus = 20;

minimumIllumination = 0.15;

maximumIllumination = 0.90;

minimumContrast = 0.08;

%% 9. Evaluate Individual Quality Conditions

focusPass = focusScore >= minimumFocus;

illuminationPass = illuminationScore >= minimumIllumination && illuminationScore <= maximumIllumination;

contrastPass = contrastScore >= minimumContrast;

%% 10. Overall Quality Decision

qualityPass = focusPass && illuminationPass && contrastPass;

%% =========================================================
% DISPLAY QUALITY RESULTS
%% =========================================================

fprintf("\n======================================================\n");
fprintf("IMAGE QUALITY METRICS\n");
fprintf("======================================================\n");

fprintf("Focus Score       : %.2f\n", focusScore);

fprintf("Illumination      : %.3f\n", illuminationScore);

fprintf("Contrast          : %.3f\n", contrastScore);

fprintf("------------------------------------------------------\n");

%% 11. Focus Status

if focusPass

    focusStatus = "PASS";

else

    focusStatus = "FAIL";

end

fprintf("Focus Status      : %s\n", focusStatus);

%% 12. Illumination Status

if illuminationPass

    illuminationStatus = "PASS";

else

    illuminationStatus = "FAIL";

end

fprintf("Illumination      : %s\n", illuminationStatus);

%% 13. Contrast Status

if contrastPass

    contrastStatus = "PASS";

else

    contrastStatus = "FAIL";

end

fprintf("Contrast Status   : %s\n", contrastStatus);

fprintf("------------------------------------------------------\n");

%% =========================================================
% FINAL QUALITY DECISION
%% =========================================================

%% 14. Acceptable Image

if qualityPass

    qualityStatus = "ACCEPTABLE";

    recaptureReason = "None";

    fprintf("IMAGE QUALITY     : ACCEPTABLE\n");

    fprintf("ACTION            : CONTINUE DR SCREENING\n");

%% 15. Poor Quality Image

else

    qualityStatus = "RECAPTURE REQUIRED";

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

    fprintf("IMAGE QUALITY     : RECAPTURE REQUIRED\n");

    fprintf("ACTION            : DO NOT RUN DR CLASSIFICATION\n");

    fprintf("REASON            : %s\n", recaptureReason);

end

fprintf("======================================================\n");

%% =========================================================
% DISPLAY FUNDUS IMAGE
%% =========================================================

%% 16. Show Image With Quality Result

figure("Name", "PixelForge - Image Quality Assessment", "NumberTitle", "off", "Position", [200 100 950 650]);

imshow(Irgb);

if qualityPass

    title({
        "PIXELFORGE IMAGE QUALITY ASSESSMENT"
        ""
        "IMAGE QUALITY: ACCEPTABLE"
        sprintf("Focus: %.2f | Illumination: %.3f | Contrast: %.3f", focusScore, illuminationScore, contrastScore)
        "Action: Continue DR Screening"
    }, "FontSize", 12);

else

    title({
        "PIXELFORGE IMAGE QUALITY ASSESSMENT"
        ""
        "IMAGE QUALITY: RECAPTURE REQUIRED"
        sprintf("Focus: %.2f | Illumination: %.3f | Contrast: %.3f", focusScore, illuminationScore, contrastScore)
        char(recaptureReason)
    }, "FontSize", 12);

end

%% =========================================================
% QUALITY METRICS VISUALIZATION
%% =========================================================

%% 17. Normalize Metrics Relative to Prototype Thresholds

focusNormalized = focusScore / minimumFocus;

contrastNormalized = contrastScore / minimumContrast;

if illuminationScore < minimumIllumination

    illuminationNormalized = illuminationScore / minimumIllumination;

elseif illuminationScore > maximumIllumination

    illuminationNormalized = maximumIllumination / illuminationScore;

else

    illuminationNormalized = 1;

end

qualityValues = [focusNormalized illuminationNormalized contrastNormalized];

%% 18. Limit Display Range

qualityValuesDisplay = min(qualityValues, 2);

%% 19. Create Quality Metrics Graph

figure("Name", "PixelForge - Quality Metrics", "NumberTitle", "off");

metricNames = categorical(["Focus", "Illumination", "Contrast"]);

bar(metricNames, qualityValuesDisplay);

hold on;

yline(1, "--", "Prototype Reference");

hold off;

ylabel("Normalized Quality Score");

title("PixelForge - Fundus Image Quality Metrics");

grid on;

%% =========================================================
% FINAL CONFIRMATION
%% =========================================================

disp(" ");

disp("======================================================");
disp("IMAGE QUALITY ASSESSMENT COMPLETED");
disp("======================================================");

fprintf("Final Status : %s\n", qualityStatus);

if ~qualityPass

    fprintf("Reason       : %s\n", recaptureReason);

end

disp("======================================================");
