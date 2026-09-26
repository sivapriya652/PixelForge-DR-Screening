clc;
clear;
close all;

%% =========================================================
% PIXELFORGE
% DISTRICT-LEVEL TELEMEDICINE WORKFLOW SIMULATION
%
% Purpose:
% Simulate patient demand, image acquisition, bandwidth,
% AI processing throughput, clinician review capacity,
% completed screenings and backlog.
%
% NOTE:
% Operational values used here are prototype assumptions
% for workflow simulation and are not measured hospital data.
%% =========================================================

disp("======================================================");
disp("PIXELFORGE - SIMULINK WORKFLOW GENERATOR");
disp("======================================================");

%% 1. Model Name

modelName = "PixelForge_Telemedicine_Workflow";

%% 2. Close Existing Model

if bdIsLoaded(modelName)
    close_system(modelName, 0);
end

%% 3. Delete Existing Model File

modelFile = modelName + ".slx";

if isfile(modelFile)
    delete(modelFile);
end

%% 4. Create New Simulink Model

new_system(modelName);

open_system(modelName);

%% =========================================================
% SIMULATION ASSUMPTIONS
%% =========================================================

% Target annual screening demand
annualPatients = 100000;

% Assume 300 operational days per year
workingDaysPerYear = 300;

% Assume 8 screening hours per working day
hoursPerDay = 8;

% Required average patient arrival rate
arrivalRate = annualPatients / workingDaysPerYear / hoursPerDay;

% Prototype workflow assumptions
imageAcquisitionCapacity = 45;

bandwidthEfficiency = 0.85;

aiProcessingCapacity = 60;

priorityFraction = 0.30;

routineFraction = 0.70;

priorityReviewCapacity = 18;

routineReviewCapacity = 30;

fprintf("\nSIMULATION ASSUMPTIONS\n");
fprintf("------------------------------------------------------\n");
fprintf("Annual Patients             : %.0f\n", annualPatients);
fprintf("Working Days / Year         : %.0f\n", workingDaysPerYear);
fprintf("Operating Hours / Day       : %.0f\n", hoursPerDay);
fprintf("Required Arrival Rate       : %.2f patients/hour\n", arrivalRate);
fprintf("Image Acquisition Capacity  : %.2f patients/hour\n", imageAcquisitionCapacity);
fprintf("Bandwidth Efficiency        : %.0f %%\n", bandwidthEfficiency * 100);
fprintf("AI Processing Capacity      : %.2f patients/hour\n", aiProcessingCapacity);
fprintf("Priority Fraction           : %.0f %%\n", priorityFraction * 100);
fprintf("Routine Fraction            : %.0f %%\n", routineFraction * 100);
fprintf("Priority Review Capacity    : %.2f patients/hour\n", priorityReviewCapacity);
fprintf("Routine Review Capacity     : %.2f patients/hour\n", routineReviewCapacity);
fprintf("------------------------------------------------------\n");

%% =========================================================
% BLOCK POSITIONS
%% =========================================================

x1 = 50;
x2 = 220;
x3 = 390;
x4 = 560;
x5 = 730;
x6 = 900;
x7 = 1070;

yMain = 160;
yPriority = 80;
yRoutine = 260;

%% =========================================================
% STAGE 1 - PATIENT ARRIVAL
%% =========================================================

add_block("simulink/Sources/Constant", modelName + "/Patient Arrival Rate");

set_param(modelName + "/Patient Arrival Rate", "Value", num2str(arrivalRate));

set_param(modelName + "/Patient Arrival Rate", "Position", [x1 yMain x1+100 yMain+50]);

%% =========================================================
% STAGE 2 - IMAGE ACQUISITION CAPACITY
%% =========================================================

add_block("simulink/Discontinuities/Saturation", modelName + "/Image Acquisition");

set_param(modelName + "/Image Acquisition", "UpperLimit", num2str(imageAcquisitionCapacity));

set_param(modelName + "/Image Acquisition", "LowerLimit", "0");

set_param(modelName + "/Image Acquisition", "Position", [x2 yMain x2+110 yMain+50]);

%% =========================================================
% STAGE 3 - BANDWIDTH CONSTRAINT
%% =========================================================

add_block("simulink/Math Operations/Gain", modelName + "/Bandwidth Constraint");

set_param(modelName + "/Bandwidth Constraint", "Gain", num2str(bandwidthEfficiency));

set_param(modelName + "/Bandwidth Constraint", "Position", [x3 yMain x3+110 yMain+50]);

%% =========================================================
% STAGE 4 - AI PROCESSING CAPACITY
%% =========================================================

add_block("simulink/Discontinuities/Saturation", modelName + "/AI Processing");

set_param(modelName + "/AI Processing", "UpperLimit", num2str(aiProcessingCapacity));

set_param(modelName + "/AI Processing", "LowerLimit", "0");

set_param(modelName + "/AI Processing", "Position", [x4 yMain x4+110 yMain+50]);

%% =========================================================
% STAGE 5 - PRIORITY / ROUTINE SPLIT
%% =========================================================

add_block("simulink/Math Operations/Gain", modelName + "/Priority Cases");

set_param(modelName + "/Priority Cases", "Gain", num2str(priorityFraction));

set_param(modelName + "/Priority Cases", "Position", [x5 yPriority x5+110 yPriority+50]);

add_block("simulink/Math Operations/Gain", modelName + "/Routine Cases");

set_param(modelName + "/Routine Cases", "Gain", num2str(routineFraction));

set_param(modelName + "/Routine Cases", "Position", [x5 yRoutine x5+110 yRoutine+50]);

%% =========================================================
% STAGE 6 - CLINICAL REVIEW CAPACITY
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
% STAGE 7 - TOTAL COMPLETED SCREENINGS
%% =========================================================

add_block("simulink/Math Operations/Add", modelName + "/Completed Screenings");

set_param(modelName + "/Completed Screenings", "Inputs", "++");

set_param(modelName + "/Completed Screenings", "Position", [x7 yMain x7+100 yMain+60]);

%% =========================================================
% BACKLOG CALCULATION
%% =========================================================

add_block("simulink/Math Operations/Add", modelName + "/Backlog Rate");

set_param(modelName + "/Backlog Rate", "Inputs", "+-");

set_param(modelName + "/Backlog Rate", "Position", [1070 350 1170 410]);

%% =========================================================
% DISPLAY BLOCKS
%% =========================================================

add_block("simulink/Sinks/Display", modelName + "/Arrival Display");

set_param(modelName + "/Arrival Display", "Position", [50 300 150 350]);

add_block("simulink/Sinks/Display", modelName + "/AI Throughput Display");

set_param(modelName + "/AI Throughput Display", "Position", [560 300 660 350]);

add_block("simulink/Sinks/Display", modelName + "/Completed Display");

set_param(modelName + "/Completed Display", "Position", [1240 150 1340 210]);

add_block("simulink/Sinks/Display", modelName + "/Backlog Display");

set_param(modelName + "/Backlog Display", "Position", [1240 350 1340 410]);

%% =========================================================
% SCOPE
%% =========================================================

add_block("simulink/Signal Routing/Mux", modelName + "/Performance Mux");

set_param(modelName + "/Performance Mux", "Inputs", "4");

set_param(modelName + "/Performance Mux", "Position", [1240 500 1245 610]);

add_block("simulink/Sinks/Scope", modelName + "/Workflow Scope");

set_param(modelName + "/Workflow Scope", "Position", [1330 510 1430 600]);

%% =========================================================
% CONNECTIONS - MAIN WORKFLOW
%% =========================================================

add_line(modelName, "Patient Arrival Rate/1", "Image Acquisition/1", "autorouting", "on");

add_line(modelName, "Image Acquisition/1", "Bandwidth Constraint/1", "autorouting", "on");

add_line(modelName, "Bandwidth Constraint/1", "AI Processing/1", "autorouting", "on");

%% =========================================================
% AI OUTPUT SPLIT
%% =========================================================

add_line(modelName, "AI Processing/1", "Priority Cases/1", "autorouting", "on");

add_line(modelName, "AI Processing/1", "Routine Cases/1", "autorouting", "on");

%% =========================================================
% REVIEW WORKFLOW
%% =========================================================

add_line(modelName, "Priority Cases/1", "Priority Review/1", "autorouting", "on");

add_line(modelName, "Routine Cases/1", "Routine Review/1", "autorouting", "on");

add_line(modelName, "Priority Review/1", "Completed Screenings/1", "autorouting", "on");

add_line(modelName, "Routine Review/1", "Completed Screenings/2", "autorouting", "on");

%% =========================================================
% BACKLOG
%% =========================================================

add_line(modelName, "AI Processing/1", "Backlog Rate/1", "autorouting", "on");

add_line(modelName, "Completed Screenings/1", "Backlog Rate/2", "autorouting", "on");

%% =========================================================
% DISPLAY CONNECTIONS
%% =========================================================

add_line(modelName, "Patient Arrival Rate/1", "Arrival Display/1", "autorouting", "on");

add_line(modelName, "AI Processing/1", "AI Throughput Display/1", "autorouting", "on");

add_line(modelName, "Completed Screenings/1", "Completed Display/1", "autorouting", "on");

add_line(modelName, "Backlog Rate/1", "Backlog Display/1", "autorouting", "on");

%% =========================================================
% SCOPE CONNECTIONS
%% =========================================================

add_line(modelName, "Patient Arrival Rate/1", "Performance Mux/1", "autorouting", "on");

add_line(modelName, "AI Processing/1", "Performance Mux/2", "autorouting", "on");

add_line(modelName, "Completed Screenings/1", "Performance Mux/3", "autorouting", "on");

add_line(modelName, "Backlog Rate/1", "Performance Mux/4", "autorouting", "on");

add_line(modelName, "Performance Mux/1", "Workflow Scope/1", "autorouting", "on");

%% =========================================================
% MODEL SETTINGS
%% =========================================================

set_param(modelName, "StopTime", "24");

set_param(modelName, "Solver", "ode45");

%% =========================================================
% ADD MODEL ANNOTATIONS
%% =========================================================

Simulink.Annotation(modelName, "PIXELFORGE - DISTRICT TELEMEDICINE WORKFLOW SIMULATION");

annotations = find_system(modelName, "FindAll", "on", "Type", "annotation");

if ~isempty(annotations)
    set_param(annotations(end), "Position", [450 20 900 45]);
end

Simulink.Annotation(modelName, "Prototype operational assumptions - not measured clinical workflow data");

annotations = find_system(modelName, "FindAll", "on", "Type", "annotation");

if ~isempty(annotations)
    set_param(annotations(end), "Position", [450 50 900 75]);
end

%% =========================================================
% SAVE MODEL
%% =========================================================

save_system(modelName);

%% =========================================================
% UPDATE MODEL
%% =========================================================

set_param(modelName, "SimulationCommand", "update");

%% =========================================================
% FIT MODEL TO VIEW
%% =========================================================

open_system(modelName);

set_param(modelName, "ZoomFactor", "FitSystem");

%% =========================================================
% FINAL MESSAGE
%% =========================================================

disp(" ");
disp("======================================================");
disp("SIMULINK MODEL CREATED SUCCESSFULLY");
disp("======================================================");

fprintf("Model File : %s.slx\n", modelName);

fprintf("Target     : %.0f patients/year\n", annualPatients);

fprintf("Required Average Arrival Rate : %.2f patients/hour\n", arrivalRate);

disp(" ");
disp("Workflow:");
disp("Patient Arrival");
disp("      -> Image Acquisition");
disp("      -> Bandwidth Constraint");
disp("      -> AI Processing");
disp("      -> Priority / Routine Split");
disp("      -> Clinical Review");
disp("      -> Completed Screening");
disp("      -> Backlog Analysis");

disp("======================================================");
