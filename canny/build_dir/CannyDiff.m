%% Read images

% Final edges
serial = imread("CannyImage_output_EDGE/serial_0.83_outputImg0.ppm");
gpu = imread("CannyImage_output_EDGE/gpu_0.83_outputImg0.ppm");
opt = imread("CannyImage_output_EDGE/opt_0.83_outputImg0.ppm");
best = imread("CannyImage_output_EDGE/best_0.83_outputImg0.ppm");

% NMS
% serial = imread("CannyImage_output_NMS/serial_0.83_outputImg0.ppm");
% gpu = imread("CannyImage_output_NMS/gpu_0.83_outputImg0.ppm");
% opt = imread("CannyImage_output_NMS/opt_0.83_outputImg0.ppm");
% best = imread("CannyImage_output_NMS/best_0.83_outputImg0.ppm");

%% Compute image histograms

[serialHist, ~] = imhist(serial);
[gpuHist, ~] = imhist(gpu);
[optHist, ~] = imhist(opt);
[bestHist, ~] = imhist(best);

%% Compute image thresholds

serialThresh = 255 * graythresh(serial);
gpuThresh = 255 * graythresh(gpu);
optThresh = 255 * graythresh(opt);
bestThresh = 255 * graythresh(best);

%% Find differences in images

gpuDiff = serial - gpu;
optDiff = serial - opt;
bestDiff = serial - best;

gpuDiffMax = max(max(serial - gpu));
optDiffMax = max(max(serial - opt));
bestDiffMax = max(max(serial - best));

gpuHistDiffMax = max(gpuHist - serialHist);
gpuHistDiffMin = min(gpuHist - serialHist);

optHistDiffMax = max(optHist - serialHist);
optHistDiffMin = min(optHist - serialHist);

bestHistDiffMax = max(bestHist - serialHist);
bestHistDiffMin = min(bestHist - serialHist);

gpuIndex = find(gpuDiff);
optIndex = find(optDiff);
bestIndex = find(bestDiff);

%writematrix(gpuDiff);
%type 'gpuDiff.txt';

%writematrix(optDiff);
%type 'optDiff.txt';

%writematrix(bestDiff);
%type 'bestDiff.txt';


%% Display

figure(1)
subplot(2,2,1)
imshow(uint8(serial))
title ('Serial')
subplot(2,2,2)
imshow(uint8(gpu))
title ('GPU')
colormap(gray(256))
subplot(2,2,3)
imshow(uint8(opt))
title ('Optimized')
subplot(2,2,4)
imshow(uint8(best))
title ('Best')
%saveas (gcf,'Couple_Canny_01.png')