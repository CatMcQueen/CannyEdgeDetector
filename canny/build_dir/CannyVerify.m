
%% Read image

Image = imread("CannyImage/Dataset/0/input.ppm");
imageIn = imread("CannyImage_output/serial_0.83_outputImg0.ppm");

imageGray = 0.21*Image(:,:,1) + 0.71*Image(:,:,2) + 0.07*Image(:,:,3);

%% Find edges

[edges, threshOut] = edge(imageGray, 'Canny', 0.245, 0.83);

figure(1)
subplot(1,2,1)
imshow(edges * 255)
title ('MatLab')
subplot(1,2,2)
imshow(imageIn * 255)
title ('C++')