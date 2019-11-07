
for yi = 1:35
       yi 
%% To read a image from file...
I = imread(['DATA\',num2str(yi),'.png']);

%% Grayscale conversion..
I = rgb2gray(I);

%% Gaussian filtering..
F = fspecial('gaussian');
I = imfilter(I,F);


Gray  = I;
%% FCM Segmentation..
fim=mat2gray(I);
level=graythresh(fim);
bwfim=im2bw(fim,level);
[bwfim0,level0]=fcmthresh(fim,0);
[seg_img,level1]=fcmthresh(fim,1);


%% Haar Feature Extraction+ LCM
[cA,cH,cV,cD] = dwt2(seg_img,'haar');
cA = mean2(cA);
cH = mean2(cH);
cV = mean2(cV);
cD = mean2(cD);

Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));
% Inverse Difference Movement
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
    
featext_fcm = [Mean, Standard_Deviation, Entropy,Variance, Smoothness, Kurtosis, Skewness, IDM, cA,cH,cV,cD];
Trainfeat1(yi,:) = featext_fcm;
end
save Trainfeat1 Trainfeat1
