clc;
close all;
clear all;

%% To read a image from file...
[filename, pathname] = uigetfile('*.*', 'Select a image');
I = imread([pathname,filename]);
figure(1);
imshow(I), title('Original Image');
   
%% Grayscale conversion..
I2 = rgb2gray(I);
figure(2), imshow(I2), title('Grayscale Image');

%% Gaussian filtering..
F = fspecial('gaussian');
I2 = imfilter(I2,F);
figure(3); imshow(I2); title('Gaussian Filtered Image');

Gray  = I2;

%% FCM Segmentation..
fim=mat2gray(I2);
level=graythresh(fim);
bwfim=im2bw(fim,level);
[bwfim0,level0]=fcmthresh(fim,0);
[seg_img,level1]=fcmthresh(fim,1);
subplot(2,2,1);
imshow(fim);title('Original');
subplot(2,2,2);
imshow(seg_img);title(sprintf('FCM1,level=%f',level1));

%%
boundaries = bwboundaries(seg_img);
daries = size(boundaries, 1);
for k = 1 : 1
	thisBoundary = boundaries{k};
      hold on;
      fill(thisBoundary(:,2), thisBoundary(:,1),'w');
      hold on;
	plot(thisBoundary(:,2), thisBoundary(:,1), 'r', 'LineWidth', 2);
end
hold off


%% Haar Feature Extraction..
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
    
featext_fcm = [Mean, Standard_Deviation, Entropy,Variance, Smoothness, Kurtosis, Skewness, IDM, cA, cH,cV,cD];

%% jdf
I1=imresize(I,[500 500]);


%% To extract Green Channel..
I1(:,:,1)=0;
I1(:,:,3)=0;
figure(3);imshow(I1);title('Green Channel');

Im=rgb2gray(I1);

%% Vessel detection..
dgrayeye1 = imadjust(Im,[0.1 0.9],[]);
figure(5);imshow(dgrayeye1);
    
se = strel('disk',1);
    
cannyeye = edge(dgrayeye1,'canny',0.15);
figure(6),imshow(cannyeye);
    
dilate = imdilate(cannyeye,se);
figure(7), imshow(dilate);

ginv = imcomplement (Im);               
adahist = adapthisteq(ginv);               
se = strel('ball',8,8);                   
gopen = imopen(adahist,se);                 
godisk = adahist - gopen;                   
medfilt = medfilt2(godisk);                 
background = imopen(medfilt,strel('disk',15));
I2 = medfilt - background;                 
I3 = imadjust(I2);                         

level = graythresh(I3);                    
bw = im2bw(I3,level);                      
bw = bwareaopen(bw, 30);                    

wname = 'sym4';
[CA,CH,CV,CD] = dwt2(bw,wname,'mode','per');
figure(8),imshow(CA),title('Approximate');


b = bwboundaries(bw);
I = imresize(I,[500 500]);
figure(9),imshow(I)
hold on

for k = 1:numel(b)
    plot(b{k}(:,2), b{k}(:,1), 'b', 'Linewidth', 1)
end

%% Microaneurysm detection..
adjustImage = imadjust(adahist,[],[],3);
comp = imcomplement(adjustImage);
J = imadjust(comp,[],[],4);
J = imcomplement(J);
J = imadjust(J,[],[],4);
K=fspecial('disk',5);
L=imfilter(J,K,'replicate');
L = im2bw(L,0.4);
M =  bwmorph(L,'tophat');
[A,H,V,D] = dwt2(M,wname,'mode','per');
figure(10),imshow(A);

b = bwboundaries(A);
I=imresize(I,[250 250]);
figure(11), imshow(I);
hold on
for area_bloodvessels = 1:numel(b)
    plot(b{area_bloodvessels}(:,2), b{area_bloodvessels}(:,1), 'b', 'Linewidth', 1)
end 

%% classification
load Trainfeat1.mat
load label.mat
svmstruct=fitcsvm(Trainfeat1,label);
Result=predict(svmstruct,featext_fcm);

if Result==1 
    disp('Microaneurysm is present');
    msgbox('Microaneurysm is present');
else
    disp('Microaneurysm is not present');
    msgbox('Microaneurysm is not present');
end

% CLASS PERFORMANCE
species=label;
load result.mat
c=result;
cp = classperf(label,result);
Sensitivity=cp.Sensitivity
Specificity=cp.Specificity
% get(cp)
 


%%
figure(12);
plotroc(label,result); 