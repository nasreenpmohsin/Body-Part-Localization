 clc; close all; 
%%Extrinisc comparison of camera poses
% loading the extrinsic calibration
load('C:\Users\Nachu\Documents\thesis papers\journal2017_v3\set66_Prpexp\CalibrationResults\RT_123_dep_v3.mat');%Proposed with depth
load('C:\Users\Nachu\Documents\thesis papers\journal2017_v3\set66_Prpexp\CalibrationResults\RT_123_ir_v3_1.mat');%Proposed with IR
load('C:\Users\Nachu\Documents\thesis papers\journal2017_v3\set70_humanmotion\CalibrationResults\RT_123_icp_v1.mat');%ICP
load('C:\Users\Nachu\Documents\thesis papers\journal2017_v3\set67_exp1&2\CalibrationResults\RT_123_chess_v2.mat');%Chessboard
load('C:\Users\Nachu\Documents\thesis papers\journal2017_v3\set67_exp1&2\CalibrationResults\RT_123_exp2_v4.mat');%v2
n_exp=5;%number of approaches used
R_12_all=zeros(3,3,4); t_12_all=zeros(3,4);R_13_all=zeros(3,3,4); t_13_all=zeros(3,4);
R_12_all(:,:,1)=R_12_chess;t_12_all(:,1)=t_12_chess;R_13_all(:,:,1)=R_13_chess;t_13_all(:,1)=t_13_chess;
R_12_all(:,:,2)=R_12_exp2_1;t_12_all(:,2)=t_12_chess;R_13_all(:,:,2)=R_13_ir;%R_13_exp2_2;
t_13_all(:,2)=[-t_13_exp2_1(1);-t_13_exp2_1(2);t_13_exp2_1(1)];%-R_13_exp2_2*t_13_exp2_1;
R_12_all(:,:,3)=R_12_icp;t_12_all(:,3)=t_12_icp;R_13_all(:,:,3)=R_13_icp;t_13_all(:,3)=t_13_icp;
R_12_all(:,:,4)=R_12_ir;t_12_all(:,4)=t_12_ir;R_13_all(:,:,4)=R_13_ir;t_13_all(:,4)=t_13_ir;
R_12_all(:,:,5)=R_12_dep;t_12_all(:,5)=t_12_dep;R_13_all(:,:,5)=R_13_dep;t_13_all(:,5)=t_13_dep;

point1 = [1,0,0];%
point2 = [0,1,0];%;
point3 = [0 0 1];%;
origin = [0,0,0];
pt1_2=zeros(3,n_exp);pt2_2=zeros(3,n_exp);pt3_2=zeros(3,n_exp);
pt1_3=zeros(3,n_exp);pt2_3=zeros(3,n_exp);pt3_3=zeros(3,n_exp);
origin2=t_12_all;origin3=t_13_all;
colors=lines(n_exp);
figure();a=zeros(1,n_exp);
for i=1:n_exp
    if i~=4
    pt1_2(:,i)= R_12_all(:,:,i)*point1'+t_12_all(:,i);
    pt2_2(:,i)= R_12_all(:,:,i)*point2'+t_12_all(:,i);
    pt3_2(:,i)= R_12_all(:,:,i)*point3'+t_12_all(:,i);
    
    pt1_3(:,i)= R_13_all(:,:,i)*point1'+t_13_all(:,i);
    pt2_3(:,i)= R_13_all(:,:,i)*point2'+t_13_all(:,i);
    pt3_3(:,i)= R_13_all(:,:,i)*point3'+t_13_all(:,i);
    
    hold on;
    arrow_size=0.4;
    %Cam1
    dir_vec1=point1-origin;dir_vec2=point2-origin;dir_vec3=point3-origin;
    % plot3([origin(1) point1(1)],[origin(2) point1(2)],[origin(3) point1(3)],'g','LineWidth',1);hold on;
    % plot3([origin(1) point2(1)],[origin(2) point2(2)],[origin(3) point2(3)],'r','LineWidth',1);hold on;
    % plot3([origin(1) point3(1)],[origin(2) point3(2)],[origin(3) point3(3)],'b','LineWidth',1);hold on;
    a(i)=quiver3(origin(1),origin(2),origin(3),dir_vec1(1),dir_vec1(2),dir_vec1(3),'Color',colors(i,:),'MaxHeadSize',arrow_size);hold on;
    quiver3(origin(1),origin(2),origin(3),dir_vec2(1),dir_vec2(2),dir_vec2(3),'Color',colors(i,:),'MaxHeadSize',arrow_size);hold on;
    quiver3(origin(1),origin(2),origin(3),dir_vec3(1),dir_vec3(2),dir_vec3(3),'Color',colors(i,:),'MaxHeadSize',arrow_size);hold on;
%     text(origin(1),origin(2),origin(3),'Sensor 1','HorizontalAlignment','left','FontSize',8);

    %Cam2
    dir_vec1=R_12_all(:,1,i);dir_vec2=R_12_all(:,2,i);dir_vec3=R_12_all(:,3,i);
    % dir_vec1=pt1_2-origin2;dir_vec2=pt2_2-origin2;dir_vec3=pt3_2(3)-origin2;
    % plot3([origin2(1,i) pt1_2(1,i)],[origin2(2,i) pt1_2(2,i)],[origin2(3,i) pt1_2(3,i)],'g','LineWidth',1);hold on;
    % plot3([origin2(1,i) pt2_2(1,i)],[origin2(2,i) pt2_2(2,i)],[origin2(3,i) pt2_2(3,i)],'r','LineWidth',1);hold on;
    % plot3([origin2(1,i) pt3_2(1,i)],[origin2(2,i) pt3_2(2,i)],[origin2(3,i) pt3_2(3,i)],'b','LineWidth',1);hold on;
    quiver3(origin2(1,i),origin2(2,i),origin2(3,i),dir_vec1(1),dir_vec1(2),dir_vec1(3),'Color',colors(i,:),'MaxHeadSize',arrow_size);hold on;
    quiver3(origin2(1,i),origin2(2,i),origin2(3,i),dir_vec2(1),dir_vec2(2),dir_vec2(3),'Color',colors(i,:),'MaxHeadSize',arrow_size);hold on;
    quiver3(origin2(1,i),origin2(2,i),origin2(3,i),dir_vec3(1),dir_vec3(2),dir_vec3(3),'Color',colors(i,:),'MaxHeadSize',arrow_size);hold on;

%     text(origin2(1,i),origin2(2,i),origin2(3,i),'Sensor 2','HorizontalAlignment','left','FontSize',8);
    %Cam3
    dir_vec1=R_13_all(:,1,i);dir_vec2=R_13_all(:,2,i);dir_vec3=R_13_all(:,3,i);
    % dir_vec1=pt1_3-origin3;dir_vec2=pt2_3-origin3;dir_vec3=pt3_3(3)-origin3;
    % plot3([origin3(1,i) pt1_3(1,i)],[origin3(2,i) pt1_3(2,i)],[origin3(3,i) pt1_3(3,i)],'g','LineWidth',1);hold on;
    % plot3([origin3(1,i) pt2_3(1,i)],[origin3(2,i) pt2_3(2,i)],[origin3(3,i) pt2_3(3,i)],'r','LineWidth',1);hold on;
    % plot3([origin3(1,i) pt3_3(1,i)],[origin3(2,i) pt3_3(2,i)],[origin3(3,i) pt3_3(3,i)],'b','LineWidth',1);hold on;
    quiver3(origin3(1,i),origin3(2,i),origin3(3,i),dir_vec1(1),dir_vec1(2),dir_vec1(3),'Color',colors(i,:),'MaxHeadSize',arrow_size);hold on;
    quiver3(origin3(1,i),origin3(2,i),origin3(3,i),dir_vec2(1),dir_vec2(2),dir_vec2(3),'Color',colors(i,:),'MaxHeadSize',arrow_size);hold on;
    quiver3(origin3(1,i),origin3(2,i),origin3(3,i),dir_vec3(1),dir_vec3(2),dir_vec3(3),'Color',colors(i,:),'MaxHeadSize',arrow_size);hold on;
%     text(origin3(1,i),origin3(2,i),origin3(3,i),'Sensor 3','HorizontalAlignment','left','FontSize',8);
    end
end
 text(origin(1),origin(2),origin(3),'Sensor 1','HorizontalAlignment','left','FontSize',8);
text(origin2(1,5)+0.5,origin2(2,5)+0.5,origin2(3,5)+0.5,'Sensor 2','HorizontalAlignment','left','FontSize',8);
text(origin3(1,5)+0.5,origin3(2,5)+0.5,origin3(3,5)+0.5,'Sensor 3','HorizontalAlignment','left','FontSize',8);
l=legend(a,'chessboard','[21]','ICP','proposed with IR data','proposed with depth data');
title(l,'Calibration Approaches');
grid on;
xlabel('X[m]');
ylabel('Y[m]');
zlabel('Z[m]');
grid on; grid minor;
ax=gca;
e_d1=-5:1.0:5;
ax.XTick=[e_d1];
ax.ZTick=[e_d1];
e_d1=-3:1.0:3;
ax.YTick=[e_d1];
xlim([-5 5]);
ylim([-3 3]);
zlim([-5 5]);

%%% 4 spheres were kept on big board with wheels. The strips planesaround sphere should be parallel to tuntable plane.The turnable table gives
%%% spawns differnt positions for the calibration setup.
set='set68_Cyeval';
%%depth images
c={'cam0','cam1','cam2'};
%filerange=[8:22,38:64,75:87,101:112,132:145,156:176,187:204,216:231,246:261,275:288,302:311,322:350,364:397,411:453];%file selecting set of data from 3 sensors
filerange=[8:8+9,38:38+9,75:75+9,101:101+9,132:132+9,156:156+9,187:187+9,216:216+9,246:246+9,275:275+9,302:302+9,322:322+9,364:364+9,411:411+9];
srcFiles = strsplit(num2str(filerange),' ');
dep = cell(3,length(srcFiles));
ir = cell(3,length(srcFiles));
reg = cell(3,length(srcFiles));
Prefixe_folder='C:\Users\Nachu\Documents\thesis papers\journal2017_v3';
for cam=1:3
%srcFiles = dir(['/home/nmohsin/cal_images/' set '/' c{cam} '/depth/*.npy']);  % the folder in which ur images exists
Resultados=[Prefixe_folder '\' set '\' c{cam} '\depth1\'];
Resultados1=[Prefixe_folder '\' set '\' c{cam} '\ir1\'];
mkdir(Resultados);
mkdir(Resultados1);
for i = 1 : length(srcFiles)%4
    %Depth
    filename = strcat([Prefixe_folder '\' set '\' c{cam} '\depth\'],srcFiles{i},'.txt');
    fid=fopen(filename,'r');
    dep{cam,i}=(fscanf(fid,'%u',[512,424]))';
    fclose(fid);
    baseFileName = sprintf('%d.png', i); % e.g. "1.png"
    fullFileName = fullfile(Resultados, baseFileName); 
    imwrite(dep{cam,i}./4500, fullFileName);
    %IR
    filename1 = strcat([Prefixe_folder '\' set '\' c{cam} '\ir\'],srcFiles{i},'.txt');
    fid1=fopen(filename1,'r');
    ir{cam,i}=(fscanf(fid1,'%u',[512,424]))';  
    fclose(fid1);
    fullFileName1 = fullfile(Resultados1, baseFileName); 
    imwrite(ir{cam,i}./65000, fullFileName1);
    %Registerd depth image
    filename2 = strcat([Prefixe_folder '\' set '\' c{cam} '\registered\'],srcFiles{i},'.jpg');
    reg{cam,i} = imread(filename2);
end
end

%% color recognition
regS=cell(3,length(srcFiles));
depS=cell(3,length(srcFiles));
for i=1 : length(srcFiles)
%cam0
I1 = rgb2ycbcr(reg{1,i});

channel1Min = 0.000;
channel1Max = 255.000;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 107.000;
channel2Max = 168.000;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 74.000;
channel3Max = 102.000;

% Create mask based on chosen histogram thresholds
BW1 = (I1(:,:,1) >= channel1Min ) & (I1(:,:,1) <= channel1Max) & ...
    (I1(:,:,2) >= channel2Min ) & (I1(:,:,2) <= channel2Max) & ...
    (I1(:,:,3) >= channel3Min ) & (I1(:,:,3) <= channel3Max);
   Tc = bwconncomp(BW1);
    numPixels = cellfun(@numel,Tc.PixelIdxList);
    Tbw = bwlabel(BW1, 8);
%     figure();imshow(Tbw);
    [~,idx] = max(numPixels);
    BW1=Tbw==idx;
%     figure();imshow(BW1);

% Initialize output masked image based on input image.
regS{1,i} = reg{1,i};
depS{1,i} = dep{1,i};

% Set background pixels where BW is false to zero.
regS{1,i}(repmat(~BW1,[1 1 3])) = 0;
depS{1,i}(~BW1) = 0;
% figure;imshow(depS{1,i}./4500);

%cam1
I2 = rgb2ycbcr(reg{2,i});

% Define thresholds for channel 1 based on histogram settings
channel1Min = 88.000;
channel1Max = 170.000;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 107.000;
channel2Max = 167.000;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 71.000;
channel3Max = 100.000;

% Create mask based on chosen histogram thresholds
BW2 = (I2(:,:,1) >= channel1Min ) & (I2(:,:,1) <= channel1Max) & ...
    (I2(:,:,2) >= channel2Min ) & (I2(:,:,2) <= channel2Max) & ...
    (I2(:,:,3) >= channel3Min ) & (I2(:,:,3) <= channel3Max);

   Tc = bwconncomp(BW2);
    numPixels = cellfun(@numel,Tc.PixelIdxList);
    Tbw = bwlabel(BW2, 8);
%     figure();imshow(Tbw);
    [~,idx] = max(numPixels);
    BW2=Tbw==idx;
%     figure();imshow(BW2);
% Initialize output masked image based on input image.
regS{2,i} = reg{2,i};
depS{2,i} = dep{2,i};

% Set background pixels where BW is false to zero.
regS{2,i}(repmat(~BW2,[1 1 3])) = 0;
depS{2,i}(~BW2) = 0;
% figure;imshow(depS{2,i}./4500);

%cam2
I3 = rgb2ycbcr(reg{3,i});

% Define thresholds for channel 1 based on histogram settings
channel1Min = 0.000;
channel1Max = 249.000;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 99.000;
channel2Max = 167.000;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 69.000;
channel3Max = 110.000;

% Create mask based on chosen histogram thresholds
BW3 = (I3(:,:,1) >= channel1Min ) & (I3(:,:,1) <= channel1Max) & ...
    (I3(:,:,2) >= channel2Min ) & (I3(:,:,2) <= channel2Max) & ...
    (I3(:,:,3) >= channel3Min ) & (I3(:,:,3) <= channel3Max);

   Tc = bwconncomp(BW3);
    numPixels = cellfun(@numel,Tc.PixelIdxList);
    Tbw = bwlabel(BW3, 8);
%     figure();imshow(Tbw);
    [group,idx] = max(numPixels);
    BW3=Tbw==idx;
%     figure();imshow(BW3);
% Initialize output masked image based on input image.
regS{3,i} = reg{3,i};
depS{3,i} = dep{3,i};

% Set background pixels where BW is false to zero.
regS{3,i}(repmat(~BW3,[1 1 3])) = 0;
depS{3,i}(~BW3) = 0;
% figure;imshow(depS{3,i}./4500);
% a=1;
end
%%
%%[Cam0cam1,ca11m2]
fx=[367.214508057,365.7945861816406,365.377105713];
fy=[367.214508057,365.7945861816406,365.377105713];
cx=[256.831695557,255.1743927001953,258.203308105];
cy=[208.951904297,206.6826934814453,205.853302002];

fx1=[316.913150054523,316.805674856439,328.860737158614];
fy1=[325.339383545605,317.960797172921,332.180856351688];
cx1=[244.218211651650,266.259547320396,265.705977025232];
cy1=[183.643422793265,189.056975101018,195.308573501476];

% rel=[5:10:12*10+5];%[5,25,50,70,100];%11,30,56,77,110;135,160,190,218,250;140,170,195,226,260];
rel=[];
for i=5%1:10
    rel=[rel;i:10:(13-1)*10+i]; 
end
[srel,nrel]=size(rel);
Cylinders=cell(3,srel,nrel);%data%
Cylinders1=cell(3,srel,nrel);%data%
CyW_all=cell(3,srel,nrel,n_exp);%transformed data
% CyW_ir=cell(3,srel,nrel);%transformed data
% CyW_chess=cell(3,srel,nrel);
% CyW_exp2=cell(3,srel,nrel);
CyF_all=cell(srel,nrel,n_exp);%fused with depth method;
% CyF_ir=cell(srel,nrel);%fused with depth method;
% CyF_chess=cell(srel,nrel);
% CyF_exp2=cell(srel,nrel);
% RR12=R_12_dep;tt12=t_12_dep;
% RR13=R_13_dep;tt13=t_13_dep;
% RR11=Rii(:,:,1)'*Rii(:,:,1);tt11=Rii(:,:,1)'*tii(:,1)-Rii(:,:,1)'*tii(:,1);
%% Extracting and transform data to 3D form
Clor=jet(nrel);
for i=1:srel
%     figure;
    for j=1:nrel       
        for c=1:3
            [row,col,v] = find(depS{c,rel(i,j)}./1000);
            X=((col-cx(c)).*v)./fx(c);
            Y=((row-cy(c)).*v)./fy(c);
            Z=v;
            
            X1=((col-cx1(c)).*v)./fx1(c);
            Y1=((row-cy1(c)).*v)./fy1(c);
            Z1=v;            
            ptcldA=pointCloud([X(:),Y(:),Z(:)]);
            ptcldA1=pointCloud([X1(:),Y1(:),Z1(:)]);
            %filtering
            ptcldB=pcdenoise(ptcldA);
            Cylinders{c,i,j}=ptcldB.Location;
            
            ptcldB1=pcdenoise(ptcldA1);
            Cylinders1{c,i,j}=ptcldB1.Location;
           
        end
    end              
end

for k=1:n_exp
for i=1:srel
    figure;
    for j=1:nrel      
%         for c=1:3
           if k==1 || k==2
                CyW_all{1,i,j,k}=Cylinders1{1,i,j};
                CyW_all{2,i,j,k}=(bsxfun(@plus,(R_12_all(:,:,k)*Cylinders1{2,i,j}')',t_12_all(:,k)'));
                CyW_all{3,i,j,k}=(bsxfun(@plus,(R_13_all(:,:,k)*Cylinders1{3,i,j}')',t_13_all(:,k)'));
                CyF_all{i,j,k}=[CyW_all{1,i,j,k};CyW_all{2,i,j,k};CyW_all{3,i,j,k}];
                for c=1:3
                    pcshow(CyW_all{c,i,j,k},Clor(j,:),'MarkerSize',5);hold on;
                end
           else
                CyW_all{1,i,j,k}=Cylinders{1,i,j};
                CyW_all{2,i,j,k}=(bsxfun(@plus,(R_12_all(:,:,k)*Cylinders{2,i,j}')',t_12_all(:,k)'));
                CyW_all{3,i,j,k}=(bsxfun(@plus,(R_13_all(:,:,k)*Cylinders{3,i,j}')',t_13_all(:,k)'));
                CyF_all{i,j,k}=[CyW_all{1,i,j,k};CyW_all{2,i,j,k};CyW_all{3,i,j,k}];
                for c=1:3
                    pcshow(CyW_all{c,i,j,k},Clor(j,:),'MarkerSize',5);hold on;
                end
                
           end
           xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)');
    end      
end             
end

% figure();
% pcshow(CyF_all{1,1,4},'r','MarkerSize',5);xlabel('X[m]');ylabel('Y[m]');zlabel('Z[m]');
%% Cylinder modelling after reconstruction
radius_dist=cell(srel,nrel,n_exp);
overall_radius_dist=zeros(srel,nrel,n_exp);
height_dist=zeros(srel,nrel,n_exp);
inliers_dist=cell(srel,nrel,n_exp);
rmse_dist=zeros(srel,nrel,n_exp);
maxDistance=0.010;%% 10mm to inlier
color1=lines(nrel);
for k=1:n_exp
    for i=1:srel
%         figure;
        for j=1:nrel 
        ptCloudIn=pointCloud(CyF_all{i,j,k});
        [model,inlierIndices,OutlierIndices,rmse] = pcfitcylinder(ptCloudIn,maxDistance);
%         pcshow(ptCloudIn,'MarkerSize',2);
%         hold on;
%         plot(model);
        v1=model.Parameters(1:3);
        v2=model.Parameters(4:6);
        overall_radius_dist(i,j,k)=model.Parameters(7);
        radius_dist{i,j,k}=point_to_line(CyF_all{i,j,k},v1,v2);
        height_dist(i,j,k)=model.Height;
        rmse_dist(i,j,k)=rmse;
        inliers_dist{i,j,k}=inlierIndices;
%         xlabel('X[m]');ylabel('Y[m]');zlabel('Z[m]'); 
        end
    end
end
%% Radius error distribution and height error distribution 
e_d=0:0.001:0.5;
error_radius=cell(srel,nrel,n_exp);
overall_error_radius=zeros(srel,nrel,n_exp);error_height=zeros(srel,nrel,n_exp);
actual_radius=0.60/(2*pi);actual_height= 1.0;
err_r=zeros(srel*nrel,n_exp);err_h=zeros(srel*nrel,n_exp);
figure();p=zeros(1,n_exp+1);
for k=1:n_exp%1:n_exp
    for i=1:srel
        for j=1:nrel 
            overall_error_radius(i,j,k)=abs(actual_radius-overall_radius_dist(i,j,k));
            error_height(i,j,k)=abs(actual_height-height_dist(i,j,k));
            err_r((i-1)*nrel+j,k)=overall_error_radius(i,j,k);
            err_h((i-1)*nrel+j,k)=error_height(i,j,k);
            error_radius{i,j,k}=actual_radius-radius_dist{i,j,k};
        end
    end
    c=histogram(err_r(:,k),e_d,'Normalization','cdf');
    c.FaceAlpha=0;
    c.EdgeAlpha=0;
    f=c.Values;x=e_d(1:length(f));
    p(k)=plot(x,f,'--','Color',colors(k,:),'LineWidth',1); hold on

%     figure()
%     histogram(err_r(:,k));
%     figure()
%     histogram(err_h(:,k));
end
l=legend(p(1:n_exp),'chessboard','planar','ICP','proposed with IR data','proposed with depth data');
title(l,'Calibration Approaches');
xlabel('radial error[m]');
ylabel('Error probability');
ylim([0 1]);

modified_error_radius=cell(srel,3,n_exp);
for k=1:n_exp
    for i=1:srel
        for j=1:nrel
            if j==1
                modified_error_radius{i,1,k}=error_radius{i,j,k};
            elseif ismember(j,2:7)
                modified_error_radius{i,2,k}=[ modified_error_radius{i,2,k};error_radius{i,j,k}];
            else ismember(j,8:13) %#ok<SEPEX>
                modified_error_radius{i,3,k}=[modified_error_radius{i,3,k};error_radius{i,j,k}];
            end
        end
    end
end

% radius distribution for each pose (13 poses of cylinder)

figure()
% e_r_bin=-1.00:0.025:1.00;
e_r_bin=0:0.025:1.00;
for k=1:n_exp%1:n_exp
    for i=1:srel
        for j=1:3
            subplot(n_exp,3,(k-1)*3+j);
            histogram(abs(modified_error_radius{i,j,k}),e_r_bin,'Normalization','probability','FaceColor',colors(k,:),'EdgeColor',colors(k,:));
            xlabel('[m]');
        end
    end
end
%% MEan and 95% confidence of the mean value comparison
mean_radius=zeros(srel,nrel,n_exp);std_radius=zeros(srel,nrel,n_exp);
conf_radius=zeros(2,srel,nrel,n_exp);
mark={'s','^','d','v','o',''};
figure()
for k=1:n_exp%1:n_exp
    for i=1:srel
        for j=1:nrel
            mean_radius(i,j,k)=abs(overall_radius_dist(i,j,k));%overall_radius_dist(i,j,k);%mean(radius_dist{i,j,k});
            std_radius(i,j,k)=std(radius_dist{i,j,k});
            SEM = std(radius_dist{i,j,k})/sqrt(length(radius_dist{i,j,k}));               % Standard Error
            ts = tinv([0.050  0.950],length(radius_dist{i,j,k})-1);      % T-Score
            conf_radius(:,i,j,k) = ts*SEM;                      % Confidence Intervals
        end
    end
%     errorbar(1:nrel,mean_radius(:,:,k),reshape(conf_radius(1,:,:,k),[1,nrel]),reshape(conf_radius(2,:,:,k),[1,nrel]),'Color', colors(k,:)); hold on;
errorbar(1:nrel,mean_radius(:,:,k),reshape(std_radius(:,:,k),[1,nrel]),reshape(std_radius(:,:,k),[1,nrel]),'Color', colors(k,:)); hold on;    
p(k)=plot(1:nrel,mean_radius(:,:,k),'Color', colors(k,:),'LineStyle','-');hold on;%,'Marker',mark{k}); hold on 
end

p(n_exp+1)=plot(0:nrel+1,actual_radius.*ones(1,nrel+2),'k--');
l=legend(p,'chessboard','[21]','ICP','proposed with IR data','proposed with depth data','Actual cylindrical radius: 0.095m');
xlabel('Cylinder pose');
ylabel('Radius of estimated cylinder [m]');
ax=gca;
e_d1=0:1:nrel;
ax.XTick=[e_d1];
ax.YTick=0:0.10:1.5;
% grid on;
ylim([0 2.0]);

figure()
A=reshape((mean_radius),13,5);
p1=bar(1:nrel,A);hold on;
errorbar((1:nrel)-0.3,mean_radius(:,:,1),reshape(std_radius(:,:,1),[1,nrel]),reshape(std_radius(:,:,1),[1,nrel]),'.','Color', colors(1,:));hold on;
errorbar((1:nrel)-0.15,mean_radius(:,:,2),reshape(std_radius(:,:,2),[1,nrel]),reshape(std_radius(:,:,2),[1,nrel]),'.','Color', colors(2,:));hold on;
errorbar((1:nrel),mean_radius(:,:,3),reshape(std_radius(:,:,3),[1,nrel]),reshape(std_radius(:,:,3),[1,nrel]),'.','Color', colors(3,:));hold on;
errorbar((1:nrel)+0.15,mean_radius(:,:,4),reshape(std_radius(:,:,4),[1,nrel]),reshape(std_radius(:,:,4),[1,nrel]),'.','Color', colors(4,:));hold on;
errorbar((1:nrel)+0.3,mean_radius(:,:,5),reshape(std_radius(:,:,5),[1,nrel]),reshape(std_radius(:,:,5),[1,nrel]),'.','Color', colors(5,:));hold on;
p2=plot(0:nrel+1,actual_radius.*ones(1,nrel+2),'k--');
l=legend([p1,p2],'chessboard','planar','ICP','proposed with IR data','proposed with depth data','Actual cylindrical radius: 0.095m');
xlabel('Cylinder pose');
ylabel('Radius of estimated cylinder [m]');
ax=gca;ax.YTick=[-0.5:0.2:2.0];

%% Cumulative distribution
e_d=0:0.025:1.2;
err_r1=cell(1,n_exp);
figure();
for k=1:n_exp%1:n_exp
     for j=1:nrel
        for i=1:srel
            err_r1{k}=[err_r1{k};abs(error_radius{i,j,k})];
        end
     end
    c=histogram(err_r1{k},e_d,'Normalization','cdf');
    c.FaceAlpha=0;
    c.EdgeAlpha=0;
    f=c.Values;x=e_d(1:length(f));
    p(k)=plot(x,f,'--','Color',colors(k,:),'LineWidth',1); hold on
end
l=legend(p(1:n_exp),'chessboard','[21]','ICP','proposed with IR data','proposed with depth data');
title(l,'Calibration Approaches');
xlabel('radial error[m]');
ylabel('Error probability');
ax=gca;
e_d1=0:0.020:1;
% xticks(e_d1);
ax.XTick=[e_d1];
ax.XAxis.MinorTickValues = [0:0.025:1];
ylim([0 1]);
xlim([0 0.55]);




