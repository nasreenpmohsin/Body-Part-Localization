clc; close all;clear all;
%%Extrinisc comparison of camera poses between Depth and IR approaches
% loading the extrinsic calibration
n_exp=4;%number of sets oof images used
R_12_all_dep=zeros(3,3,4); t_12_all_dep=zeros(3,4);R_13_all_dep=zeros(3,3,4); t_13_all_dep=zeros(3,4);
load('C:\Users\Nachu\Documents\thesis papers\journal2017_v3\set66_Prpexp\CalibrationResults\RT_123_dep_v3.mat');%Proposed with depth
R_12_all_dep(:,:,4)=R_12_dep;t_12_all_dep(:,4)=t_12_dep;R_13_all_dep(:,:,4)=R_13_dep;t_13_all_dep(:,4)=t_13_dep;
load('C:\Users\Nachu\Documents\thesis papers\journal2017_v3\set66_Prpexp\CalibrationResults\RT_2_123_dep_v1.mat');%Proposed with IR
R_12_all_dep(:,:,1)=R_12_dep;t_12_all_dep(:,1)=t_12_dep;R_13_all_dep(:,:,1)=R_13_dep;t_13_all_dep(:,1)=t_13_dep;
load('C:\Users\Nachu\Documents\thesis papers\journal2017_v3\set66_Prpexp\CalibrationResults\RT_1_123_dep_v1.mat');
R_12_all_dep(:,:,2)=R_12_dep;t_12_all_dep(:,2)=t_12_dep;R_13_all_dep(:,:,2)=R_13_dep;t_13_all_dep(:,2)=t_13_dep;
load('C:\Users\Nachu\Documents\thesis papers\journal2017_v3\set66_Prpexp\CalibrationResults\RT_3_123_dep_v1.mat');
R_12_all_dep(:,:,3)=R_12_dep;t_12_all_dep(:,3)=t_12_dep;R_13_all_dep(:,:,3)=R_13_dep;t_13_all_dep(:,3)=t_13_dep;

set='set69_eval1';
%%depth images
c={'cam0','cam1','cam2'};
%filerange=[6:22,38:53,66:81,96:106,112:122,133:140,151:161,176:185,202:217];%file selecting set of data from 3 sensors
filerange=[8,40,70,100,120,138,156,180,210];
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
    [group,idx] = max(numPixels);
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
    [group,idx] = max(numPixels);
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
% rel=[];
% for i=5%1:10
%     rel=[rel;i:10:(13-1)*10+i]; 
% end
rel=1: length(filerange);
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
Clor={[0.5,0,0],[0,0.5,0],[0,0,0.5],[0.5,0,0.5]};
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
colors1=lines(nrel);
for k=1:n_exp
for i=1:srel
    figure;
    for j=1:nrel      
           if k==1 | k==2
                CyW_all{1,i,j,k}=Cylinders1{1,i,j};
                CyW_all{2,i,j,k}=(bsxfun(@plus,(R_12_all_dep(:,:,k)*Cylinders1{2,i,j}')',t_12_all_dep(:,k)'));
                CyW_all{3,i,j,k}=(bsxfun(@plus,(R_13_all_dep(:,:,k)*Cylinders1{3,i,j}')',t_13_all_dep(:,k)'));
                CyF_all{i,j,k}=[CyW_all{1,i,j,k};CyW_all{2,i,j,k};CyW_all{3,i,j,k}];
                for c=1:3
                    pcshow(CyW_all{c,i,j,k},colors1(j,:));hold on;
                end
           else
                CyW_all{1,i,j,k}=Cylinders{1,i,j};
                CyW_all{2,i,j,k}=(bsxfun(@plus,(R_12_all_dep(:,:,k)*Cylinders{2,i,j}')',t_12_all_dep(:,k)'));
                CyW_all{3,i,j,k}=(bsxfun(@plus,(R_13_all_dep(:,:,k)*Cylinders{3,i,j}')',t_13_all_dep(:,k)'));
                CyF_all{i,j,k}=[CyW_all{1,i,j,k};CyW_all{2,i,j,k};CyW_all{3,i,j,k}];
                for c=1:3
                    pcshow(CyW_all{c,i,j,k},colors1(j,:));hold on;
                end
                
           end
           
    end  
        xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)');
        xlim([-1 1]);
        ylim([-2 2]);
        zlim([0 3]);       
    end              
end


% figure();
% pcshow(CyF_all{1,1,4},'r','MarkerSize',5);xlabel('X[m]');ylabel('Y[m]');zlabel('Z[m]');
%%
radius_dist=cell(srel,nrel,n_exp);
overall_radius_dist=zeros(srel,nrel,n_exp);
height_dist=zeros(srel,nrel,n_exp);
rmse_dist=zeros(srel,nrel,n_exp);
overall_centre=zeros(3,srel,nrel,n_exp);
maxDistance=0.01;%% 10mm to inlier
color1=lines(nrel);
for k=1:n_exp
    for i=1:srel
%         figure;
        for j=1:nrel 
        ptCloudIn=pointCloud(CyF_all{i,j,k});
        [model,~,~,rmse] = pcfitcylinder(ptCloudIn,maxDistance);
%         pcshow(ptCloudIn,'MarkerSize',2);
%         hold on;
%         plot(model);
        v1=model.Parameters(1:3);
        v2=model.Parameters(4:6);
        overall_radius_dist(i,j,k)=model.Parameters(7);
        overall_centre(:,i,j,k)=model.Center';
        radius_dist{i,j,k}=point_to_line(CyF_all{i,j,k},v1,v2);
        height_dist(i,j,k)=model.Height;
        rmse_dist(i,j,k)=rmse;
%         xlabel('X[m]');ylabel('Y[m]');zlabel('Z[m]'); 
        end
    end
end

%% Radius error distribution and height error distribution
e_d=0:0.005:1.5;
error_radius=cell(srel,nrel,n_exp);
overall_error_radius=zeros(srel,nrel,n_exp);error_height=zeros(srel,nrel,n_exp);
actual_radius=0.60/(2*pi);actual_height= 1.0;
err_r=zeros(srel*nrel,n_exp);err_h=zeros(srel*nrel,n_exp);
figure();p=zeros(1,5);
colors=lines(10);
for k=1:n_exp;%1:n_exp
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
l=legend(p(1:n_exp),'1','2','3','4');
title(l,'No. of sets of images used for calibration');
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
            else ismember(j,8:13)
                modified_error_radius{i,3,k}=[modified_error_radius{i,3,k};error_radius{i,j,k}];
            end
        end
    end
end

%% radius distribution for each pose (13 poses of cylinder)

% figure()
% e_r_bin=-0.1:0.025:0.1;
% for k=1:n_exp;%1:n_exp
%     for i=1:srel
%         for j=1:3
%             subplot(n_exp,3,(k-1)*3+j);
%             histogram(modified_error_radius{i,j,k},e_r_bin,'Normalization','pdf');
%         end
%     end
% end
%% MEan and 95% confidence of the mean value comparison
mean_radius=zeros(srel,nrel,n_exp);
conf_radius=zeros(2,srel,nrel,n_exp);
figure()
for k=1:n_exp;%1:n_exp
    for i=1:srel
        for j=1:nrel
            mean_radius(i,j,k)=mean(radius_dist{i,j,k});
            SEM = std(radius_dist{i,j,k})/sqrt(length(radius_dist{i,j,k}));               % Standard Error
            ts = tinv([0.050  0.950],length(radius_dist{i,j,k})-1);      % T-Score
            conf_radius(:,i,j,k) = ts*SEM;                      % Confidence Intervals
        end
    end 
    errorbar(1:nrel,mean_radius(:,:,k),reshape(conf_radius(1,:,:,k),[1,nrel]),reshape(conf_radius(2,:,:,k),[1,nrel]),'Color', colors(k,:))
    hold on;
   p(k)= plot(1:nrel,mean_radius(:,:,k),'Color', colors(k,:)); hold on 
end
p(n_exp+1)=plot(1:nrel,actual_radius.*ones(1,nrel),'k--');
l=legend(p,'1','2','3','4','Actual Radius');
title(l,'No. of sets of images used for calibration');
% legend(p(5),'Actual Radius');
xlabel('Cylinder pose');
ylabel('Radius [m]');

%% Cumulative distribution
err_r1=cell(1,n_exp);
figure();
for k=1:n_exp %1:n_exp
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
l=legend(p(1:4),'1','2','3','4');
title(l,'No. of frame sets used for proposed calibration');
xlabel('radial error[m]');
ylabel('Error probability');
ylim([0 1]);
xlim([0 0.7]);
ax=gca;
e_d1=0:0.05:1;
ax.XTick=[e_d1];

%% Evaluation on spatial Shift
shft_cntr=zeros(nrel-1,n_exp);
actual_shift=0.15;
p=zeros(1,5);
figure();
for k=1:n_exp
%     figure();
    for i=1:srel
%         figure;
        for j=1:nrel
%             plot3(overall_centre(1,i,j,k),overall_centre(2,i,j,k),overall_centre(3,i,j,k),'Color',colors(k,:),'Marker','*'); hold on
            if j ~= nrel
                A=[overall_centre(1,i,j,k),overall_centre(2,i,j,k),overall_centre(3,i,j,k)];
                B=[overall_centre(1,i,j+1,k),overall_centre(2,i,j+1,k),overall_centre(3,i,j+1,k)];
                shft_cntr(j,k)=pdist([A;B],'euclidean');
            end
        end
    end
    p(k)=plot(1:nrel-1,shft_cntr(:,k),'Color',colors(k,:)); hold on
end
p(5)=plot(1:nrel-1,actual_shift.*ones(1,nrel-1),'k--')
l=legend(p(1:4),'1','2','3','4');
title(l,'No. of frame sets used for calibration');
legend(p(5),'Nominal Shift');
xlabel('samples'); ylabel('shift from previous pose[m]');
xlim([0 8]);

figure()
bar(abs(actual_shift-mean(shft_cntr)),'FaceColor',colors(2,:));hold on;
% p(5)=plot(0:4,actual_shift.*ones(1,5),'k--');
xlabel('No. of frame sets used for proposed calibration');
ylabel('Overall shift error (m)');
