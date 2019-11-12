close all;
%%% Human motion. ICP.
set='set71';
%%depth images
c={'cam0','cam1','cam2'};
%filerange=[6:22,38:53,66:81,96:106,112:122,133:140,151:161,176:185,202:217];%file selecting set of data from 3 sensors
filerange=[32];%79;%58,[35,37,39]
srcFiles = strsplit(num2str(filerange),' ');
dep = cell(3,length(srcFiles));
ir = cell(3,length(srcFiles));
reg = cell(3,length(srcFiles));
mask = cell(3,length(srcFiles));
depS = cell(3,length(srcFiles));
Prefixe_folder='C:\Users\Nachu\Documents\thesis papers\journal2017_v3';
for cam=1:3
%srcFiles = dir(['/home/nmohsin/cal_images/' set '/' c{cam} '/depth/*.npy']);  % the folder in which ur images exists
Resultados=[Prefixe_folder '\' set '\' c{cam} '\depth1\'];
Resultados1=[Prefixe_folder '\' set '\' c{cam} '\ir1\'];
Resultados2=[Prefixe_folder '\' set '\' c{cam} '\mask\'];
mkdir(Resultados);
mkdir(Resultados1);mkdir(Resultados2);
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
    %masked Depth image
    filename3 = strcat([Prefixe_folder '\' set '\' c{cam} '\depth_mask\'],srcFiles{i},'.txt');
    fid3=fopen(filename3,'r');
    mask{cam,i}=(fscanf(fid3,'%u',[512,424]))';
    fclose(fid3);
    baseFileName = sprintf('%d.png', i); % e.g. "1.png"
    fullFileName3 = fullfile(Resultados2, baseFileName); 
    imwrite(mask{cam,i}, fullFileName3);
    depS{cam,i}=mask{cam,i};%bitand(dep{cam,i},mask{cam,i});
    mask{cam,i}=mask{cam,i}>0;
    RegM{cam,i}=bsxfun(@times, reg{cam,i}, cast(mask{cam,i}, 'like', reg{cam,i}));
%     figure();imshow(depS{cam,i}./4500);
end
end
rel=1;
BW=cell(3,length(rel));
depS=dep;
for c=1:3
    for i=1:rel
figure();
imshow(dep{c,i});
BW{c,i}=roipoly();
figure ();
depS{c,i}(~BW{c,i})=0;
imshow (depS{c,i});
    end
end
c;
%%
Body=cell(3,nrel);%data%
Body1=cell(3,nrel);%data%
for c=1:3
    figure();
    imshow(depS{c,i}./4500);
    [row,col,v] = find(depS{c,rel(j)}./1000);       
            X=((col-cx(c)).*v)./fx(c);
            Y=((row-cy(c)).*v)./fy(c);
            Z=v;
            X1=((col-cx1(c)).*v)./fx1(c);
            Y1=((row-cy1(c)).*v)./fy1(c);
            Z1=v;
     Body{c,i}=[X,Y,Z];
     Body1{c,i}=[X1,Y1,Z1];
     figure();
     pcshow(Body{c,i});
     
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


%%
% rel=[5:10:12*10+5];%[5,25,50,70,100];%11,30,56,77,110;135,160,190,218,250;140,170,195,226,260];
% rel=[];
% for i=5%1:10
%     rel=[rel;i:10:(13-1)*10+i]; 
% end
% rel=1: length(filerange);
% [srel,nrel]=size(rel);
% Body=cell(3,nrel);%data%
% Body1=cell(3,nrel);%data%
% Colr=cell(3,nrel);%data%
% Colr1=cell(3,nrel);%data%
% 
% CyW_all=cell(3,nrel);%transformed data
% % CyW_ir=cell(3,srel,nrel);%transformed data
% % CyW_chess=cell(3,srel,nrel);
% % CyW_exp2=cell(3,srel,nrel);
% CyF_all=cell(nrel);%fused with depth method;
% 
% %% Extracting and transform data to 3D form
% Clor={[0.5,0,0],[0,0.5,0],[0,0,0.5],[0.5,0,0.5]};
% minDistance=0.5;
% %     figure;
%     for j=1:nrel       
%         for c=1:3
%             [row,col,v] = find(depS{c,rel(j)}./1000);
%             
%             X=((col-cx(c)).*v)./fx(c);
%             Y=((row-cy(c)).*v)./fy(c);
%             Z=v;
%             
%             X1=((col-cx1(c)).*v)./fx1(c);
%             Y1=((row-cy1(c)).*v)./fy1(c);
%             Z1=v;   
%             RC=[];RegB=RegM{c,rel(j)};
%             for ii=1:length(row)
%                 RC=[RC;RegB(row(ii),col(ii),1),RegB(row(ii),col(ii),2),RegB(row(ii),col(ii),3)];
%             end
%             ptcldA=pointCloud([X(:),Y(:),Z(:)],'Color',RC);
%             ptcldA1=pointCloud([X1(:),Y1(:),Z1(:)],'Color',RC);
%             %filtering
%             ptcldBB=pcdenoise(ptcldA,'NumNeighbors',200,'Threshold',0.8);
%             maxDistance = 0.01;
% %             referenceVector = [0,-1,1];
% %             [~,inliers,outliers] = pcfitplane(ptcldB,maxDistance,referenceVector,5);
% %      
% %             ptCloudWithoutGround = select(ptcldB,outliers,'OutputSize','full');
% %             ptcldBB=ptCloudWithoutGround;
% %             figure();pcshow(ptcldB);xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)');
% %             figure();pcshow(ptcldBB);
% %             xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)');
%                 roi = [-inf,inf;-inf,inf;2.6,inf];
%                 sampleIndices = findPointsInROI(ptcldBB,roi);
% 
%                 [model2,inlierIndices,outlierIndices] = pcfitplane(ptcldBB,...
%                             maxDistance,'SampleIndices',sampleIndices);
%                 plane2 = select(ptcldBB,inlierIndices);
%                 ptcldBB = select(ptcldBB,outlierIndices);
% %                 figure();pcshow(ptcldBB);
% %             xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)');
%             [labels,numClusters] = pcsegdist(ptcldBB,minDistance);
% %              figure();pcshow(ptcldBB.Location,labels);
% %             colormap(hsv(numClusters))
% %             title('Point Cloud Clusters');
%             ptcldBB=pointCloud(ptcldBB.Location(labels==mode(labels),:),'Color',ptcldBB.Color(labels==mode(labels),:));
%             if c~=3
%             ptcldBB=pcdenoise(ptcldBB,'NumNeighbors',200,'Threshold',0.8);
%             end
%             Body1{c,j}=ptcldBB.Location;
%             Colr1{c,j}=ptcldBB.Color;
%             gridStep = 0.03;
%             ptcldBB = pcdownsample(ptcldBB,'gridAverage',gridStep);
%             Body{c,j}=ptcldBB.Location;
%             Colr{c,j}=ptcldBB.Color;
% %             figure();pcshow(ptcldBB);
%             
% %             [labels,numClusters] = pcsegdist(pointCloud(Body{c,j}),0.2);
% % %             figure();pcshow(Body{c,j},labels);
% %             colormap(hsv(numClusters))
% %             title('Point Cloud Clusters');
% %             Body{c,j}=Body{c,j}(labels==mode(labels),:);
% %             ptcldB1=pcdenoise(ptcldA1,'NumNeighbors',50,'Threshold',0.7);
% %             Body1{c,j}=ptcldB1.Location;
% %             figure();pcshow(ptcldB);
%            
%         end
%     end              
%% ICP
tform12=cell(1,nrel);tform13=cell(1,nrel);
tform21=cell(1,nrel);tform23=cell(1,nrel);
tform31=cell(1,nrel);tform32=cell(1,nrel);
m12=cell(1,nrel);m13=cell(1,nrel);
m21=cell(1,nrel);m23=cell(1,nrel);
m31=cell(1,nrel);m32=cell(1,nrel);
for j=1:nrel
    ptcld1=pointCloud(Body{1,j});%,'Color',Colr{1,j});
    ptcld2=pointCloud(Body{2,j})%,'Color',Colr{2,j});
    ptcld3=pointCloud(Body{3,j})%,'Color',Colr{3,j});
    ptcld1.Normal = pcnormals(ptcld1,20);
    ptcld2.Normal = pcnormals(ptcld2,30);
    ptcld3.Normal = pcnormals(ptcld3,20);
%     tempc=affine3d([-0.383286164513739,-0.315501619025022,0.868072833629147,0;0.371480830942510,0.807824127088059,0.457626673104484,0;-0.845632135348822,0.497874389868575,-0.192425007674119,0;1.73040627564365,-0.927521537579616,2.71023426472541,1]);
    
    ptcld1_1=pointCloud(Body1{1,j});%,'Color',Colr1{1,j});
    ptcld1_2=pointCloud(Body1{2,j});%,'Color',Colr1{2,j});
    ptcld1_3=pointCloud(Body1{3,j});%,'Color',Colr1{3,j});
    [tform12{j},m12{j}] = pcregistericp(ptcld2,ptcld1,'Extrapolate',true,'Metric','pointToPlane','Tolerance',[0.001, 0.005],'MaxIterations',100,'InitialTransform',tempc12);
    [tform13{j},m13{j}] = pcregistericp(ptcld3,ptcld1,'Extrapolate',true,'Metric','pointToPlane','Tolerance',[0.001, 0.005],'MaxIterations',100,'InitialTransform',tempc13);
%     [tform21{j},m21{j}] = pcregistericp(ptcld1,ptcld2,'Extrapolate',true,'Metric','pointToPlane','Tolerance',[0.001, 0.005],'MaxIterations',100);
%     [tform23{j},m23{j}] = pcregistericp(ptcld3,ptcld2,'Extrapolate',true,'Metric','pointToPlane','Tolerance',[0.001, 0.005],'MaxIterations',100);
%     [tform31{j},m31{j}] = pcregistericp(ptcld1,ptcld3,'Extrapolate',true,'Metric','pointToPlane','Tolerance',[0.001, 0.005],'MaxIterations',100);
%     [tform32{j},m32{j}] = pcregistericp(ptcld2,ptcld3,'Extrapolate',true,'Metric','pointToPlane','Tolerance',[0.001, 0.005],'MaxIterations',100);
%     m12{j}=pctransform(ptcld1_2,tform12{j});
%     m13{j}=pctransform(ptcld1_3,tform13{j});
%     m21{j}=pctransform(ptcld1_1,tform21{j});
%     m23{j}=pctransform(ptcld1_3,tform23{j});
%     m31{j}=pctransform(ptcld1_1,tform31{j});
%     m32{j}=pctransform(ptcld1_2,tform32{j});
% [tform222,m12] = pcregisterndt(ptcld2,ptcld1,0.1);
    %     tform2 = invert(tform222);
%     disp(tform12{j}.T);
% %     tform222=affine3d([-0.432191371696383,-0.217656429930671,0.875120732665518,0;0.401237969220719,0.915497349758051,0.483002079309550,0;-0.807600588339825,0.338364125013009,-0.0295414055457370,0;1.68434758673553,-0.527781937898316,1.24513913548475,1]);
%     m12{j} = pctransform(ptcld1_2,tform12{j});
%     m12 = pctransform(m12,tform_180);
%     A=tform_180.T';B=tform222.T'
%     C=A(1:3,1:3)*B(1:3,1:3)
%     CI=eye(4)
%     CI(1:3,1:3)=C
%     CI(4,:)=tform222.T(4,:)
%     tform222=affine3d(CI);
%     m12 = pctransform(ptcld12,tform222);
    
%     [tform13{j},m13{j}] = pcregistericp(ptcld3,ptcld1,'Extrapolate',true,'Metric','pointToPlane');
% [tform333,m13] = pcregisterndt(ptcld3,ptcld1,0.1);
    %     tform3 = invert(tform333);
%     disp(tform13{j}.T);
%     m13{j} = pctransform(ptcld1_3,tform13{j});
%     m31 = pctransform(ptcld1_1,invert(tform13{j}));
    figure();
    pcshow(Body1{1,j},[1,0,0],'MarkerSize',3);hold on;
    pcshow(m12{j}.Location,[0,1,0],'MarkerSize',3);hold on;    
    pcshow(m13{j}.Location,[0,0,1],'MarkerSize',3);hold on;
    xlabel('X[m]');ylabel('Y[m]');zlabel('Z[m]');
     
        figure();
    pcshow(Body1{2,j},[0,1,0],'MarkerSize',3);hold on;
    pcshow(m21{j}.Location,[1,0,0],'MarkerSize',3);hold on;    
    pcshow(m23{j}.Location,[0,0,1],'MarkerSize',3);hold on;
    xlabel('X[m]');ylabel('Y[m]');zlabel('Z[m]');
    
        figure();
    pcshow(Body1{3,j},[0,0,1],'MarkerSize',3);hold on;
    pcshow(m31{j}.Location,[1,0,0],'MarkerSize',3);hold on;    
    pcshow(m32{j}.Location,[0,1,0],'MarkerSize',3);hold on;
    xlabel('X[m]');ylabel('Y[m]');zlabel('Z[m]');
    
  
    
end
R_12_icp=tform12{1}.T(1:3,1:3)';R_13_icp=tform13{1}.T(1:3,1:3)';
t_12_icp=tform12{1}.T(4,1:3)';t_13_icp=tform13{1}.T(4,1:3)';
