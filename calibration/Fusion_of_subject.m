close all;
%%% Human motion. ICP.
set='set70_humanmotion';
%%depth images
c={'cam0','cam1','cam2'};
%filerange=[6:22,38:53,66:81,96:106,112:122,133:140,151:161,176:185,202:217];%file selecting set of data from 3 sensors
filerange=[58,17,35];%79;%58,[35,37,39]
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
load('C:\Users\Nachu\Documents\thesis papers\journal2017_v3\set66_Prpexp\CalibrationResults\RT_123_dep_v3.mat');%Proposed with depth

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
Body=cell(3,nrel);%data%
Body1=cell(3,nrel);%data%
Colr=cell(3,nrel);%data%
Colr1=cell(3,nrel);%data%

CyW_all=cell(3,nrel);%transformed data
% CyW_ir=cell(3,srel,nrel);%transformed data
% CyW_chess=cell(3,srel,nrel);
% CyW_exp2=cell(3,srel,nrel);
CyF_all=cell(1,nrel);%fused with depth method;

%% Extracting and transform data to 3D form
Clor={[0.5,0,0],[0,0.5,0],[0,0,0.5],[0.5,0,0.5]};
minDistance=0.5;
%     figure;
    for j=1:nrel       
        for c=1:3
            [row,col,v] = find(depS{c,rel(j)}./1000);
            
            X=((col-cx(c)).*v)./fx(c);
            Y=((row-cy(c)).*v)./fy(c);
            Z=v;
            
            X1=((col-cx1(c)).*v)./fx1(c);
            Y1=((row-cy1(c)).*v)./fy1(c);
            Z1=v;   
            RC=[];RegB=RegM{c,rel(j)};
            for ii=1:length(row)
                RC=[RC;RegB(row(ii),col(ii),1),RegB(row(ii),col(ii),2),RegB(row(ii),col(ii),3)];
            end
            ptcldA=pointCloud([X(:),Y(:),Z(:)],'Color',RC);
            ptcldA1=pointCloud([X1(:),Y1(:),Z1(:)],'Color',RC);
            %filtering
            ptcldBB=pcdenoise(ptcldA,'NumNeighbors',200,'Threshold',0.8);
            maxDistance = 0.09;
%             referenceVector = [0,-1,1];
%             [~,inliers,outliers] = pcfitplane(ptcldB,maxDistance,referenceVector,5);
%      
%             ptCloudWithoutGround = select(ptcldB,outliers,'OutputSize','full');
%             ptcldBB=ptCloudWithoutGround;
%             figure();pcshow(ptcldB);xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)');
%             figure();pcshow(ptcldBB);
%             xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)');
                roi = [-inf,inf;-inf,inf;2.6,inf];
                sampleIndices = findPointsInROI(ptcldBB,roi);

                [model2,inlierIndices,outlierIndices] = pcfitplane(ptcldBB,...
                            maxDistance,'SampleIndices',sampleIndices);
                plane2 = select(ptcldBB,inlierIndices);
                ptcldBB = select(ptcldBB,outlierIndices);
%                 figure();pcshow(ptcldBB);
%             xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)');
            [labels,numClusters] = pcsegdist(ptcldBB,minDistance);
%              figure();pcshow(ptcldBB.Location,labels);
%             colormap(hsv(numClusters))
%             title('Point Cloud Clusters');
            ptcldBB=pointCloud(ptcldBB.Location(labels==mode(labels),:),'Color',ptcldBB.Color(labels==mode(labels),:));
            if c~=3
            ptcldBB=pcdenoise(ptcldBB,'NumNeighbors',200,'Threshold',0.8);
            end
            Body1{c,j}=ptcldBB.Location;
            Colr1{c,j}=ptcldBB.Color;
            gridStep = 0.09;
            ptcldBB = pcdownsample(ptcldBB,'gridAverage',gridStep);
%             ptcldBB=pcdownsample(ptcldBB,'random',0.10);
            Body{c,j}=ptcldBB.Location;
            Colr{c,j}=ptcldBB.Color;
%             figure();pcshow(ptcldBB);
            
%             [labels,numClusters] = pcsegdist(pointCloud(Body{c,j}),0.2);
% %             figure();pcshow(Body{c,j},labels);
%             colormap(hsv(numClusters))
%             title('Point Cloud Clusters');
%             Body{c,j}=Body{c,j}(labels==mode(labels),:);
%             ptcldB1=pcdenoise(ptcldA1,'NumNeighbors',50,'Threshold',0.7);
%             Body1{c,j}=ptcldB1.Location;
%             figure();pcshow(ptcldB);
           
        end
    end             
    
%% Fusion

% for j=1:nrel
%     figure()
%     ptcld1=pointCloud(Body{1,j},'Color',Colr{1,j});
%     ptcld2=pointCloud(Body{2,j},'Color',Colr{2,j});
%     ptcld3=pointCloud(Body{3,j},'Color',Colr{3,j});
%     
%     CyW_all{1,j}=Body1{1,j};
%     CyW_all{2,j}=(bsxfun(@plus,(R_12_dep*Body1{2,j}')',t_12_dep'));
%     CyW_all{3,j}=(bsxfun(@plus,(R_13_dep*Body1{3,j}')',t_13_dep'));
%     CyF_all{j}=[CyW_all{1,j};CyW_all{2,j};CyW_all{3,j}];
%                 for c=1:3
%                     pcshow(CyW_all{c,j},Clor{c},'MarkerSize',3);hold on;
%                 end
%         xlabel('X[m]');ylabel('Y[m]');zlabel('Z[m]');
%         xlim([-2 2]);ylim([-2 2]);zlim([-0 3]);
% end

for j=1:1%nrel
    figure()
    ptcld1=pointCloud(Body{1,j},'Color',Colr{1,j});
    ptcld2=pointCloud(Body{2,j},'Color',Colr{2,j});
    ptcld3=pointCloud(Body{3,j},'Color',Colr{3,j});
    
    CyW_all{1,j}=Body{1,j};
    CyW_all{2,j}=(bsxfun(@plus,(R_12_dep*Body{2,j}')',t_12_dep'));
    CyW_all{3,j}=(bsxfun(@plus,(R_13_dep*Body{3,j}')',t_13_dep'));
    CyF_all{j}=[CyW_all{1,j};CyW_all{2,j};CyW_all{3,j}];
    lar=zeros(1,3);% largest number of points
    rmse=zeros(1,3);
                for c=1:3
                    lar(c)=length(CyW_all{c,j});
%                     pcshow(CyW_all{c,j},Clor{c},'MarkerSize',3);hold on;
                    [~,~,~,rmse(c)] = pcfitplane(pointCloud(CyW_all{c,j}),0.01);
                end
                [~,tm]=min(rmse);
                pcshow(CyW_all{tm,j},Clor{tm},'MarkerSize',3);
        xlabel('X[m]');ylabel('Y[m]');zlabel('Z[m]');
        xlim([-2 2]);ylim([-2 2]);zlim([-0 3]);
        ncolF=round(((CyW_all{tm,j}(:,1).*fx(c))./CyW_all{tm,j}(:,3))+cx(c));
        nrowF=round(((CyW_all{tm,j}(:,2).*fy(c))./CyW_all{tm,j}(:,3))+cy(c));
        nwpt2D=[ncolF,nrowF];
        newDep=zeros(size(D));
        for it=1:length(nrowF)
           newDep(nrowF(it),ncolF(it))=CyW_all{tm,j}(it,3);
        end
        
%      [model] = pcfitplane(pointCloud(CyW_all{tm,j}),0.01);
%      hold on;
%      plot(model);
%      (0,c,?b)T , (?c,0,a)T and (b,?a,0)T;
%      norm1=model.Normal;
%      v1=[-n(3),0,n(1)];%[0,n(3),-n(2)];%v2=[-n(3),0,n(1)];v3=[n(2),-n(1),0]; Any one of these vectors 
%      v2=cross(n,v1);
%      uv=null(norm1(:).');
%      nwpt2D  = CyW_all{tm,j}*uv;
%      cenPt2D= mean(nwpt2D);
%      cen2DIndx = dsearchn(nwpt2D,cenPt2D);
%      newZ = point_plane_shortest_dist_vec(CyW_all{tm,j},model.Parameters);
%      nwpt3D=CyW_all{tm,j}*(uv*uv.');
%      nwpt3D(:,3)=newZ;
%      cenPt3D=mean(nwpt3D);
%      cen3DIndx = dsearchn(nwpt3D,cenPt3D);
%      figure();
%      plot(nwpt2D(:,1),nwpt2D(:,2),'*');
%      figure()
%      pcshow(nwpt3D);hold on
%      plot3(cenPt3D(1),cenPt3D(2),cenPt3D(3),'r*');hold on;
%      plot3(nwpt3D(cen3DIndx,1),nwpt3D(cen3DIndx,2),nwpt3D(cen3DIndx,3),'b*');
%      pcshow(CyF_all{j});hold on
%      plot(model)
% 
%      [G1,Geodist]=network_building(nwpt3D,nwpt2D);
%      Geodist1 = sparse(Geodist);
%      G2=graph(Geodist1);
%     [ TR,D,E] = shortestpathtree(G2,cen3DIndx,'all');
%     [~,AGEX(1)]=max(D);
%     G2=addedge(G2,cen3DIndx,AGEX(1),0.001);
%     [ TR,D,E] = shortestpathtree(G2,cen3DIndx,'all');
%     [~,AGEX(2)]=max(D);
%     G2=addedge(G2,cen3DIndx,AGEX(2),0.001);
%     [ TR,D,E] = shortestpathtree(G2,cen3DIndx,'all');
%     [~,AGEX(3)]=max(D);
%     G2=addedge(G2,cen3DIndx,AGEX(3),0.001);
%     [ TR,D,E] = shortestpathtree(G2,cen3DIndx,'all');
%     [~,AGEX(4)]=max(D);
%     
%              figure()
%      pcshow(nwpt3D,'MarkerSize',15);hold on
%      plot3(nwpt3D(AGEX,1),nwpt3D(AGEX,2),nwpt3D(AGEX,3),'r*');hold on;
%      plot3(nwpt3D(cen3DIndx,1),nwpt3D(cen3DIndx,2),nwpt3D(cen3DIndx,3),'b*');
% %%     sel=Geodist(cen3DIndx,:);
%      [X,Y]=meshgrid(sort(nwpt2D(:,1))',sort(nwpt2D(:,2))');
%      Z=NaN(size(X));
%      for k1= 1:size(X,2)
%          for k2=1:size(X,1)
%             for k3=1:length(nwpt2D)
%              if X(k1,k2)==nwpt2D(k3,1) && Y(k1,k2)==nwpt2D(k3,2)
%                  if isfinite(D(k3))
%                     Z(k1,k2)=3-D(k3);
%                  end
%              end
%            end
%          end
%      end
%      figure();imshow(Z);
     
end