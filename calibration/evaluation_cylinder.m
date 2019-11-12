%%%% 4 spheres were kept on big board with wheels. The strips planesaround sphere should be parallel to tuntable plane.The turnable table gives
%%%% spawns differnt positions for the calibration setup.
% set='set59_eval';
% %%depth images
% c={'cam0','cam1','cam2'};
% filerange=1:286;%file selecting set of data from 3 sensors
% srcFiles = strsplit(num2str(filerange),' ');
% dep = cell(3,length(srcFiles));
% ir = cell(3,length(srcFiles));
% reg = cell(3,length(srcFiles));
% Prefixe_folder='C:\Users\Nachu\Documents\thesis papers\journal2017_v3';
% for cam=1:3
% %srcFiles = dir(['/home/nmohsin/cal_images/' set '/' c{cam} '/depth/*.npy']);  % the folder in which ur images exists
% Resultados=[Prefixe_folder '\' set '\' c{cam} '\depth1\'];
% Resultados1=[Prefixe_folder '\' set '\' c{cam} '\ir1\'];
% mkdir(Resultados);
% mkdir(Resultados1);
% for i = 1 : length(srcFiles)%4
%     %Depth
%     filename = strcat([Prefixe_folder '\' set '\' c{cam} '\depth\'],srcFiles{i},'.txt');
%     fid=fopen(filename,'r');
%     dep{cam,i}=(fscanf(fid,'%u',[512,424]))';
%     fclose(fid);
%     baseFileName = sprintf('%d.png', i); % e.g. "1.png"
%     fullFileName = fullfile(Resultados, baseFileName); 
%     imwrite(dep{cam,i}./4500, fullFileName);
%     %IR
%     filename1 = strcat([Prefixe_folder '\' set '\' c{cam} '\ir\'],srcFiles{i},'.txt');
%     fid1=fopen(filename1,'r');
%     ir{cam,i}=(fscanf(fid1,'%u',[512,424]))';  
%     fclose(fid1);
%     fullFileName1 = fullfile(Resultados1, baseFileName); 
%     imwrite(ir{cam,i}./65000, fullFileName1);
%     %Registerd depth image
%     filename2 = strcat([Prefixe_folder '\' set '\' c{cam} '\registered\'],srcFiles{i},'.jpg');
%     reg{cam,i} = imread(filename2);
% end
% end

% %% color recognition
% regS=cell(3,length(srcFiles));
% depS=cell(3,length(srcFiles));
% for i=1 : length(srcFiles)
% %cam0
% I1 = rgb2ycbcr(reg{1,i});
% 
% channel1Min = 0.000;
% channel1Max = 255.000;
% 
% % Define thresholds for channel 2 based on histogram settings
% channel2Min = 107.000;
% channel2Max = 168.000;
% 
% % Define thresholds for channel 3 based on histogram settings
% channel3Min = 74.000;
% channel3Max = 102.000;
% 
% % Create mask based on chosen histogram thresholds
% BW1 = (I1(:,:,1) >= channel1Min ) & (I1(:,:,1) <= channel1Max) & ...
%     (I1(:,:,2) >= channel2Min ) & (I1(:,:,2) <= channel2Max) & ...
%     (I1(:,:,3) >= channel3Min ) & (I1(:,:,3) <= channel3Max);
%    Tc = bwconncomp(BW1);
%     numPixels = cellfun(@numel,Tc.PixelIdxList);
%     Tbw = bwlabel(BW1, 8);
% %     figure();imshow(Tbw);
%     [group,idx] = max(numPixels);
%     BW1=Tbw==idx;
% %     figure();imshow(BW1);
% 
% % Initialize output masked image based on input image.
% regS{1,i} = reg{1,i};
% depS{1,i} = dep{1,i};
% 
% % Set background pixels where BW is false to zero.
% regS{1,i}(repmat(~BW1,[1 1 3])) = 0;
% depS{1,i}(~BW1) = 0;
% % figure;imshow(depS{1,i}./4500);
% 
% %cam1
% I2 = rgb2ycbcr(reg{2,i});
% 
% % Define thresholds for channel 1 based on histogram settings
% channel1Min = 88.000;
% channel1Max = 170.000;
% 
% % Define thresholds for channel 2 based on histogram settings
% channel2Min = 107.000;
% channel2Max = 167.000;
% 
% % Define thresholds for channel 3 based on histogram settings
% channel3Min = 71.000;
% channel3Max = 100.000;
% 
% % Create mask based on chosen histogram thresholds
% BW2 = (I2(:,:,1) >= channel1Min ) & (I2(:,:,1) <= channel1Max) & ...
%     (I2(:,:,2) >= channel2Min ) & (I2(:,:,2) <= channel2Max) & ...
%     (I2(:,:,3) >= channel3Min ) & (I2(:,:,3) <= channel3Max);
% 
%    Tc = bwconncomp(BW2);
%     numPixels = cellfun(@numel,Tc.PixelIdxList);
%     Tbw = bwlabel(BW2, 8);
% %     figure();imshow(Tbw);
%     [group,idx] = max(numPixels);
%     BW2=Tbw==idx;
% %     figure();imshow(BW2);
% % Initialize output masked image based on input image.
% regS{2,i} = reg{2,i};
% depS{2,i} = dep{2,i};
% 
% % Set background pixels where BW is false to zero.
% regS{2,i}(repmat(~BW2,[1 1 3])) = 0;
% depS{2,i}(~BW2) = 0;
% % figure;imshow(depS{2,i}./4500);
% 
% %cam2
% I3 = rgb2ycbcr(reg{3,i});
% 
% % Define thresholds for channel 1 based on histogram settings
% channel1Min = 0.000;
% channel1Max = 249.000;
% 
% % Define thresholds for channel 2 based on histogram settings
% channel2Min = 100.000;
% channel2Max = 166.000;
% 
% % Define thresholds for channel 3 based on histogram settings
% channel3Min = 77.000;
% channel3Max = 106.000;
% 
% % Create mask based on chosen histogram thresholds
% BW3 = (I3(:,:,1) >= channel1Min ) & (I3(:,:,1) <= channel1Max) & ...
%     (I3(:,:,2) >= channel2Min ) & (I3(:,:,2) <= channel2Max) & ...
%     (I3(:,:,3) >= channel3Min ) & (I3(:,:,3) <= channel3Max);
% 
%    Tc = bwconncomp(BW3);
%     numPixels = cellfun(@numel,Tc.PixelIdxList);
%     Tbw = bwlabel(BW3, 8);
% %     figure();imshow(Tbw);
%     [group,idx] = max(numPixels);
%     BW3=Tbw==idx;
% %     figure();imshow(BW3);
% % Initialize output masked image based on input image.
% regS{3,i} = reg{3,i};
% depS{3,i} = dep{3,i};
% 
% % Set background pixels where BW is false to zero.
% regS{3,i}(repmat(~BW3,[1 1 3])) = 0;
% depS{3,i}(~BW3) = 0;
% % figure;imshow(depS{3,i}./4500);
% % a=1;
% end
%%
%%[Cam0cam1,ca11m2]
fx=[367.214508057,365.7945861816406,365.377105713];
fy=[367.214508057,365.7945861816406,365.377105713];
cx=[256.831695557,255.1743927001953,258.203308105];
cy=[208.951904297,206.6826934814453,205.853302002];

rel=[5,25,50,70,100];%11,30,56,77,110;135,160,190,218,250;140,170,195,226,260];
[srel,nrel]=size(rel);
Cylinders=cell(3,srel,nrel);%data%
CyW_dep=cell(3,srel,nrel);%transformed data
CyW_ir=cell(3,srel,nrel);%transformed data
CyW_chess=cell(3,srel,nrel);
CyW_exp2=cell(3,srel,nrel);
CyF_dep=cell(srel,nrel);%fused with depth method;
CyF_ir=cell(srel,nrel);%fused with depth method;
CyF_chess=cell(srel,nrel);
CyF_exp2=cell(srel,nrel);
RR12=R_12_dep;tt12=t_12_dep;
RR13=R_13_dep;tt13=t_13_dep;
RR11=Rii(:,:,1)'*Rii(:,:,1);tt11=Rii(:,:,1)'*tii(:,1)-Rii(:,:,1)'*tii(:,1);
Clor={[0.5,0,0],[0,0.5,0],[0,0,0.5],[0.5,0,0.5]};
for i=1:srel
%     figure;
    for j=1:nrel       
        for c=1:3
            [row,col,v] = find(depS{c,rel(i,j)}./1000);
            X=((col-cx(c)).*v)./fx(c);
            Y=((row-cy(c)).*v)./fy(c);
            Z=v;
            Cylinders{c,i,j}=[X(:),Y(:),Z(:)];
        end
    end
              
end
% figure;
for i=1:srel
    figure;
    for j=1:nrel      
        for c=1:3
            if (c==1)
            
%                 CyW{1,i,j}=bsxfun(@plus,RR11*Cylinders{c,i,j}',tt11);
                W1=(bsxfun(@plus,Rii(:,:,1)*Cylinders{1,i,j}',tii(:,1)));%Rii_P1'*Rii_Pc+Rii_P1'*tii_Pc-Rii_P1'*tii_P1
                 CyW_dep{1,i,j}=(bsxfun(@plus,(Rii(:,:,1)'*W1),-Rii(:,:,1)'*tii(:,1)));
                 
                 CyW_ir{1,i,j}=Cylinders{c,i,j}';
                 CyW_chess{1,i,j}=Cylinders{c,i,j}';
                 CyW_exp2{1,i,j}=Cylinders{c,i,j}';
%                  W0=(bsxfun(@plus,Rii(:,:,1)*[0 ;0 ;0],tii(:,1)));
%                  W2=(bsxfun(@plus,Rii(:,:,1)'*W0,-Rii(:,:,1)'*tii(:,1)));
%                 break;
            elseif c==2
                 CyW_dep{2,i,j}=bsxfun(@plus,R_12_dep*Cylinders{c,i,j}',t_12_dep);
                 CyW_ir{2,i,j}=bsxfun(@plus,R_12_ir*Cylinders{c,i,j}',t_12_ir);
                 grr=-R_12_exp1*t_12_exp1;
                 CyW_chess{2,i,j}=bsxfun(@plus,R_12_exp1*Cylinders{c,i,j}',grr);%R_12_exp1*[t_12_exp1(1);t_12_exp1(2);t_12_exp1(3)/10]);
                 CyW_exp2{2,i,j}=bsxfun(@plus,R_12_exp2*Cylinders{c,i,j}',t_12_exp2./1000);
%                     CyW{2,i,j}=bsxfun(@minus,Cylinders{2,i,j},tt12');
%                  CyW{2,i,j}=(RR12'*CyW{2,i,j}')';
%                  break;
                else 
                    CyW_dep{3,i,j}=bsxfun(@plus,R_13_dep*Cylinders{c,i,j}',t_13_dep);
                    CyW_ir{3,i,j}=bsxfun(@plus,R_13_ir*Cylinders{c,i,j}',t_13_ir);
                    CyW_chess{3,i,j}=bsxfun(@plus,(R_12_exp1*R_23_exp1)*Cylinders{c,i,j}',-R_12_exp1*t_12_exp1-R_12_exp1*R_23_exp1*t_23_exp1);
                    CyW_exp2{3,i,j}=bsxfun(@plus,(R_23_exp2_1)*Cylinders{c,i,j}',(t_23_exp2./1000));
%                     CyW{c,i,j}=bsxfun(@minus,Cylinders{c,i,j},tt12');
%                  CyW(c,i,j)=(RR12'*CyW{c,i,j}')';
%                     break;

            end
            Wdep=(bsxfun(@plus,(Rii(:,:,c)*Cylinders{c,i,j}')',tii(:,c)'));%Rii_P1'*Rii_Pc+Rii_P1'*tii_Pc-Rii_P1'*tii_P1
            Wdep=(bsxfun(@plus,(Rii(:,:,1)'*Wdep')',(-Rii(:,:,1)'*tii(:,1))'));
            Wir_=(bsxfun(@plus,(RiiIR(:,:,c)*Cylinders{c,i,j}')',tiiIR(:,c)'));%Rii_P1'*Rii_Pc+Rii_P1'*tii_Pc-Rii_P1'*tii_P1
            Wir=(bsxfun(@plus,(RiiIR(:,:,1)'*Wir_')',(-RiiIR(:,1)'*tiiIR(:,1))'));
            
            %(bsxfun(@plus,(Rii*Spheres1{i,2}')',tii'));
           %fusion cam0 as reference
        CyF_dep{i,j}=[CyF_dep{i,j};Wdep];
        CyF_ir{i,j}=[CyF_ir{i,j};CyW_ir{c,i,j}'];
        CyF_chess{i,j}=[CyF_chess{i,j};CyW_chess{c,i,j}'];
        CyF_exp2{i,j}=[CyF_exp2{i,j};CyW_exp2{c,i,j}'];
%             CyF{i,j}=[CyF{i,j};CyW{c,i,j}'];
        pcshow([Wdep(:,1),Wdep(:,2),Wdep(:,3)],Clor{1},'MarkerSize',5); hold on;
% %         pcshow([Wir(:,1),Wir(:,2),Wir(:,3)],Clor{2},'MarkerSize',5); hold on;
        pcshow(CyW_ir{c,i,j}',Clor{2},'MarkerSize',5);hold on;
        pcshow(CyW_chess{c,i,j}',Clor{3},'MarkerSize',5);hold on;
        pcshow(CyW_exp2{c,i,j}',Clor{4},'MarkerSize',5);hold on;
        xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)');
        end
%         hold on;pcshow(CyF_chess{i,j},Clor{i},'MarkerSize',5);hold on;      
    end
              
end
figure();
pcshow(CyF_dep{1,1},'r','MarkersSize',5);xlabel('X');ylabel('Y');zlabel('Z');
    


%

