function [G1,Geodist]=network_building(Pt,pt2D)

pt_len=length(Pt); mt=0;d_2D=0;
Geodist=zeros(pt_len,pt_len);
for i=1:pt_len
    for j= 1:pt_len
        if Geodist(i,j) ~=0
            continue
        end
        if i == j
            Geodist(i,j)=0;
            continue;
        end
        mt=pdist([Pt(i,:);Pt(j,:)],'euclidean');
        d_2D=pdist([pt2D(i,:);pt2D(j,:)],'euclidean');
        if d_2D<0.3 && mt<=0.08
                Geodist(i,j)=mt;
                Geodist(j,i)=mt;
            
        end  
    end
end
G1=graph(Geodist);
% Pt=[Xn,Yn,Zn];
% E1=unique(sort([tri1(:,2) tri1(:,1); tri1(:,3) tri1(:,2); tri1(:,1) tri1(:,3)],2),'rows');
% nE1=[];Geodist=[];
% figure();pcshow([Xn,Yn,Zn],[0.5 0.5 0.2],'MarkerSize',5);hold on;
% xlabel('X');ylabel('Y');zlabel('Z');
% count=0;
% for i=1:size(E1,1)
%     fl=0;
%     px1=[round(nrow(E1(i,1))),round(ncol(E1(i,1)))];
%     px2=[round(nrow(E1(i,2))),round(ncol(E1(i,2)))];    
%     P1=Pt(E1(i,1),:);P2=Pt(E1(i,2),:);
%   if ~isnan(P1(3))&&~isnan(P2(3))
%     d_2D=pdist([px1(2),px1(1);px2(2),px2(1)],'euclidean');
%     mt=pdist([P1;P2],'euclidean');
%     if d_2D<=1
%         if mt<=0.08
% %             disp('hit');
%             nE1=[nE1;E1(i,:)]; 
%             Geodist=[Geodist;mt];  
%             if rem(count,10)==0
%             plot3(Pt(E1(i,:),1),Pt(E1(i,:),2),Pt(E1(i,:),3),'g'); hold on;
%             end
%              count=count+1;
%         end
%     else
% %         coefficients = polyfit([px1(1), px2(1)], [px1(2), px2(2)], 1);
%                     xx = [px1(2), px2(2)];
%                     yy = [px1(1), px2(1)];
%                     c = [[1; 1]  xx(:)]\yy(:);                        % Calculate Parameter Vector
%                     coefficients = [c(2),c(1)];
%         buff_min=0;
%         for j=min(px1(2),px2(2)):max(px1(2),px2(2))
%             if j== px1(2) 
%                 continue;
%             end
%             if j==px2(2)
%                 continue; 
%             end
%             k=ceil(coefficients(1)*j+coefficients(2));
%             if newDepF(k,j)==0
%                 fl=1; 
%             end;
%             if newDepF(k,j)>buff_min
%             buff_min=newDepF(k,j);
%             end
%          end
%         if buff_min<min(P1(3),P2(3)) && buff_min~=0 && fl==0
%             nE1=[nE1;E1(i,:)]; 
%             Geodist=[Geodist;mt];  
%            
%           if rem(count,10)==0
%             plot3(Pt(E1(i,:),1),Pt(E1(i,:),2),Pt(E1(i,:),3),'g'); hold on;
%           end
%            count=count+1;
%         end
%     end
%    end
% %  
% %     F=neighbors_2pixels(px1,px2);
% %     if F && mt<0.2 && ~isnan(P1(3))&&~isnan(P2(3))%%  %&&  mt2<0.3 &&  mt1<0.3 
% %         nE1=[nE1;E1(i,:)]; 
% %         Geodist=[Geodist;mt];
% %     end
% end
% G1=graph(nE1(:,1),nE1(:,2),Geodist);