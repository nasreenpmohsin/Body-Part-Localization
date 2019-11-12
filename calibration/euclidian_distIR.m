function F=euclidian_distIR(Z1)
global C1
for i=1:9
if(Z1(i)>90)
    Z1(i)=360-Z1(i);
elseif(Z1(i)<-90)
    Z1(i)=360+Z1(i);
end
end  
Eu=reshape(Z1(1:9),[3,3]);
% R=[0.223454272846637,-0.0985478415648413,-0.969719810496564;-0.0420148725166465,0.992977315962425,-0.110592949465456;0.973808471124580,0.0654551213196480,0.217744549091451];
 R=SpinCalc('EA321toDCM',Eu,0.1,0);%reshape(Z1(1:9),[3,3]);%
%  R=R';
t = Z1(10:18);
t=reshape(t,[3,3]);%[Z1(4);Z1(5);Z1(6)];%[Z1(10);Z1(11);Z1(12)];%
Ct=Z1(19:end);
num_c=length(Ct)/(3*4);
Ct=reshape(Ct,[3,4,num_c]);
% Ct=reshape(Ct,3,4,[]);

F=[];
for i=1:3
   for j=1:4
    for k=1:num_c
        F=[F ;norm(Ct(:,j,k)-R(:,:,i)*C1(:,j,i,k)-t(:,i))];
    end
   end
end

% for i=2
%     for j=1:num_c
%         Ctt=R'*(Ct(:,j)-t);
%         F=[F ;norm(Ctt-C(:,j,i))];
%     end
% end
end