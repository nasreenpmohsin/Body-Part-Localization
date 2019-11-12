function dist = point_plane_shortest_dist_vec(xyz,norm)
% Point to plane shortest distance vector
% The plane is defined by: ax+by+cz+d = 0
% The point is defined by: x, y, and z

dist = (norm(1).*xyz(:,1)+norm(2).*xyz(:,2)+norm(3).*xyz(:,3)+norm(4).*ones(size(xyz(:,1)))) ./ sqrt(norm(1)^2+norm(2)^2+norm(3)^2);
% N = [a; b; c]/norm([a; b; c]);
% v = N.*dist;

end