% test galahad_wcp
% Nick Gould for GALAHAD productions 06/November/2008

clear A

%m = 800 ;
m = 5 ;
n = 2 * m ;
for i = 1:m
 for j = 1:n
  A(i,j) = i+j ;
 end
end
for i = 1:m
 c_l(i) = i ;
 c_u(i) =  5 * i ;
end
for i = 1:n
 x_l(i) = 0 ;
 x_u(i) =  inf;
end
[ x, inform, aux ] = galahad_wcp( A, c_l, c_u, x_l, x_u )
