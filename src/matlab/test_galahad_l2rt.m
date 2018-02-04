% test galahad_l2rt
% Nick Gould for GALAHAD productions 5/March/2009

clear A control

m = 10 ;
n = 2 ;
control.out = 0 ;
f = 1.0 ;
p = 3.0 ;
sigma = 100.0 ;
mu = 0.01 ;
b(1:m)= 1.0 ;
for i = 1:m
 for j = 1:n
  A(i,j) = i+j ;
 end
end
% [ x, obj, inform ] = galahad_l2rt( A, b, p, sigma, mu )
%[ x, obj ] = galahad_l2rt( A, b, p, sigma, mu, control )
[ x, obj, inform ] = galahad_l2rt( A, b, p, sigma, mu, control )
