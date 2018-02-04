% test galahad_lsrt
% Nick Gould for GALAHAD productions 5/March/2009

clear A control

m = 10 ;
n = 2 ;
control.out = 0 ;
f = 1.0 ;
p = 3.0 ;
sigma = 10.0 ;
b(1:m)= 1.0 ;
for i = 1:m
 for j = 1:n
  A(i,j) = i+j ;
 end
end
% [ x, obj, inform ] = galahad_lsrt( A, b, p, sigma )
%[ x, obj ] = galahad_lsrt( A, b, p, sigma, control )
[ x, obj, inform ] = galahad_lsrt( A, b, p, sigma, control ) ;
