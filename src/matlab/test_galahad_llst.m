% test galahad_llst
% Nick Gould for GALAHAD productions 2/March/2014

clear A S control

m = 10 ;
n = 2 ;
radius = 10.0 ;
b(1:m)= 1.0 ;
for i = 1:m
 for j = 1:n
  A(i,j) = i+j ;
 end
end
S(1:n,1:n) = 0.0 ;
for i = 1:n
  S(i,i) = 1.0 ;
end
[ control ] = galahad_llst( 'initial' ) ;
%[ x, obj, inform ] = galahad_llst( A, b, radius, control )
[ x, obj, inform ] = galahad_llst( A, b, radius, control, S )
