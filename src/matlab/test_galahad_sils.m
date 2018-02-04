% test galahad_sils
% Nick Gould for GALAHAD productions 19/February/2009

clear A

n = 10 ;
b(1:n,1)= 1.0 ;
b(n,1)=2.9;
A(1:n,1:n) = 0.0 ;
for i = 1:n
 A(i,i) = i-(n/2) ;
end
A(1:n,1) = 1.0 ;
A(1,1:n) = 1.0 ;
[ x, inform ] = galahad_sils( A, b )
