% test galahad_eqp
% Nick Gould for GALAHAD productions 18/February/2010

clear A H

%m = 800 ;
m = 5 ;
n = 2 * m ;
f = 1.0 ;
g(1:n)= 1.0 ;
H(1:n,1:n) = 0.0 ;
for i = 1:n
 H(i,i) = i ;
end
for i = 1:m
 for j = 1:n
  A(i,j) = i+j ;
 end
  A(i,i) = 0.0 ;
end
for i = 1:m
 c(i) =  5 * i ;
end
control.out = 6 ;
control.print_level = 1 ;
%[ x, inform ] = galahad_eqp( H, g, f, A, c, control )
[ x, inform, aux ] = galahad_eqp( H, g, f, A, c, control )
