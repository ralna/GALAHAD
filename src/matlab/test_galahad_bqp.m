% test galahad_bqp
% Nick Gould for GALAHAD productions 06/November/2009

clear H control

n = 10 ;
f = 1.0 ;
g(1)= -1.0 ;
g(2:n)= 1.0 ;
H(1:n,1:n) = 0.0 ;
for i = 1:n
 H(i,i) = i ;
end
for i = 1:n
 x_l(i) = 0 ;
 x_u(i) =  inf;
end
%control.out = 6 ;
control.print_level = 1 ;
[ x, inform, aux ] = galahad_bqp( H, g, f, x_l, x_u, control )
