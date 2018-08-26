% test galahad_bqp
% Nick Gould for GALAHAD productions 06/November/2009

clear H SH control inform

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
control.print_level = 0 ;

fprintf('solve dense example \n')
[ x, inform, aux ] = galahad_bqp( H, g, f, x_l, x_u, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )

fprintf('solve sparse example \n')
SH = sparse(H) ;
[ x, inform, aux ] = galahad_bqp( SH, g, f, x_l, x_u, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )
