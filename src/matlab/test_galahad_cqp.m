% test galahad_cqp
% Nick Gould for GALAHAD productions 06/November/2009

clear A H SA SH control inform

%m = 800 ;
m = 1 ;
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
end
for i = 1:m
 c_l(i) = i ;
 c_u(i) =  5 * i ;
end
for i = 1:n
 x_l(i) = 0 ;
 x_u(i) =  inf;
end

fprintf('solve dense example \n')

control.SBLS_control.definite_linear_solver = 'ma57' ;
%control.SBLS_control.out = 6 ;
%control.SBLS_control.print_level = 1 ;
control.out = 6 ;
%control.print_level = 1 ;
[ x, inform, aux ] = galahad_cqp( H, g, f, A, c_l, c_u, x_l, x_u, ...
                                  control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - cqp: optimal f =', inform.obj, '- status =', inform.status ) )

%  solve the sparse system

fprintf('solve sparse example \n')

SA = sparse(A) ;
SH = sparse(H) ;

[ control ] = galahad_cqp( 'initial' ) ;
control.print_level = 0 ;
[ x, inform, aux ] = galahad_cqp( 'existing', H, g, f, A, c_l, c_u, ...
                                  x_l, x_u, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - cqp: optimal f =', inform.obj, '- status =', inform.status ) )
galahad_cqp( 'final' )
