% test galahad_lpa
% Nick Gould for GALAHAD productions 12/November/2020

clear A SA control inform

%m = 800 ;
m = 1 ;
n = 2 * m ;
f = 1.0 ;
g(1:n)= 1.0 ;
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

[ control ] = galahad_lpa( 'initial' ) ;

%control

%control.SBLS_control.definite_linear_solver = 'ma57' ;
%control.SBLS_control.out = 6 ;
%control.SBLS_control.print_level = 1 ;
control.out = 6 ;
control.print_level = 0 ;

fprintf('solve dense example \n')
[ x, inform, aux ] = galahad_lpa( g, f, A, c_l, c_u, x_l, x_u, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - lpa: optimal f =', inform.obj, '- status =', inform.status ) )

fprintf('solve sparse example \n')
SA = sparse(A) ;
[ x, inform, aux ] = galahad_lpa( g, f, SA, c_l, c_u, x_l, x_u, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - lpa: optimal f =', inform.obj, '- status =', inform.status ) )

galahad_lpa( 'final' )
