% test galahad_dqp
% Nick Gould for GALAHAD productions 01/August/2012

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
control.PSLS_control.definite_linear_solver = 'sils' ;
control.PSLS_control.preconditioner = 1 ;
control.PSLS_control.out = 6 ;
control.PSLS_control.print_level = 1 ;
control.out = 6 ;
control.print_level = 0 ;

fprintf('solve dense example \n')
[ x, inform, aux ] = galahad_dqp( H, g, f, A, c_l, c_u, x_l, x_u, control );
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - dqp: optimal f =', inform.obj, '- status =', inform.status ) )

fprintf('solve sparse example \n')
SH = sparse(H) ;
SA = sparse(A) ;
[ x, inform, aux ] = galahad_dqp( SH, g, f, SA, c_l, c_u, x_l, x_u, control );
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - dqp: optimal f =', inform.obj, '- status =', inform.status ) )
