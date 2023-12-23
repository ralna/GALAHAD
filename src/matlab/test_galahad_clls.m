% test galahad_clls
% Nick Gould for GALAHAD productions 18/December/2023

clear A_o A control inform

n = 5 ;
o = n + 1 ;
m = 2 * n ;
b(1)= -1.0 ;
b(2:o)= 1.0 ;
A_o(1:o,1:n) = 0.0 ;
for i = 1:n
 A_o(i,i) = i ;
 A_o(n+1,i) = 1 ;
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

control.SLS_control.definite_linear_solver = 'ma57' ;
%control.SBLS_control.out = 6 ;
%control.SLS_control.print_level = 1 ;
control.out = 6 ;
%control.print_level = 1 ;
[ x, inform, aux ] = galahad_clls( A_o, b, 0.0, A, c_l, c_u, x_l, x_u, ...
                                   control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - clls: optimal f =', inform.obj, '- status =', inform.status ) )

%  solve the sparse system

fprintf('solve sparse example \n')

SA = sparse(A) ;
SA_o = sparse(A_o) ;

[ control ] = galahad_clls( 'initial' ) ;
control.print_level = 0 ;
[ x, inform, aux ] = galahad_clls( 'existing', A_o, b, 0.0, A, c_l, c_u, ...
                                    x_l, x_u, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - clls: optimal f =', inform.obj, '- status =', inform.status ) )
galahad_clls( 'final' )

%  solve the example with explicit weights

fprintf('solve example with weights\n')

w(1:o)= 2.0 ;

[ control ] = galahad_clls( 'initial' ) ;
control.print_level = 0 ;
[ x, inform, aux ] = galahad_clls( 'existing', A_o, b, 0.0, A, c_l, c_u, ...
                                    x_l, x_u, w, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - clls: optimal f =', inform.obj, '- status =', inform.status ) )
galahad_clls( 'final' )
