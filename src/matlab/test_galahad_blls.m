% test galahad_blls
% Nick Gould for GALAHAD productions 12/December/2020

clear A SA control inform

m = 10 ;
n = 5 ;
b(1)= -1.0 ;
b(2:m)= 1.0 ;
A(1:m,1:n) = 0.0 ;
for i = 1:n
 A(i,i) = i ;
 A(n+i,i) = 1 ;
 A(2*i,i) = 1 ;
end
for i = 1:n
 x_l(i) = 0 ;
 x_u(i) =  inf;
end
control.out = 6 ;
%control.print_level = 1 ;

fprintf('solve dense example \n')
[ x, inform, aux ] = galahad_blls( A, b, x_l, x_u, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - blls: optimal f =', inform.obj, '- status =', inform.status ) )

fprintf('solve sparse example \n')
SA = sparse(A) ;
[ x, inform, aux ] = galahad_blls( SA, b, x_l, x_u, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - blls: optimal f =', inform.obj, '- status =', inform.status ) )
