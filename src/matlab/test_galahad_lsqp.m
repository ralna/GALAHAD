% test galahad_lsqp
% Nick Gould for GALAHAD productions 17/February/2010

clear A SA control inform

m = 3 ;
n = 5 ;
f = 1.0 ;
g(1:n)= 1.0 ;
for i = 1:m
 for j = 1:n
   if j < i
     A(i,j) = 2 ;
   else
     A(i,j) = 1 ;
   end
 end
end
for i = 1:m
 c_l(i) = 1 ;
 c_u(i) =  3 * n ;
end
for i = 1:n
 x_l(i) = 0 ;
 x_u(i) =  100.0 ;
end
for i = 1:n
 w(i) = 1.0 ;
 x0(i) = 10.0 ;
end

control.out = 6 ;
control.print_level = 0 ;
%control.generate_sif_file = 1 ;
%control.sif_file_device = 6 ;

fprintf('solve dense examples \n')
[ x, inform, aux ] = galahad_lsqp( g, f, A, c_l, c_u, x_l, x_u, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )
[ x, inform, aux ] = galahad_lsqp( g, f, A, c_l, c_u, x_l, x_u, ...
                                   w, x0, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )

fprintf('solve sparse examples \n')
SA = sparse(A) ;
[ x, inform, aux ] = galahad_lsqp( g, f, SA, c_l, c_u, x_l, x_u, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )
[ x, inform, aux ] = galahad_lsqp( g, f, SA, c_l, c_u, x_l, x_u, ...
                                   w, x0, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )
