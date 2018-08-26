% test galahad_lstr
% Nick Gould for GALAHAD productions 5/March/2009

clear A SA control inform

m = 10 ;
n = 2 ;
control.steihaug_toint = 0 ;
f = 1.0 ;
radius = 10.0 ;
radius = 0.1 ;
b(1:m)= 1.0 ;
for i = 1:m
 for j = 1:n
  A(i,j) = i+j ;
 end
end
% [ x, obj, inform ] = galahad_lstr( A, b, radius )
%[ x, obj ] = galahad_lstr( A, b, radius, control )

fprintf('solve dense example \n')
[ x, obj, inform ] = galahad_lstr( A, b, radius, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', obj, '- status =', inform.status ) )

fprintf('solve sparse example \n')
SA = sparse(A) ;
[ x, obj, inform ] = galahad_lstr( SA, b, radius, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', obj, '- status =', inform.status ) )
