% test galahad_sils
% Nick Gould for GALAHAD productions 19/February/2009

clear A SA control inform

n = 10 ;
b(1:n,1)= 1.0 ;
b(n,1)=2.9;
A(1:n,1:n) = 0.0 ;
for i = 1:n
 A(i,i) = i-(n/2) ;
end
A(1:n,1) = 1.0 ;
A(1,1:n) = 1.0 ;

[ control ] = galahad_sils( 'initial' ) ;

fprintf('solve dense example \n')
[ x, inform ] = galahad_sils( A, b, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - |||Ax-b|| =', norm(A*x - b), '- status =', inform.flag ) )

fprintf('solve sparse example \n')
SA = sparse(A) ;
[ x, inform ] = galahad_sils( SA, b, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - |||Ax-b|| =', norm(A*x - b), '- status =', inform.flag ) )
