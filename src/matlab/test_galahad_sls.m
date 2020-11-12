% test galahad_sls
% Nick Gould for GALAHAD productions 10/November/2020

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

[ control ] = galahad_sls( 'initial' ) ;

fprintf('solve dense example \n')
[ x, inform ] = galahad_sls( A, b, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - sls: ||Ax-b|| =', norm(A*x - b), '- status =', inform.status ) )

fprintf('solve sparse example \n')
SA = sparse(A) ;
[ x, inform ] = galahad_sls( SA, b, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - sls: ||Ax-b|| =', norm(A*x - b), '- status =', inform.status ) )

fprintf('sophisticated calls \n')

[ control ] = galahad_sls( 'initial' ) ;
[ inform ] = galahad_sls( 'factor', A, control ) ;
[ x, inform ] = galahad_sls( 'solve', b ) ;
[ inform ] = galahad_sls( 'final' ) ;

disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - sls: ||Ax-b|| =', norm(A*x - b), '- status =', inform.status ) )

%fprintf('different solver \n')
%[ x, inform ] = galahad_sls( A, b, control, 'ma57' ) ;
%control.ordering=0;
%[ x, inform ] = galahad_sls( A, b, control, 'ma97' ) ;
%disp( sprintf( '%s %13.6e %s %2.0f', ...
%  ' - sls: ||Ax-b|| =', norm(A*x - b), '- status =', inform.status ) )

%fprintf('different solver - ma57\n')

%[ control ] = galahad_sls( 'initial', 'ma57' ) ;
%[ inform ] = galahad_sls( 'factor', A, control ) ;
%[ x, inform ] = galahad_sls( 'solve', b ) ;
%[ inform ] = galahad_sls( 'final' ) ;

%disp( sprintf( '%s %13.6e %s %2.0f', ...
%  ' - sls: ||Ax-b|| =', norm(A*x - b), '- status =', inform.status ) )

%fprintf('different solver - ma97\n')

%[ control ] = galahad_sls( 'initial', 'ma57' ) ;
%[ inform ] = galahad_sls( 'factor', A, control ) ;
%[ x, inform ] = galahad_sls( 'solve', b ) ;
%[ inform ] = galahad_sls( 'final' ) ;

%disp( sprintf( '%s %13.6e %s %2.0f', ...
%  ' - sls: ||Ax-b|| =', norm(A*x - b), '- status =', inform.status ) )
%fprintf('end of test \n')
