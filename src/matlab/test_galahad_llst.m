% test galahad_llst
% Nick Gould for GALAHAD productions 2/March/2014

clear A S SA SS control inform

m = 10 ;
n = 2 ;
radius = 1.0 ;
b(1:m)= 1.0 ;
for i = 1:m
 for j = 1:n
  A(i,j) = i+j ;
 end
end
S(1:n,1:n) = 0.0 ;
for i = 1:n
  S(i,i) = 1.0 ;
end
[ control ] = galahad_llst( 'initial' ) ;
%[ x, obj, inform ] = galahad_llst( A, b, radius, control )

fprintf('solve dense examples \n')
[ x, obj, inform ] = galahad_llst( A, b, radius, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - llst: optimal f =', obj, '- status =', inform.status ) )
[ x, obj, inform ] = galahad_llst( A, b, radius, control, S );
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - llst: optimal f =', obj, '- status =', inform.status ) )

fprintf('solve sparse examples \n')
SA = sparse(A) ;
SS = sparse(S) ;
[ x, obj, inform ] = galahad_llst( SA, b, radius, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - llst: optimal f =', obj, '- status =', inform.status ) )
[ x, obj, inform ] = galahad_llst( SA, b, radius, control, SS ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - llst: optimal f =', obj, '- status =', inform.status ) )
