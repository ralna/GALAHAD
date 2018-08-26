% test galahad_gltr
% Nick Gould for GALAHAD productions 3/March/2009

clear H M SH SM control inform

n = 10 ;
f = 1.0 ;
radius = 10.0 ;
c(1:n)= 1.0 ;
H(1:n,1:n) = 0.0 ;
for i = 1:n
 H(i,i) = i-(n/2) ;
end
M(1:n,1:n) = 0.0 ;
for i = 1:n
  M(i,i) = 1.0 ;
end
%M = 2 * M ;
% [ x, obj, inform ] = galahad_gltr( H, c, f, radius )
%[ x, obj ] = galahad_gltr( H, c, f, radius, control, M )

[ control ]   = galahad_gltr( 'initial' ) ;
control.out = 6 ;
control.print_level = 0 ;

fprintf('solve dense examples \n')
[ x, obj, inform ] = galahad_gltr( H, c, f, radius, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', obj, '- status =', inform.status ) )
galahad_gltr( 'final' )
control.unitm = 0 ;
[ x, obj, inform ] = galahad_gltr( H, c, f, radius, control, M ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', obj, '- status =', inform.status ) )
galahad_gltr( 'final' )

fprintf('solve sparse examples \n')
SH = sparse(H) ;
SM = sparse(M) ;
control.unitm = 1 ;
[ x, obj, inform ] = galahad_gltr( SH, c, f, radius, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', obj, '- status =', inform.status ) )
galahad_gltr( 'final' )
control.unitm = 0 ;
[ x, obj, inform ] = galahad_gltr( SH, c, f, radius, control, SM ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', obj, '- status =', inform.status ) )
galahad_gltr( 'final' )
