% test galahad_glrt
% Nick Gould for GALAHAD productions 5/March/2009

clear H M SH SM control inform

n = 10 ;
f = 1.0 ;
p = 3 ;
sigma = 1.0 ;
c(1:n)= 1.0 ;
H(1:n,1:n) = 0.0 ;
for i = 1:n
 H(i,i) = i-(n/2) ;
end
M(1:n,1:n) = 0.0 ;
for i = 1:n
  M(i,i) = 1.0 ;
end
% [ x, obj, inform ] = galahad_glrt( H, c, f, p, sigma )
%[ x, obj ] = galahad_glrt( H, c, f, p, sigma, control, M )

[ control ]   = galahad_glrt( 'initial' ) ;
control.out = 0 ;

fprintf('solve dense examples \n')
[ x, obj, inform ] = galahad_glrt( H, c, f, p, sigma, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )
galahad_glrt( 'final' )
control.unitm = 0 ;
[ x, obj, inform ] = galahad_glrt( H, c, f, p, sigma, control, M ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )
galahad_glrt( 'final' )

fprintf('solve sparse examples \n')
SH = sparse(H) ;
SM = sparse(M) ;
control.unitm = 1 ;
[ x, obj, inform ] = galahad_glrt( SH, c, f, p, sigma, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )
galahad_glrt( 'final' )
control.unitm = 0 ;
[ x, obj, inform ] = galahad_glrt( SH, c, f, p, sigma, control, SM ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )
galahad_glrt( 'final' )
