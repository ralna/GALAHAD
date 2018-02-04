% test galahad_gltr
% Nick Gould for GALAHAD productions 3/March/2009

clear H M

n = 10 ;
[ control ]   = galahad_gltr( 'initial' )
control.out = 6 ;
control.print_level = 1 ;
control.unitm = 0 ;
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
%M = 2 * M
% [ x, obj, inform ] = galahad_gltr( H, c, f, radius )
%[ x, obj ] = galahad_gltr( H, c, f, radius, control, M )
[ x, obj, inform ] = galahad_gltr( H, c, f, radius, control, M )
galahad_gltr( 'final' )
