% test galahad_glrt
% Nick Gould for GALAHAD productions 5/March/2009

clear H M

n = 10 ;
control.out = 0 ;
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
[ x, obj, inform ] = galahad_glrt( H, c, f, p, sigma, control, M )
galahad_gltr( 'final' )
