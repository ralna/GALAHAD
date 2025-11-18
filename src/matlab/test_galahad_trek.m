% test galahad_trek
% Nick Gould for GALAHAD productions 15/February/2009

clear H S SA SH SS control inform

n = 10 ;
control.out = 0 ;

radius = 10.0 ;
c(1)= 0.0 ;
c(2:n)= 1.0 ;
H(1:n,1:n) = 0.0 ;
for i = 1:n
 H(i,i) = i-(n/2) ;
end
S(1:n,1:n) = 0.0 ;
for i = 1:n
  S(i,i) = 1.0 ;
end

fprintf('solve dense examples \n')
[ x, inform ] = galahad_trek( H, c, radius, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - trek: optimal f =', inform.obj, '- status =', inform.status ) )
[ x, inform ] = galahad_trek( H, c, radius, control, S ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - trek: optimal f =', inform.obj, '- status =', inform.status ) )

fprintf('solve sparse examples \n')
SH = sparse(H) ;
SS = sparse(S) ;
[ x, inform ] = galahad_trek( SH, c, radius, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - trek: optimal f =', inform.obj, '- status =', inform.status ) )
[ x, inform ] = galahad_trek( SH, c, radius, control, SS ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - trek: optimal f =', inform.obj, '- status =', inform.status ) )
