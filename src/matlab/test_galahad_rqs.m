% test galahad_rqs
% Nick Gould for GALAHAD productions 19/February/2009

clear A H M SA SH SM control inform

n = 10 ;

f = 1.0 ;
p = 3.0 ;
sigma = 1.0 ;
c(1)= 0.0 ;
c(2:n)= 1.0 ;
H(1:n,1:n) = 0.0 ;
for i = 1:n
 H(i,i) = i-(n/2) ;
end
M(1:n,1:n) = 0.0 ;
for i = 1:n
  M(i,i) = 1.0 ;
end

[ control ] = galahad_rqs( 'initial' );
control.out = 6 ;
control.IR_control.acceptable_residual_relative = 0.9 ;
control.IR_control.acceptable_residual_absolute = 0.9 ;

fprintf('solve dense examples \n')
[ x, inform ] = galahad_rqs( H, c, f, p, sigma, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - rqs: optimal f =', inform.obj, '- status =', inform.status ) )
[ x, inform ] = galahad_rqs( H, c, f, p, sigma, control, M ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - rqs: optimal f =', inform.obj, '- status =', inform.status ) )

m = 2 ;
for i = 1:m
 for j = 1:n
  A(i,j) = i+j ;
 end
end

control.print_level = 0;
[ x, inform ] = galahad_rqs( H, c, f, p, sigma, control, M, A ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - rqs: optimal f =', inform.obj, '- status =', inform.status ) )

fprintf('solve sparse examples \n')
SH = sparse(H) ;
SA = sparse(A) ;
SM = sparse(M) ;
%control.print_level = 0;
[ x, inform ] = galahad_rqs( SH, c, f, p, sigma, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - rqs: optimal f =', inform.obj, '- status =', inform.status ) )
[ x, inform ] = galahad_rqs( SH, c, f, p, sigma, control, SM ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - rqs: optimal f =', inform.obj, '- status =', inform.status ) )

control.print_level = 0;
[ x, inform ] = galahad_rqs( SH, c, f, p, sigma, control, SM, SA ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - rqs: optimal f =', inform.obj, '- status =', inform.status ) )
