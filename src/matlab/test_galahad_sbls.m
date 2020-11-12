% test galahad_sbls
% Nick Gould for GALAHAD productions 15/February/2010

clear A H C SA SH SC b d control inform

m = 4 ;
n = 10 ;

fprintf('solve dense example \n')

[ control ] = galahad_sbls( 'initial' ) ;
control.out = 6 ;
control.IR_control.acceptable_residual_relative = 0.9 ;
control.IR_control.acceptable_residual_absolute = 0.9 ;
control.preconditioner = 2 ;
control.get_norm_residual = 1 ;

H(1:n,1:n) = 0.0 ;
for i = 1:n
 H(i,i) = i ;
end
for i = 1:m
 for j = 1:n
  A(i,j) = i+j ;
 end
end
C(1:m,1:m) = 0.0 ;
for i = 1:m
  C(i,i) = 1.0 ;
end

%  form and factorize the preconditioner

% [ inform ] = galahad_sbls( 'form_and_factorize', H, A, C, control ) ;
 galahad_sbls( 'form_and_factorize', H, A, C, control ) ;

d(1:m,1) = 0.0 ;
for j = 1:n
 b(j) = j ;
 for i = 1:m
  d(i) = d(i) + i + j ;
 end
end

%  solve the system

 [ x, y, inform ] = galahad_sbls( 'solve', b, d, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - sbls: ||Kx-r|| =', norm([ H * x + A' * y - b' ;  A * x + C * y - d ]), ...
  '- status =', inform.status ) )
[ inform ] = galahad_sbls( 'final' ) ;

fprintf('solve sparse example \n')

%  solve the sparse system

SA = sparse(A) ;
SH = sparse(H) ;
SC = sparse(C) ;

[ control ] = galahad_sbls( 'initial' ) ;
control.out = 6 ;
control.IR_control.acceptable_residual_relative = 0.9 ;
control.IR_control.acceptable_residual_absolute = 0.9 ;
control.preconditioner = 2 ;
control.get_norm_residual = 1 ;

[ inform ] = galahad_sbls( 'form_and_factorize', SH, SA, SC, control ...
                           ) ;
[ x, y, inform ] = galahad_sbls( 'solve', b, d, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - sbls: ||Kx-r|| =', norm([ H * x + A' * y - b' ;  A * x + C * y - d ]), ...
  '- status =', inform.status ) )
[ inform ] = galahad_sbls( 'final' ) ;
