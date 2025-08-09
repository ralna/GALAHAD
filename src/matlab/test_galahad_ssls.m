% test galahad_ssls
% Nick Gould for GALAHAD productions 9/August/2025

clear A H C SA SH SC b d x y control inform

m = 4 ;
n = 10 ;

fprintf('solve dense example \n')

[ control ] = galahad_ssls( 'initial' ) ;
control.out = 6 ;

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

%  form and factorize the block matrix

% [ inform ] = galahad_ssls( 'form_and_factorize', H, A, C, control ) ;
 galahad_ssls( 'form_and_factorize', H, A, C, control ) ;

d(1:m,1) = 0.0 ;
for j = 1:n
 b(j) = j ;
 for i = 1:m
  d(i) = d(i) + i + j ;
 end
end

%  solve the block system

 [ x, y, inform ] = galahad_ssls( 'solve', b, d, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - ssls: ||Kx-r|| =', norm([ H * x + A' * y - b' ;  A * x + C * y - d ]), ...
  '- status =', inform.status ) )
[ inform ] = galahad_ssls( 'final' ) ;

fprintf('solve sparse example \n')

%  solve the sparse system

SA = sparse(A) ;
SH = sparse(H) ;
SC = sparse(C) ;

[ control ] = galahad_ssls( 'initial' ) ;
control.out = 6 ;

[ inform ] = galahad_ssls( 'form_and_factorize', SH, SA, SC, control ...
                           ) ;
[ x, y, inform ] = galahad_ssls( 'solve', b, d, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - ssls: ||Kx-r|| =', norm([ H * x + A' * y - b' ;  A * x + C * y - d ]), ...
  '- status =', inform.status ) )
[ inform ] = galahad_ssls( 'final' ) ;
