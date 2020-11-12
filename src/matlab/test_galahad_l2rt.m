% test galahad_l2rt
% Nick Gould for GALAHAD productions 5/March/2009

clear A SA control inform

m = 10 ;
n = 2 ;
f = 1.0 ;
p = 3.0 ;
sigma = 100.0 ;
mu = 0.01 ;
b(1:m)= 1.0 ;
for i = 1:m
 for j = 1:n
  A(i,j) = i+j ;
 end
end

control.out = 0 ;

% [ x, obj, inform ] = galahad_l2rt( A, b, p, sigma, mu )
%[ x, obj ] = galahad_l2rt( A, b, p, sigma, mu, control )

fprintf('solve dense example \n')
[ x, obj, inform ] = galahad_l2rt( A, b, p, sigma, mu, control );
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - l2rt: optimal f =', obj, '- status =', inform.status ) )

fprintf('solve sparse example \n')
SA = sparse(A) ;
[ x, obj, inform ] = galahad_l2rt( SA, b, p, sigma, mu, control );
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - l2rt: optimal f =', obj, '- status =', inform.status ) )
