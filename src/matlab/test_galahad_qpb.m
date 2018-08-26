% test galahad_qpb
% Nick Gould for GALAHAD productions 17/February/2010

clear H A SH SA control inform aux

%m = 800 ;
m = 5 ;
n = 2 * m ;
f = 1.0 ;
g(1:n)= 1.0 ;
H(1:n,1:n) = 0.0 ;
for i = 1:n
 H(i,i) = i-(n/2) ;
end
for i = 1:m
 for j = 1:n
  A(i,j) = i+j ;
 end
end
for i = 1:m
 c_l(i) = i ;
 c_u(i) =  5 * i ;
end
for i = 1:n
 x_l(i) = 0 ;
 x_u(i) =  inf;
end

[ control ] = galahad_qpb( 'initial' );
%control.print_level = 3 ;
%control.SBLS_control.print_level = 1 ;

%control.SBLS_control
fprintf('solve dense example \n')
[ x, inform, aux ] = galahad_qpb( H, g, f, A, c_l, c_u, x_l, x_u, ...
                                  control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )
galahad_qpb( 'final' )


fprintf('solve sparse example \n')
%control.print_level = 3 ;
%control.SBLS_control.print_level = 1 ;
%control.SBLS_control
SH = sparse(H) ;
SA = sparse(A) ;
[ x, inform, aux ] = galahad_qpb( SH, g, f, SA, c_l, c_u, x_l, x_u, ...
                                  control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - optimal f =', inform.obj, '- status =', inform.status ) )
galahad_qpb( 'final' )
