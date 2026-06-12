% test galahad_blls
% Nick Gould for GALAHAD productions 25/March/2026

clear Ao SAo b x_l x_u x_0 w x_s control inform

o = 10 ;
n = 5 ;
b(1)= -1.0 ;
b(2:o)= 1.0 ;
Ao(1:o,1:n) = 0.0 ;
for i = 1:n
 Ao(i,i) = i ;
 Ao(n+i,i) = 1 ;
 Ao(2*i,i) = 1 ;
end
for i = 1:n
 x_l(i) = 0 ;
 x_u(i) =  inf;
end
x_0(1:n)= 0.0 ;
control.out = 6 ;
%control.print_level = 1 ;

fprintf('solve dense example \n')
[ x, inform, aux ] = galahad_blls( Ao, b, 0.0, x_l, x_u, x_0, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - blls: optimal f =', inform.obj, '- status =', inform.status ) )

fprintf('solve sparse example \n')
SAo = sparse(Ao) ;
[ x, inform, aux ] = galahad_blls( SAo, b, 0.0, x_l, x_u, x_0, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - blls: optimal f =', inform.obj, '- status =', inform.status ) )

fprintf('solve sparse example all options\n')
w(1:o) = 1.0;
x_s(1:n) = 1.0;
[ x, inform, aux ] = galahad_blls( SAo, b, 0.0, x_l, x_u, x_0, ...
                                   w, x_s, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - blls: optimal f =', inform.obj, '- status =', inform.status ) )
