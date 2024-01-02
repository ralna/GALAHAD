% test galahad_slls
% Nick Gould for GALAHAD productions 2023-12-30

clear Ao SAo control inform

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
control.out = 6 ;
%control.print_level = 1 ;

fprintf('solve dense example \n')
[ x, inform, aux ] = galahad_slls( Ao, b, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - slls: optimal f =', inform.obj, '- status =', inform.status ) )

fprintf('solve sparse example \n')
SAo = sparse(Ao) ;
[ x, inform, aux ] = galahad_slls( SAo, b, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - slls: optimal f =', inform.obj, '- status =', inform.status ) )
