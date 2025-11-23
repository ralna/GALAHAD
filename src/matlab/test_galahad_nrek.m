% test galahad_nrek
% Nick Gould for GALAHAD productions 22/November/2025

clear H S SA SH SS c power weight control inform

n = 10 ;
control.out = 0 ;

power = 3.0 ;
weight = 0.1 ;
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
[ x, inform ] = galahad_nrek( H, c, power, weight, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - nrek: optimal f =', inform.obj, '- status =', inform.status ) )

[ control ] = galahad_nrek( 'initial' ) ;
[ x, inform ] = galahad_nrek( 'existing', H, c, power, weight, control, S ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - nrek: optimal f =', inform.obj, '- status =', inform.status ) )
weight = inform.next_weight ;
control.new_weight = true ;
%control.print_level = 1;
[ x, inform ] = galahad_nrek( 'existing', H, c, power, weight, control, S ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - nrek: optimal f =', inform.obj, '- status =', inform.status ) )
galahad_nrek( 'final' ) ;

fprintf('solve sparse examples \n')
SH = sparse(H) ;
SS = sparse(S) ;
weight = 0.1 ;
control.new_weight = false ;
[ x, inform ] = galahad_nrek( SH, c, power, weight, control ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - nrek: optimal f =', inform.obj, '- status =', inform.status ) )
[ x, inform ] = galahad_nrek( SH, c, power, weight, control, SS ) ;
disp( sprintf( '%s %13.6e %s %2.0f', ...
  ' - nrek: optimal f =', inform.obj, '- status =', inform.status ) )
