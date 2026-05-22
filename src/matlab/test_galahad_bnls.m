% test galahad_bnls
% Nick Gould for GALAHAD productions 21/May/2026

clear p x_l x_u x_0 x control inform

% BNLS documentation problem (dense Jacobian)

p.m_r = 4 ;
[ control ] = galahad_bnls( 'initial' );
control.prine_level = 1;

x_l = [ 0.0 0.0 0.0 0.0 0.0 ];
x_u = [ 1.0 1.0 1.0 1.0 1.0 ];
x_0 = [ 0.5 0.5 0.5 0.5 0.5 ];
[x, inform ] = galahad_bnls( p, x_l, x_u, x_0, 'bnls_r', 'bnls_jr', control );

disp( sprintf( 'dense example: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - bnls: final f =', inform.obj, ...
       '- ||r|| =', inform.norm_r, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

%return

% BNLS documentation problem (spare Jacobian)

p.jr_row = [ 1, 1, 2, 2, 3, 3, 4, 4 ] ;
p.jr_col = [ 1, 2, 2, 3, 3, 4, 4, 5 ] ;

x_0 = [ 0.5 0.5 0.5 0.5 0.5 ];
[ control ] = galahad_bnls( 'initial' );
control.jacobian_available = 2 ;

[x, inform ] = galahad_bnls( p, x_l, x_u, x_0, 'bnls_r', 'bnls_jr_sparse', ...
                             control );

disp( sprintf( 'sparse example: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - bnls: final f =', inform.obj, ...
       '- ||r|| =', inform.norm_r, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

% BNLS documentation problem (spare Jacobian with options)

p.jr_row = [ 1, 1, 2, 2, 3, 3, 4, 4 ] ;
p.jr_col = [ 1, 2, 2, 3, 3, 4, 4, 5 ] ;

x_0 = [ 0.5 0.5 0.5 0.5 0.5 ];
w = [ 1 1 1 1 ];
[ control ] = galahad_bnls( 'initial' );
control.jacobian_available = 2 ;

[x, inform ] = galahad_bnls( p, x_l, x_u, x_0, 'bnls_r', 'bnls_jr_sparse', ...
                             w, control );

disp( sprintf( 'sparse options example: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - bnls: final f =', inform.obj, ...
       '- ||r|| =', inform.norm_r, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )
