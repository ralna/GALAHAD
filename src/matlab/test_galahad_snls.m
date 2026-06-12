% test galahad_snls
% Nick Gould for GALAHAD productions 27/March/2026

clear p x0 x control inform

% SNLS documentation problem (dense Jacobian)

p.m_r = 4 ;
[ control ] = galahad_snls( 'initial' );
control.prine_level = 1;

x0 = [ 0.5 0.5 0.5 0.5 0.5 ];
[x, inform ] = galahad_snls( p, x0, 'snls_r', 'snls_jr', control );

disp( sprintf( 'dense example: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - snls: final f =', inform.obj, ...
       '- ||r|| =', inform.norm_r, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

%return

% SNLS documentation problem (spare Jacobian)

p.jr_row = [ 1, 1, 2, 2, 3, 3, 4, 4 ] ;
p.jr_col = [ 1, 2, 2, 3, 3, 4, 4, 5 ] ;

x0 = [ 0.5 0.5 0.5 0.5 0.5 ];
[ control ] = galahad_snls( 'initial' );
control.jacobian_available = 2 ;

[x, inform ] = galahad_snls( p, x0, 'snls_r', 'snls_jr_sparse', control );

disp( sprintf( 'sparse example: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - snls: final f =', inform.obj, ...
       '- ||r|| =', inform.norm_r, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

% SNLS documentation problem (spare Jacobian with options)

p.jr_row = [ 1, 1, 2, 2, 3, 3, 4, 4 ] ;
p.jr_col = [ 1, 2, 2, 3, 3, 4, 4, 5 ] ;

x0 = [ 0.5 0.5 0.5 0.5 0.5 ];
%cohort = [ 1, 2, 0, 1, 2 ] ;
cohort = [ 1, 1, 1, 1, 1 ] ;
w = [ 1 1 1 1 ];
[ control ] = galahad_snls( 'initial' );
control.jacobian_available = 2 ;

[x, inform ] = galahad_snls( p, x0, 'snls_r', 'snls_jr_sparse', ...
                             cohort, w, control );

disp( sprintf( 'sparse options example: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - snls: final f =', inform.obj, ...
       '- ||r|| =', inform.norm_r, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )
