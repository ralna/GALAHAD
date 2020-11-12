% test galahad_nls
% Nick Gould for GALAHAD productions 08/March/2019

clear control inform p

% NLS documentation problem (dense matrices)

p.m = 2 ;
[ control ] = galahad_nls( 'initial' );

x0 = [ 1 1 1 ];
%control.print_level = 1;
[x, inform ] = galahad_nls( p, x0, 'nls_r', 'nls_j', control );

disp( sprintf( 'example 1: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - nls: final f =', inform.obj, ...
       '- ||g|| =', inform.norm_g, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

%return

x0 = [ 1 1 1 ];
control.model = 4;
%control.maxit = 1000;
%control.print_level = 1;
[x, inform ] = galahad_nls( p, x0, 'nls_r', 'nls_j', 'nls_h', control );

disp( sprintf( 'example 2: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - nls: final f =', inform.obj, ...
       '- ||g|| =', inform.norm_g, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

galahad_nls( 'final' )

x0 = [ 1 1 1 ];
[ control ] = galahad_nls( 'initial' );
control.model = 6;
%control.print_level = 1;
[x, inform ] = galahad_nls( p, x0, 'nls_r', 'nls_j', 'nls_h', 'nls_p', ...
                            control );

disp( sprintf( 'example 3: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - nls: final f =', inform.obj, ...
       '- ||g|| =', inform.norm_g, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

% NLS documentation problem (spare matrices)

p.j_row = [ 1 1 2 2 ] ;
p.j_col = [ 1 3 2 3 ] ;

x0 = [ 1 1 1 ];
[ control ] = galahad_nls( 'initial' );
control.model = 3;
[x, inform ] = galahad_nls( p, x0, 'nls_r', 'nls_j_sparse', control );

disp( sprintf( 'example 4: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - nls: final f =', inform.obj, ...
       '- ||g|| =', inform.norm_g, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

p.h_row = [ 1 2 3 ] ;
p.h_col = [ 1 2 1 ] ;

x0 = [ 1 1 1 ];
control.model = 4;
[x, inform ] = galahad_nls( p, x0, 'nls_r', 'nls_j_sparse', ...
                            'nls_h_sparse', control );

disp( sprintf( 'example 5: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - nls: final f =', inform.obj, ...
       '- ||g|| =', inform.norm_g, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

p.p_row = [ 1 3 2 ] ;
p.p_col = [ 1 1 2 ] ;

x0 = [ 1 1 1 ];
control.model = 6;
%control.print_level = 1;
[x, inform ] = galahad_nls( p, x0, 'nls_r', 'nls_j_sparse', ...
                           'nls_h_sparse', 'nls_p_sparse', control );

disp( sprintf( 'example 6: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - nls: final f =', inform.obj, ...
       '- ||g|| =', inform.norm_g, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

clear p

% NLS documentation problem (spare matrices with weights)

p.m = 2 ;
[ control ] = galahad_nls( 'initial' );

p.w = [ 1 1 ] ;
p.j_row = [ 1 1 2 2 ] ;
p.j_col = [ 1 3 2 3 ] ;

x0 = [ 1 1 1 ];
[ control ] = galahad_nls( 'initial' );
control.model = 3;
[x, inform ] = galahad_nls( p, x0, 'nls_r', 'nls_j_sparse', control );

disp( sprintf( 'example 7: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - nls: final f =', inform.obj, ...
       '- ||g|| =', inform.norm_g, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

p.h_row = [ 1 2 3 ] ;
p.h_col = [ 1 2 1 ] ;

x0 = [ 1 1 1 ];
control.model = 4;
[x, inform ] = galahad_nls( p, x0, 'nls_r', 'nls_j_sparse', ...
                            'nls_h_sparse', control );

disp( sprintf( 'example 8: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - nls: final f =', inform.obj, ...
       '- ||g|| =', inform.norm_g, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

p.p_row = [ 1 3 2 ] ;
p.p_col = [ 1 1 2 ] ;

x0 = [ 1 1 1 ];
control.model = 6;
%control.print_level = 1;
[x, inform ] = galahad_nls( p, x0, 'nls_r', 'nls_j_sparse', ...
                           'nls_h_sparse', 'nls_p_sparse', control );

disp( sprintf( 'example 9: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - nls: final f =', inform.obj, ...
       '- ||g|| =', inform.norm_g, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

%return
% options

x0 = [ 4 3 2 ];
[ control ] = galahad_nls( 'initial' );
control.model = 3;
control.print_level = 0;
[x, inform ] = galahad_nls( p, x0, 'nls_r', 'nls_j_sparse', control );

%  solve with default values

x0 = [ 4 3 2 ];
[x, inform ] = galahad_nls( 'existing', p, x0, 'nls_r', 'nls_j_sparse', ...
                            control );

disp( sprintf( 'example 10: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - nls: final f =', inform.obj, ...
       '- ||g|| =', inform.norm_g, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

%  solve with direct solution of the subproblems

x0 = [ 4 3 2 ];
control.subproblem_direct = 1;
[x, inform ] = galahad_nls( 'existing', p, x0, 'nls_r', 'nls_j_sparse', ...
                            control );

disp( sprintf( 'example 11: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
       ' - nls: final f =', inform.obj, ...
       '- ||g|| =', inform.norm_g, ...
       '- iter =', inform.iter, ...
       '- status =', inform.status ) )

galahad_nls( 'final' )
