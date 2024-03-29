% test galahad_trb
% Nick Gould for GALAHAD productions 21/July/2021

clear control inform

%[ x, inform ] = galahad_trb( x_l, x_u, x0, 'rosenbrock_f', 'rosenbrock_g', ...
%                             'rosenbrock_h' )

%  set default control values

[ control ] = galahad_trb( 'initial' ) ;

% Rosenbrock problem (dense Hessian)

%  solve with default values

x_l = [ 2.0 0.0 ];
x_u = [ inf inf ];
x_0 = [ -1.2 1 ];
[ x, inform, z ] = galahad_trb( 'existing', x_l, x_u, x_0, 'rosenbrock_f', ...
                                'rosenbrock_g', 'rosenbrock_h' ) ;
disp( sprintf( 'Rosenbrock: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
      ' - trb: final f =', inform.obj, ...
      '- ||pg|| =', inform.norm_pg, ...
      '- iter =', inform.iter, ...
      '- status =', inform.status ) )

%  solve with direct solution of the subproblems

control.subproblem_direct = 1;
%control.print_level = 1;
[ x, inform ]  = galahad_trb( 'existing', x_l, x_u, x_0, 'rosenbrock_f', ...
                              'rosenbrock_g', 'rosenbrock_h', control ...
                              ) ;
disp( sprintf( '%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
      ' - trb: final f =', inform.obj, ...
      '- ||pg|| =', inform.norm_pg, ...
      '- iter =', inform.iter, ...
      '- status =', inform.status ) )

galahad_trb( 'final' )

% Tridiagonal problem (sparse Hessian)

[ control ] = galahad_trb( 'initial' ) ;

%  solve with default values

x_l = [ 2.0 0.0 2.0 ];
x_u = [ inf inf inf ];
x_0 = [ 4 3 2 ];
pattern_h = [ 1 1 ; 2 1 ; 2 2 ; 3 2 ; 3 3 ];
[ x, inform ]  = galahad_trb( 'existing', x_l, x_u, x_0, 'tridia_f', ...
                              'tridia_g', 'tridia_h', pattern_h ) ;

disp( sprintf( 'Tridia: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
      ' - trb: final f =', inform.obj, ...
      '- ||pg|| =', inform.norm_pg, ...
      '- iter =', inform.iter, ...
      '- status =', inform.status ) )

%  solve with direct solution of the subproblems

control.subproblem_direct = 1;
%control.print_level = 1;
[ x, inform, z ]  = galahad_trb( 'existing', x_l, x_u, x_0, 'tridia_f', ...
                              'tridia_g', 'tridia_h', pattern_h, ...
                              control ) ;
disp( sprintf( '%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
      ' - trb: final f =', inform.obj, ...
      '- ||pg|| =', inform.norm_pg, ...
      '- iter =', inform.iter, ...
      '- status =', inform.status ) )

galahad_trb( 'final' )
