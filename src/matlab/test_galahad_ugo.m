% test galahad_ugo
% Nick Gould for GALAHAD productions 15/March/2022

clear control inform

%  set default control values

[ control ] = galahad_ugo( 'initial' ) ;

% Rosenbrock problem (dense Hessian)

%  solve with default values

x_l = -1.0;
x_u = 2.0;
[ x, f, g, h, inform ] = galahad_ugo( 'existing', x_l, x_u, 'wiggly_fgh' );
disp( sprintf( 'Wiggly: \n%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
      ' - ugo: final f =', f, ...
      '- g =', g, ...
      '- eval =', inform.f_eval, ...
      '- status =', inform.status ) )

%  solve with direct solution of the subproblems

control.second_derivative_available = false;
%control.print_level = 1;
[ x, f, g, h, inform ]  = galahad_ugo( 'existing', x_l, x_u, 'wiggly_fgh', ...
                                       control );
disp( sprintf( '%s %13.6e %s %8.4e %s %2.0f %s %2.0f', ...
      ' - ugo: final f =', f, ...
      '- g =', g, ...
      '- eval =', inform.f_eval, ...
      '- status =', inform.status ) )

galahad_ugo( 'final' )
