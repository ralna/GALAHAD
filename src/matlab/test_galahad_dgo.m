% test galahad_dgo
% Nick Gould for GALAHAD productions 16/March/2022

clear control inform

%[ x, inform ] = galahad_dgo( x_l, x_u, x0, 'camel6_f', 'camel6_g', ...
%                             'camel6_h', 'camel6_hprod' )

%  set default control values

[ control ] = galahad_dgo( 'initial' ) ;

control.maxit = 2000;

% Dgo problem (sparse Hessian)

pattern_h = [ 1 1 ; 2 1 ; 2 2 ];

%  solve with default values

x_l = [ -3.0 -2.0 ];
x_u = [ 3.0 2.0 ];
x_0 = [ 0.0 0.0 ];

[ x, inform ] = galahad_dgo( 'existing', x_l, x_u, x_0, 'camel6_f', ...
                             'camel6_g', 'camel6_h', 'camel6_hprod', ...
                             pattern_h, control) ;
disp( sprintf( 'dgo example: \n%s %13.6e %s %6.2e %s %2.0f %s %2.0f', ...
      ' - dgo: final f =', inform.obj, ...
      '- ||pg|| =', inform.norm_pg, ...
      '- evals =', inform.f_eval, ...
      '- status =', inform.status ) )

%  solve without local optimization

control.perform_local_optimization = false;
%control.print_level = 1;
[ x, inform ]  = galahad_dgo( 'existing', x_l, x_u, x_0, 'camel6_f', ...
                              'camel6_g', 'camel6_h', 'camel6_hprod', ...
                              control ) ;
disp( sprintf( '%s %13.6e %s %6.2e %s %2.0f %s %2.0f', ...
      ' - dgo: final f =', inform.obj, ...
      '- ||pg|| =', inform.norm_pg, ...
      '- evals =', inform.f_eval, ...
      '- status =', inform.status ) )

galahad_dgo( 'final' )
