% test galahad_bgo
% Nick Gould for GALAHAD productions 16/March/2022

clear control inform

%[ x, inform ] = galahad_bgo( x_l, x_u, x0, 'bgo_f', 'bgo_g', ...
%                             'bgo_h', 'bgo_hprod' )

%  set default control values

[ control ] = galahad_bgo( 'initial' ) ;

control.TRB_control.subproblem_direct = true;
control.attempts_max = 1000;
control.max_evals = 1000;
control.TRB_control.maxit = 10;
control.random_multistart = true;

% Bgo problem (sparse Hessian)

pattern_h = [ 1 1 ; 2 2 ; 3 1 ; 3 2 ; 3 3 ];

%  solve with default values

x_l = [ -10.0 -10.0 -10.0 ];
x_u = [ 0.5 0.5 0.5 ];
x_0 = [ 0.0 0.0 0.0 ];

[ x, inform ] = galahad_bgo( 'existing', x_l, x_u, x_0, 'bgo_f', ...
                             'bgo_g', 'bgo_h', 'bgo_hprod', ...
                             pattern_h, control) ;
disp( sprintf( 'bgo example: \n%s %13.6e %s %6.2e %s %2.0f %s %2.0f', ...
      ' - bgo: final f =', inform.obj, ...
      '- ||pg|| =', inform.norm_pg, ...
      '- evals =', inform.f_eval, ...
      '- status =', inform.status ) )

%  solve with minimize-and-probe strategy

control.random_multistart = false;
%control.print_level = 1;
[ x, inform ]  = galahad_bgo( 'existing', x_l, x_u, x_0, 'bgo_f', ...
                              'bgo_g', 'bgo_h', 'bgo_hprod', control ) ;
disp( sprintf( '%s %13.6e %s %6.2e %s %2.0f %s %2.0f', ...
      ' - bgo: final f =', inform.obj, ...
      '- ||pg|| =', inform.norm_pg, ...
      '- evals =', inform.f_eval, ...
      '- status =', inform.status ) )

galahad_bgo( 'final' )
