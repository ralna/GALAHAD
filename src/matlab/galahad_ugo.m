% GALAHAD_UGO -
%
%  find a global bound-constrained minimizer of a twice differentiable objective
%  function f(x) of a real variable x over the finite interval [x_l,x_u]
%
%  Simple usage -
%
%  to find the minimizer
%   [ x, f, g, h, inform ]
%    = galahad_ugo( x_l, x_u, eval_fgh, control )
%   [ x, f, g, h, inform, z ]
%    = galahad_ugo( x_l, x_u, eval_fgh )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%    = galahad_ugo( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, f, g, h, inform ]
%    = galahad_ugo( 'existing', x_l, x_u, eval_fgh, control )
%   [ x, f, g, h, inform ]
%    = galahad_ugo( 'existing', x_l, x_u, eval_fgh )
%
%  to remove data structures after solution
%   galahad_ugo( 'final' )
%
%  Usual Input -
%       x_l: the finite lower bound x_l
%       x_u: the finite upper bound x_u
%    eval_fgh: a user-provided subroutine named eval_fgh.m for which
%              [f,g,h,status] = eval_fgh(x)
%            returns the value of objective function f and its first
%            derivative g = f'(x) at x. Additionally, if
%            control.second_derivative_available is true, also returns
%            the value of the second derivative h = f''(x) at x; h need
%            not be set otherwise. status should be set to 0 if the
%            evaluations succeed, and a non-zero value if an evaluation fails.
%
%  Optional Input -
%     control: a structure containing control parameters.
%              The components are of the form control.value, where
%              value is the name of the corresponding component of
%              the derived type UGO_control_type as described in
%              the manual for the fortran 90 package GALAHAD_UGO.
%              See: http://galahad.rl.ac.uk/galahad-www/doc/ugo.pdf
%
%  Usual Output -
%          x: the estimated global minimizer
%          f: the objective function value at x
%          g: the first derivative of the objective function value at x
%          h: the second derivative of the objective function value at x
%             when control.second_derivative_available is true
%
%  Optional Output -
%    control: see above. Returned values are the defaults
%     inform: a structure containing information parameters
%             The components are of the form inform.value, where
%             value is the name of the corresponding component of the
%             derived type UGO_inform_type as described in the manual
%             for the fortran 90 package GALAHAD_UGO. The component
%             inform.time is itself a structure, holding the
%             components of the derived types UGO_time_type.
%             See: http://galahad.rl.ac.uk/galahad-www/doc/ugo.pdf
%
% This version copyright Nick Gould for GALAHAD productions 16/March/2022
