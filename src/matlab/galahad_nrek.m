% GALAHAD_NREK -
%
%  Given a symmetric n by n matrix H (and possibly S), an n-vector g,
%  a constant f, and scalars power and weight, find the solution of the
%  NORM-REGULARIZATION subproblem
%    minimize 0.5 * x' * H * x + c' * x + f + (weight/power) * ||x||_S^power
%  Here ||x||_S^2 = x' * S * x and S is diagonally dominant; if S is
%  not given, S=I and ||x||_S is thus taken to be the Euclidean (l_2-)norm
%  sqrt(x' * x). H need not be definite. Advantage is taken of sparse H and S.
%
%  Simple usage -
%
%  to solve the norm-regularization subproblem in the Euclidean norm
%   [ x, inform ]
%     = galahad_nrek( H, c, power, weight, control, S )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%     = galahad_nrek( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform ]
%     = galahad_nrek( 'existing', H, c, power, weight, control, S )
%
%  to remove data structures after solution
%   galahad_nrek( 'final' )
%
%  Usual Input -
%          H: the symmetric n by n matrix H
%          c: the n-vector c
%     weight: the regularization weight (weight>0)
%
%  Optional Input -
%    control: a structure containing control parameters.
%            The components are of the form control.value, where
%            value is the name of the corresponding component of
%            the derived type NREK_control_type as described in
%            the manual for the modern fortran package GALAHAD_NREK.
%            See: http://galahad.rl.ac.uk/galahad-www/doc/nrek.pdf
%          S: the n by n symmetric, diagonally-dominant matrix S
%
%  Usual Output -
%          x: the global minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of the
%      derived type NREK_inform_type as described in the manual
%      for the fortran 90 package GALAHAD_NREK. The components
%      of inform.time, inform.NREK_inform and inform.SLS_inform 
%      are themselves structures, holding the
%      components of the derived types NREK_time_type, 
%      NREK_inform_type and SLS_inform_type, respectively.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/nrek.pdf
%
% This version copyright Nick Gould for GALAHAD productions 22/November/2025
