% GALAHAD_TREK -
%
%  Given a symmetric n by n matrix H (and possibly S), an n-vector g, 
%  a constant f, and a scalar radius, find the solution of the 
%  TRUST-REGION subproblem
%    minimize 0.5 * x' * H * x + c' * x + f
%    subject to ||x||_M <= radius
%  Here ||x||_S^2 = x' * S * x and S is diagonally dominant; if S is
%  not given, S=I and ||x||_S is thus taken to be the Euclidean (l_2-)norm
%  sqrt(x' * x). H need not be definite. Advantage is taken of sparse H and S.
%
%  Simple usage -
%
%  to solve the trust-region subproblem in the Euclidean norm
%   [ x, inform ]
%     = galahad_trek( H, c, radius, control, S )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%     = galahad_trek( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform ]
%     = galahad_trek( 'existing', H, c, radius, control, S )
%
%  to remove data structures after solution
%   galahad_trek( 'final' )
%
%  Usual Input -
%          H: the symmetric n by n matrix H
%          c: the n-vector c
%     radius: the trust-region radius (radius>0)
%
%  Optional Input -
%    control: a structure containing control parameters.
%            The components are of the form control.value, where
%            value is the name of the corresponding component of
%            the derived type TREK_control_type as described in
%            the manual for the fortran 90 package GALAHAD_TREK.
%            See: http://galahad.rl.ac.uk/galahad-www/doc/trek.pdf
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
%      derived type TREK_inform_type as described in the manual
%      for the fortran 90 package GALAHAD_TREK. The components
%      of inform.time, inform.TRS_inform and inform.SLS_inform 
%      are themselves structures, holding the
%      components of the derived types TREK_time_type, 
%      TRS_inform_type and SLS_inform_type, respectively.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/trek.pdf
%
% This version copyright Nick Gould for GALAHAD productions 17/November/2025
