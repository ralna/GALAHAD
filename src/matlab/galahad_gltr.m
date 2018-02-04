% GALAHAD_GLTR -
%
%  Given a symmetric n by n matrix H (and possibly M), an n-vector g, 
%  a constant f, and a scalar radius, find an approximate solution of 
%  the TRUST-REGION subproblem
%    minimize 0.5 * x' * H * x + c' * x + f
%    subject to ||x||_M <= radius
%  using an iterative method.
%  Here ||x||_M^2 = x' * M * x and M is positive definite; if M is
%  not given, M=I and ||x||_M is thus taken to be the Euclidean (l_2-)norm 
%  sqrt(x' * x). H need not be definite. Advantage is taken of sparse H. 
%
%  Simple usage -
%
%  to solve the trust-region subproblem in the M norm
%   [ x, obj, inform ] 
%     = galahad_gltr( H, c, f, radius, control, M )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ] 
%     = galahad_gltr( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, obj, inform ]
%     = galahad_gltr( 'existing', H, c, f, radius, control, M )
%
%  to remove data structures after solution
%   galahad_gltr( 'final' )
%
%  Usual Input -
%          H: the symmetric n by n matrix H
%          c: the n-vector c
%          f: the scalar f
%     radius: the trust-region radius (radius>0)
%
%  Optional Input -
%    control: a structure containing control parameters.
%            The components are of the form control.value, where
%            value is the name of the corresponding component of
%            the derived type GLTR_CONTROL as described in the 
%            manual for the fortran 90 package GALAHAD_GLTR.
%            See: http://galahad.rl.ac.uk/galahad-www/doc/gltr.pdf
%          M: the n by n symmetric, positive-definite matrix M
%
%  Usual Output -
%          x: the global minimizer
%        obj: the optimal value of the objective function
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type GLTR_INFORM as described in the manual for 
%      the fortran 90 package GALAHAD_GLTR. 
%      See: http://galahad.rl.ac.uk/galahad-www/doc/gltr.pdf
%
% This version copyright Nick Gould for GALAHAD productions 2/March/2009
