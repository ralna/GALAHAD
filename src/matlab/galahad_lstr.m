% GALAHAD_LSTR -
%
%  Given an m by n matrix A, an m-vector b, and a scalar radius, find
%  an approximate solution of the LEAST-SQUARES TRUST-REGION subproblem
%    minimize || A x - b ||_2
%    subject to ||x||_2 <= radius
%  using an iterative method. Here ||.||_2 is the Euclidean (l_2-)norm.
%  Advantage is taken of sparse A. 
%
%  Simple usage -
%
%  to solve the least-squares trust-region subproblem in the Euclidean norm
%   [ x, obj, inform ] 
%     = galahad_lstr( A, b, radius, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ] 
%     = galahad_lstr( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, obj, inform ]
%     = galahad_lstr( 'existing', A, b, radius, control )
%
%  to remove data structures after solution
%   galahad_lstr( 'final' )
%
%  Usual Input -
%          A: the m by n matrix A
%          b: the m-vector b
%     radius: the trust-region radius (radius>0)
%
%  Optional Input -
%    control: a structure containing control parameters.
%            The components are of the form control.value, where
%            value is the name of the corresponding component of
%            the derived type LSTR_CONTROL as described in the 
%            manual for the fortran 90 package GALAHAD_LSTR.
%            See: http://galahad.rl.ac.uk/galahad-www/doc/lstr.pdf
%
%  Usual Output -
%          x: the global minimizer
%        obj: the optimal value of the objective function ||Ax-b||_2
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%           The components are of the form inform.value, where
%           value is the name of the corresponding component of the
%           derived type LSTR_INFORM as described in the manual for 
%           the fortran 90 package GALAHAD_LSTR. 
%           See: http://galahad.rl.ac.uk/galahad-www/doc/lstr.pdf
%           Note that as the objective value is already available
%           the component r_norm from LSTR_inform is omitted.
%
% This version copyright Nick Gould for GALAHAD productions 5/March/2009
