% GALAHAD_SLS -
%
%  Given a symmetric n by n matrix A and an n-vector b or an n by r
%  matrix B, solve the system A x = b or the system AX=B. The matrix
%  A need not be definite. Advantage is taken of sparse A. Options
%  are provided to factorize a matrix A without solving the system,
%  and to solve systems using previously-determined factors.
%
%  Simple usage -
%
%  to solve a system Ax=b or AX=B
%   [ x, inform ] = galahad_sls( A, b, control, solver )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to factorization
%   [ control ]
%     = galahad_sls( 'initial', solver )
%
%  to factorize A
%   [ inform ] = galahad_sls( 'factor', A, control )
%
%  to solve Ax=b or AX=B using existing factors
%   [ x, inform ] = galahad_sls( 'solve', b )
%
%  to remove data structures after solution
%   [ inform ] = galahad_sls( 'final' )
%
%  Usual Input -
%    A: the symmetric matrix A
%    b: a column vector b or matrix of right-hand sides B
%
%  Optional Input -
%    control, a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type SLS_control_type as described in the
%      manual for the fortran 90 package GALAHAD_SLS.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/sls.pdf
%    solver, the name of the desired linear solver. Possible values are:
%        'sils'
%        'ma27'
%        'ma57'
%        'ma77'
%        'ma86'
%        'ma87'
%        'ma97'
%        'ssids'
%        'pardiso'
%        'wsmp'
%        'potr'
%        'sytr'
%        'pbtr'
%      The default is 'sils'. Not all options will be available. For more
%      details, see: http://galahad.rl.ac.uk/galahad-www/doc/sls.pdf
%
%  Usual Output -
%   x: the vector of solutions to Ax=b or matrix of solutions to AX=B
%
%  Optional Output -
%    control: see above. Returned values are the defaults
%    inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type SLS_inform_type as described
%      in the manual for the fortran 90 package GALAHAD_SLS.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/sls.pdf
%
% This version copyright Nick Gould for GALAHAD productions 9/November/2020
