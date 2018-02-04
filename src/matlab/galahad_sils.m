% GALAHAD_SILS -
%
%  Given a symmetric n by n matrix A and an n-vector b or an n by r 
%  matrix B, solve the system A x = b or the system AX=B. The matrix 
%  A need not be definite. Advantage is taken of sparse A. Options
%  are provided to factorize a matrix A without solving the system, 
%  and to solve systems using previously-determined factors.
%
%  Usage -
%
%  Simple usage -
%
%  to solve a system Ax=b or AX=B
   %  [ x, inform ] = galahad_sils( A, b, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to factorization
%   [ control ] 
%     = galahad_sils( 'initial' )
%
%  to factorize A
   %  [ inform ] = galahad_sils( 'factor', A, control )
%
%  to solve Ax=b or AX=B using existing factors
   %  [ x, inform ] = galahad_sils( 'solve', b )
%
%  to remove data structures after solution
%  galahad_sils( 'final' )
%
%  Usual Input -
%    A: the symmetric matrix A
%    b a column vector b or matrix of right-hand sides B
%
%  Optional Input -
%    control, a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type SILS_CONTROL as described in the 
%      manual for the fortran 90 package GALAHARS_SILS.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/sils.pdf
%
%  Usual Output -
%   x: the vector of solutions to Ax=b or matrix of solutions to AX=B
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%    inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type SILS_AINFO/FINFO/SINFO as described 
%      in the manual for the fortran 90 package GALAHARS_SILS.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/sils.pdf
%
% This version copyright Nick Gould for GALAHAD productions 12/July/2007
