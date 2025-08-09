% GALAHAD_SSLS -
%
%  Given a BLOCK, REAL SYMMETRIC MATRIX
%
%     K = ( H  A^T ),
%         ( A  - C )
%
%  this package forms and factorizes K, and solves the block linear system
%
%       ( H  A^T ) ( x ) = ( b ).
%       ( A  - C ) ( y )   ( d )
%
%  Full advantage is taken of any zero coefficients in the matrices H, A and C.
%
%  Simple usage -
%
%  to form and factorize the matrix K
%   [ inform ]
%     = galahad_ssls( 'form_and_factorize', H, A, C, control )
%
%  to solve the block linear system after factorizing K
%
%   [ x, y, inform ]
%     = galahad_ssls( 'solve', b, d, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%     = galahad_ssls( 'initial' )
%
%  to remove data structures after solution
%  [ inform ]
%    = galahad_ssls( 'final' )
%
%  Usual Input (form-and-factorize) -
%          H: the real symmetric n by n matrix H
%          A: the real m by n matrix A
%          C: the real symmetric m by m matrix C
%  or (solve) -
%          b: the real m-vector b
%          d: the real n-vector d
%
%  Optional Input -
%    control: a structure containing control parameters.
%            The components are of the form control.value, where
%            value is the name of the corresponding component of
%            the derived type SSLS_control_type as described in
%            the manual for the fortran 90 package GALAHAD_SSLS.
%            See: http://galahad.rl.ac.uk/galahad-www/doc/ssls.pdf
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of the
%      derived type SSLS_inform_type as described in the manual
%      for the fortran 90 package GALAHAD_SSLS. The component
%      inform.SLS_inform is itself a structure, holding the 
%      components of the derived type SLS_inform_type.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/ssls.pdf
%
% This version copyright Nick Gould for GALAHAD productions 9/August/2025
