% GALAHAD_SBLS -
%
%  Given a BLOCK, REAL SYMMETRIC MATRIX
%
%         ( H  A^T ),
%         ( A  - C )
%
%  this package constructs a variety of PRECONDITIONERS of the form
%
%     K = ( G  A^T ).
%         ( A  - C )
%
%  Here, the leading-block matrix G is a suitably-chosen approximation 
%  to H; it may either be prescribed EXPLICITLY, in which case a symmetric 
%  indefinite factorization of K will be formed using the GALAHAD package 
%  SLS, or IMPLICITLY by requiring certain sub-blocks of G be zero. In the 
%  latter case, a factorization of K will be obtained implicitly (and more 
%  efficiently) without recourse to SLS.
%
%  Once the preconditioner has been constructed, solutions to the 
%  preconditioning system
%
%       ( G  A^T ) ( x ) = ( b )
%       ( A  - C ) ( y )   ( d )
%
%  may be obtained by the package. Full advantage is taken of any zero 
%  coefficients in the matrices H, A and C.
%
%  Simple usage -
%
%  to form and factorize the matrix K
%   [ inform ] 
%     = galahad_sbls( 'form_and_factorize', H, A, C, control )
%
%  to solve the preconditioning system after factorizing K
%
%   [ x, y, inform ] 
%     = galahad_sbls( 'solve', b, d, control )

%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ] 
%     = galahad_sbls( 'initial' )
%
%  to remove data structures after solution
%  [ inform ] 
%    = galahad_sbls( 'final' )
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
%            the derived type SBLS_control_type as described in
%            the manual for the fortran 90 package GALAHAD_SBLS.
%            See: http://galahad.rl.ac.uk/galahad-www/doc/sbls.pdf
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of the
%      derived type SBLS_inform_type as described in the manual
%      for the fortran 90 package GALAHAD_SBLS. The components
%      of inform.SLS_inform and inform.ULS_inform are themselves 
%      structures, holding the components of the derived types 
%      SLS_inform_type and ULS_inform_type, respectively.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/sbls.pdf
%
% This version copyright Nick Gould for GALAHAD productions 15/February/2010
