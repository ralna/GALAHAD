! THIS VERSION: GALAHAD 4.1 - 2022-07-20 AT 15:25 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ L S P    M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released as modified qpp GALAHAD Version 4.1. July 24th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_LSP_double

!     --------------------------------------------------------------
!     |                                                            |
!     | Preprocess the data for the constrained linear             |
!     | least-squares problem                                      |
!     |                                                            |
!     |    minimize     1/2 || A x - b ||^2                        |
!     |                                                            |
!     |    subject to     c_l <=  L x  <= c_u                      |
!     |    and            x_l <=   x   <= x_u                      |
!     |                                                            |
!     | to make further manipulations more efficient               |
!     |                                                            |
!     | Optionally instead do this for the parametric problem      |
!     |                                                            |
!     |    minimize     1/2 || A x - b + theta b||^2               |
!     |                                                            |
!     |    subject to c_l + theta dc_l <= L x <= c_u + theta dc_u  |
!     |    and        x_l + theta dx_l <=  x  <= x_u + theta dx_u  |
!     |                                                            |
!     | where theta is a scalar parameter                          |
!     |                                                            |
!     --------------------------------------------------------------

      USE GALAHAD_SYMBOLS
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SPACE_double
      USE GALAHAD_SORT_double,                                                 &
        ONLY: SORT_inplace_permute, SORT_inverse_permute, SORT_quicksort
      USE GALAHAD_LMS_double, ONLY: LMS_data_type, LMS_apply_lbfgs
      USE GALAHAD_LAPACK_interface, ONLY : POTRS
      USE GALAHAD_BLAS_interface, ONLY : GEMV

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SMT_put, SMT_get, QPT_problem_type, QPT_dimensions_type,       &
                LSP_initialize, LSP_reorder, LSP_apply, LSP_restore,           &
                LSP_get_values, LSP_terminate

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )

      INTEGER, PARAMETER :: coordinate = 0
      INTEGER, PARAMETER :: sparse_by_rows = 1
      INTEGER, PARAMETER :: sparse_by_columns = 2
      INTEGER, PARAMETER :: dense = 3
      INTEGER, PARAMETER :: dense_by_rows = dense
      INTEGER, PARAMETER :: dense_by_columns = 4

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LSP_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = infinity

!    any problem bound with the value zero will be treated as if it were a
!     general value if true

        LOGICAL :: treat_zero_bounds_as_general = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LSP_inform_type

!  return status. See LSP_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )
      END TYPE

      TYPE, PUBLIC :: LSP_map_type
        INTEGER :: m, n, o, a_ne, l_ne, a_type, l_type, n_reordered
        INTEGER :: o_reordered, m_reordered, a_ne_original, l_ne_original
        INTEGER :: h_ne_original, h_type, h_ne, h_diag_end_fixed
        LOGICAL :: a_perm, l_perm
        LOGICAL :: set = .FALSE.
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: x_map
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: a_map
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: c_map
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: a_map_inverse
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: l_map_inverse
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: h_map_inverse
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: ptr_a_fixed
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: ptr_l_fixed
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
        CHARACTER, ALLOCATABLE, DIMENSION( : ) :: a_type_original
        CHARACTER, ALLOCATABLE, DIMENSION( : ) :: l_type_original
        CHARACTER, ALLOCATABLE, DIMENSION( : ) :: h_type_original
      END TYPE

   CONTAINS

!-*-*-*-*-*-   L S P _ i n i t i a l i z e   S U B R O U T I N E  -*-*-*-*-*-

      SUBROUTINE LSP_initialize( map, control )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for LSP. This routine should be called
!  before LSP_reorder
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  map      map arrays
!  control  a structure containing control information. Components are -
!
!  INTEGER control parameter:
!
!   error. Error and warning diagnostics occur on stream error
!
!  REAL control parameter:
!
!   infinity. Any bound larger or equal to infinity in abolute value
!    will be considered to be infinite
!
!  LOGICAL control parameter:
!
!    treat_zero_bounds_as_general. If true, any problem bound with the value
!     zero will be treated as if it were a general value
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSP_map_type ), INTENT( OUT ) :: map
      TYPE ( LSP_control_type ), INTENT( OUT ) :: control

!  Real parameter

      control%infinity = infinity

      map%set = .FALSE.

      RETURN

!  End of LSP_initialize

      END SUBROUTINE LSP_initialize

!-*-*-*-*-*-*-*-   L S P _ r e o r d e r    S U B R O U T I N E  -*-*-*-*-*-*-

      SUBROUTINE LSP_reorder( map, control, inform, dims, prob,               &
                              get_x, get_y, get_z, parametric )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find a reordering of the data for the problem
!
!     minimize          q(x) = 1/2 || A x - b ||^2
!
!     subject to the bounds  x_l <=  x  <= x_u
!     and constraints        c_l <= L x <= c_u ,
!
!  where x is a vector of n components ( x_1, .... , x_n ), A is an o by n 
!  matrix, L is an m by n matrix, there are o obserservations b, and any of 
!  the bounds x_l, x_u, c_l, c_u may be infinite.
!
!  The reordered problem has the following properties:
!
!  * the variables are ordered so that their bounds appear in the order
!
!    free                      x
!    non-negativity      0  <= x
!    lower              x_l <= x
!    range              x_l <= x <= x_u
!    upper                     x <= x_u
!    non-positivity            x <=  0
!
!    Fixed variables will be removed. Within each category, the variables
!    are further ordered so that those with non-zero diagonal Hessian
!    entries occur before the remainder
!
!  * the constraints are ordered so that their bounds appear in the order
!
!    equality           c_l  = L x
!    lower              c_l <= L x
!    range              c_l <= L x <= c_u
!    upper                     L x <= c_u
!
!    Free constraints will be removed.

!  * additional constraints may be added, bounds tightened,
!    to reduce the size of the feasible region if this is
!    possible
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  dims is a structure of type QPT_dimensions_type, whose components hold SCALAR
!   information about the transformed problem on output.
!
!  prob is a structure of type QPT_type, whose components hold the
!   details of the problem. The following components must be set

!   f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
!
!   n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: n >= 1
!
!   o is an INTEGER variable, which must be set by the user to the
!    number of observations, o.  RESTRICTION: o >= 1
!
!   m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m.  RESTRICTION: m >= 0
!
!   A is a structure of type SMT_type used to hold the matrix A.
!    Five storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!       A%ne         the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%val( : )   the values of the components of A, stored row by row
!       A%col( : )   the column indices of the components of A
!       A%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) sparse, by columns
!
!       In this case, the following must be set:
!
!       A%type( 1 : 17 ) = TRANSFER( 'SPARSE_BY_COLUMNS', A%type )
!       A%val( : )   the values of the components of A, stored column by column
!       A%row( : )   the row indices of the components of A
!       A%ptr( : )   pointers to the start of each column, and past the end of
!                    the last column
!
!    iv) dense, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 5 ) = TRANSFER( 'DENSE', A%type )
!         [ or A%type( 1 : 13 ) = TRANSFER( 'DENSE_BY_ROWS', A%type ) ]
!       A%val( : )   the values of the components of A, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    v) dense, by columns
!
!       In this case, the following must be set:
!
!       A%type( 1 : 16 ) = TRANSFER( 'DENSE_BY_COLUMNS', A%type )
!       A%val( : )   the values of the components of A, stored column by column
!                    with each the entries in each column in order of
!                    increasing row indicies.
!
!    On exit, the components will most likely have been reordered. The 
!    output  matrix will be stored by columns, according to scheme (iii) above.
!    However, if scheme (i) is used for input, the output A%col will contain
!    the column numbers corresponding to the values in A%val, and thus in this
!    case the output matrix will be available in both formats (i) and (iii).
!
!   B is a REAL array of length o, which must be set by the user to the value 
!    of the observations, b. The i-th component of B, i = 1, ...., o should 
!    contain the value of b_i. On exit, B will most likely have been reordered.
!
!   Hessian_kind is an INTEGER variable which defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero, and the analytic centre of the
!        feasible region will be found. WEIGHT (see below) need not be set
!
!     1  all the weights will be one. WEIGHT (see below) need not be set
!
!     2  the weights will be those given by WEIGHT (see below)
!
!    <0  the Hessian H will be used
!
!   H is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of H (except for the L-BFGS case). Eight storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       H%type( 1 : 10 ) = TRANSFER( 'COORDINATE', H%type )
!       H%val( : )   the values of the components of H
!       H%row( : )   the row indices of the components of H
!       H%col( : )   the column indices of the components of H
!       H%ne         the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of H
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', H%type )
!       H%val( : )   the values of the components of H, stored row by row
!       H%col( : )   the column indices of the components of H
!       H%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = TRANSFER( 'DENSE', H%type )
!       H%val( : )   the values of the components of H, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', H%type )
!       H%val( : )   the values of the diagonals of H, stored in order
!
!   v) scaled identity
!
!       In this case, the following must be set:
!
!       H%type( 1 : 15) = 'SCALED-IDENTITY'
!       H%val( 1 )  the value assigned to each diagonal of H
!
!   vi) identity
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = 'IDENTITY'
!
!   vii) no Hessian
!
!       In this case, the following must be set:
!
!       H%type( 1 : 4 ) = 'ZERO' or 'NONE'
!
!   viii) L-BFGS Hessian
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = 'LBFGS'
!
!       The Hessian in this case is available via the component H_lm below
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above,
!    except for scheme (viii), for which a permutation will be set within H_lm.
!    However, if scheme (i) is used for input, the output H%row will contain
!    the row numbers corresponding to the values in H%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   H_lm is a structure of type LMS_data_type, whose components hold the
!     L-BFGS Hessian. Access to this structure is via the module GALAHAD_LMS,
!     and this component needs only be set if %H%type( 1 : 5 ) = 'LBFGS.'
!
!   WEIGHT is a REAL array, which need only be set if %Hessian_kind is larger
!    than 1. If this is so, it must be of length at least %n, and contain the
!    weights W for the objective function.
!
!   target_kind is an INTEGER variable that defines possible special
!     targets X0. Possible values are
!
!     0  X0 will be a vector of zeros.
!        %X0 (see below) need not be set
!
!     1  X0 will be a vector of ones.
!        %X0 (see below) need not be set
!
!     any other value - the values of X0 will be those given by X0 (see below)
!
!   X0 is a REAL array, which need only be set if %Hessian_kind is larger
!    that 0 and %target_kind /= 0,1. If this is so, it must be of length at
!    least %n, and contain the targets X^0 for the objective function.
!
!   gradient_kind is an INTEGER variable which defines the type of linear
!    term of the objective function to be used. Possible values are
!
!     0  the linear term g will be zero, and the analytic centre of the
!        feasible region will be found if in addition %Hessian_kind is 0.
!        %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by G (see below)
!
!   G is a REAL array of length n, which must be set by
!    the user to the value of the gradient, g, of the linear term of the
!    quadratic objective function. The i-th component of G, i = 1, ....,
!    n, should contain the value of g_i.
!    On exit, G will most likely have been reordered.
!
!   DG is a REAL array of length n, which, if allocated, must
!    be set by the user to the values of the array dg of the parametric
!    linear term of the quadratic objective function.
!    On exit, present DG will most likely have been reordered.
!
!   L is a structure of type SMT_type used to hold the matrix L.
!    Five storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       L%type( 1 : 10 ) = TRANSFER( 'COORDINATE', L%type )
!       L%val( : )   the values of the components of L
!       L%row( : )   the row indices of the components of L
!       L%col( : )   the column indices of the components of L
!       L%ne         the number of nonzeros used to store L
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       L%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', L%type )
!       L%val( : )   the values of the components of L, stored row by row
!       L%col( : )   the column indices of the components of L
!       L%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) sparse, by columns
!
!       In this case, the following must be set:
!
!       L%type( 1 : 17 ) = TRANSFER( 'SPARSE_BY_COLUMNS', L%type )
!       L%val( : )   the values of the components of L, stored column by column
!       L%row( : )   the row indices of the components of L
!       L%ptr( : )   pointers to the start of each column, and past the end of
!                    the last column
!
!    iv) dense, by rows
!
!       In this case, the following must be set:
!
!       L%type( 1 : 5 ) = TRANSFER( 'DENSE', L%type )
!         [ or L%type( 1 : 13 ) = TRANSFER( 'DENSE_BY_ROWS', L%type ) ]
!       L%val( : )   the values of the components of L, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    v) dense, by columns
!
!       In this case, the following must be set:
!
!       L%type( 1 : 16 ) = TRANSFER( 'DENSE_BY_COLUMNS', L%type )
!       L%val( : )   the values of the components of L, stored column by column
!                    with each the entries in each column in order of
!                    increasing row indicies.
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output L%row will contain
!    the row numbers corresponding to the values in L%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   C is a REAL array of length m, which is used to store the values of
!    A x. On exit, it will have been filled with appropriate values.
!
!   get_x is a LOGICAL variable. See X.
!
!   X is a REAL array of length n, which is used to store the values of
!   the variables x. If the user has assigned values to X, get_x must be .FALSE.
!   on entry, and X filled with the values of x. In this case, on exit,
!   X will most likely have been reordered, and any fixed variables moved
!   to their bounds. If the user does not wish to provide values for X,
!   get_x must be .TRUE. on entry. On exit it will have been filled with
!   appropriate values.
!
!   X_l, X_u are REAL arrays of length n, which must be set by the user
!    to the values of the arrays x_l and x_u of lower and upper bounds on x.
!    Any bound x_l_i or x_u_i larger than or equal to infinity in absolute value
!    will be regarded as being infinite (see the entry control%infinity).
!    Thus, an infinite lower bound may be specified by setting the appropriate
!    component of X_l to a value smaller than -infinity, while an infinite
!    upper bound can be specified by setting the appropriate element of X_u
!    to a value larger than infinity.
!    On exit, X_l and X_u will most likely have been reordered.
!
!   DX_l, DX_u are REAL arrays of length n, which, if allocated, must
!    be set by the user to the values of the arrays dx_l and dx_u of parametric
!    lower and upper bounds on x.
!    On exit, present DX_l and DX_u will most likely have been reordered.
!
!   get_z is a LOGICAL variable. See Z
!
!   Z is a REAL array of length n, which are used to store the values
!    of the dual variables (Lagrange multipliers) corresponding to the simple
!    bound constraints x_l <= x and x <= x_u. If the
!    user has assigned values to Z, get_z must be .FALSE. on entry.
!    In this case, on exit, Z will most likely have been reordered.
!    If the user does not wish to provide values for Z, get_Z must be
!    .TRUE.on entry. On exit it will have been filled with appropriate values.
!
!   C_l, C_u are REAL array of length m, which must be set by the user to
!    the values of the arrays c_l and c_u of lower and upper bounds on A x.
!    Any bound c_l_i or c_u_i larger than or equal to infinity in absolute value
!    will be regarded as being infinite (see the entry control%infinity).
!    Thus, an infinite lower bound may be specified by setting the appropriate
!    component of C_l to a value smaller than -infinity, while an infinite
!    upper bound can be specified by setting the appropriate element of C_u
!    to a value larger than infinity.
!    On exit, C_l and C_u will most likely have been reordered.
!
!   DC_l, DC_u are REAL arrays of length n, which, if allocated, must
!    be set by the user to the values of the arrays dc_l and dc_u of parametric
!    lower and upper bounds on A x.
!    On exit, present DC_l and DC_u will most likely have been reordered
!
!   get_y is a LOGICAL variable. See Y.
!
!   Y is a REAL array of length m, which are used to store the values
!    of the Lagrange multipliers corresponding to the general bound constraints
!    c_l <= A x and A x <= c_u. If the user has assigned values
!    to Y, get_y must be .FALSE. on entry. In this case, on exit,
!    Y will most likely have been reordered. If the user does not
!    wish to provide values for Y, get_y must be .TRUE. on entry.
!    On exit it will have been filled with appropriate values.
!
!  map is a structure of type LSP_data_type which holds internal
!   mapping arrays and related data, and which need not be set by the user
!
!  control is a structure of type LSP_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to LSP_initialize. See
!   LSP_initialize for details
!
!  inform is a structure of type LSP_inform_type that provides
!    information on exit from LSP_reorder. The component status
!    has possible values:
!
!     0 Normal termination
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!    -3 one of the restrictions
!          prob%n     >=  1
!          prob%m     >=  0
!          prob%o     >=  1
!          prob%A%type and prob%L%type in 
!            { 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS', 
!              'SPARSE_BY_ROWS', 'SPARSE_BY_COLUMNS', 'COORDINATE' }
!       has been violated
!
!    -5 The constraints are inconsistent
!
!   -31 an attempt to use LSP_apply/LSP_restore is made prior to a
!        successful call to LSP_reorder
!
!   -32 the storage format has changed without recalling LSP_reorder
!
!   -33 the array L/H have not been reordered, but the given option requires
!       them to have been
!
!   -34 Neither the array prob%Y nor the pair prob%Y_l and prob%Y_u have
!       been allocated
!
!   -35 Neither the array prob%Z nor the pair prob%Z_l and prob%Z_u have
!       been allocated
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSP_map_type ), INTENT( INOUT ) :: map
      TYPE ( QPT_dimensions_type ), INTENT( OUT ) :: dims
      LOGICAL, INTENT( IN ) :: get_x, get_y, get_z
      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LSP_control_type ), INTENT( IN ) :: control
      TYPE ( LSP_inform_type ), INTENT( OUT ) :: inform
      LOGICAL, OPTIONAL :: parametric

!  local variables

      INTEGER :: i, j, k, l, ll
      INTEGER :: free, nonneg, lower, range, upper, nonpos, fixed, equality
      INTEGER :: l_free, l_lower, l_range, l_upper, l_equality
      INTEGER :: a_free, a_fixed
      INTEGER :: d_free, o_free, d_nonneg, o_nonneg, d_lower, o_lower, d_range
      INTEGER :: o_range, d_upper, o_upper, d_nonpos, o_nonpos, d_fixed, o_fixed
      REAL ( KIND = wp ) :: xl, xu, cl, cu, val
      LOGICAL :: apy, apyl, apyu, apz, apzl, apzu
      CHARACTER ( LEN = 80 ) :: array_name

!  check that Y or Y_l/Y_u and Z or Z_l/Z_u has been allocated

      IF ( ALLOCATED( prob%Y ) ) THEN
        apy = SIZE( prob%Y ) >= prob%m
      ELSE
        apy = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Y_l ) ) THEN
        apyl = SIZE( prob%Y_l ) >= prob%m
      ELSE
        apyl = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Y_u ) ) THEN
        apyu = SIZE( prob%Y_u ) >= prob%m
      ELSE
        apyu = .FALSE.
      END IF
      IF ( .NOT. ( apy .OR. ( apyl .AND. apyu ) ) ) THEN
        inform%status = GALAHAD_error_y_unallocated ; RETURN
      END IF

      IF ( ALLOCATED( prob%Z ) ) THEN
        apz = SIZE( prob%Z ) >= prob%n
      ELSE
        apz = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Z_l ) ) THEN
        apzl = SIZE( prob%Z_l ) >= prob%n
      ELSE
        apzl = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Z_u ) ) THEN
        apzu = SIZE( prob%Z_u ) >= prob%n
      ELSE
        apzu = .FALSE.
      END IF
      IF ( .NOT. ( apz .OR. ( apzl .AND. apzu ) ) ) THEN
        inform%status = GALAHAD_error_z_unallocated ; RETURN
      END IF

!  check input parameters

      IF ( prob%n <= 0 .OR. prob%o <= 0 .OR. prob%m < 0 .OR.                   &
           .NOT. QPT_keyword_A( prob%A%type ) .OR.                             &
           .NOT. QPT_keyword_A( prob%L%type ) ) THEN
        inform%status = GALAHAD_error_restrictions ; RETURN
      END IF

!  store original problem dimensions

      map%n = prob%n ; map%m = prob%m ; map%o = prob%o
      CALL QPT_put_A( map%a_type_original, SMT_get( prob%A%type ) )
      CALL QPT_put_A( map%l_type_original, SMT_get( prob%L%type ) )
      IF ( SMT_get( prob%A%type ) == 'COORDINATE' )                            &
         map%a_ne_original = prob%A%ne
      IF ( SMT_get( prob%L%type ) == 'COORDINATE' )                            &
         map%l_ne_original = prob%L%ne

!  see which storage scheme is used for L

!  dense by rows

      IF ( SMT_get( prob%L%type ) == 'DENSE' .OR.                              &
           SMT_get( prob%L%type ) == 'DENSE_BY_ROWS' ) THEN
        map%l_type = dense
        map%l_ne = prob%n * prob%m

!  dense by columns

      ELSE IF ( SMT_get( prob%L%type ) == 'DENSE_BY_COLUMNS' ) THEN
        map%l_type = dense_by_columns
        map%l_ne = prob%n * prob%m

!  row-wise, sparse

      ELSE IF ( SMT_get( prob%L%type ) == 'SPARSE_BY_ROWS' ) THEN
        map%l_type = sparse_by_rows
        map%l_ne = prob%L%ptr( prob%m + 1 ) - 1

!  column-wise, sparse

      ELSE IF ( SMT_get( prob%L%type ) == 'SPARSE_BY_COLUMNS' ) THEN
        map%l_type = sparse_by_columns
        map%l_ne = prob%L%ptr( prob%n + 1 ) - 1

!  co-ordinate, sparse

      ELSE
        map%l_type = coordinate
        map%l_ne = prob%L%ne
      END IF

!  do the same for A

!  dense by rows

      IF ( SMT_get( prob%A%type ) == 'DENSE' .OR.                              &
           SMT_get( prob%A%type ) == 'DENSE_BY_ROWS' ) THEN
        map%a_type = dense
        map%a_ne = prob%n * prob%m

!  dense by columns

      ELSE IF ( SMT_get( prob%A%type ) == 'DENSE_BY_COLUMNS' ) THEN
        map%a_type = dense_by_columns
        map%a_ne = prob%n * prob%m

!  row-wise, sparse

      ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
        map%a_type = sparse_by_rows
        map%a_ne = prob%A%ptr( prob%m + 1 ) - 1

!  column-wise, sparse

      ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_COLUMNS' ) THEN
        map%a_type = sparse_by_columns
        map%a_ne = prob%A%ptr( prob%n + 1 ) - 1

!  co-ordinate, sparse

      ELSE
        map%a_type = coordinate
        map%a_ne = prob%A%ne
      END IF

!  allocate workspace array

      array_name = 'lsp: map%IW'
      CALL SPACE_resize_array( MAX( map%n, map%o, map%m ), map%IW,             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error )
!            exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  subtract A_fx x_fx from observations

!  original co-ordinate storage

      IF ( map%a_type == coordinate ) THEN
        DO l = 1, map%a_ne
          j = prob%A%col( l ) ; xl = prob%X_l( j )
          IF ( xl == prob%X_u( j ) ) THEN
            i = prob%A%row( l )
            prob%B( i ) = prob%B( i ) - prob%A%val( l ) * xl
          END IF
        END DO

!  original row-wise storage

      ELSE IF ( map%a_type == sparse_by_rows ) THEN
        DO i = 1, map%o
          DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            j = prob%A%col( l ) ; xl = prob%X_l( j )
            IF ( xl == prob%X_u( j ) ) THEN
              prob%B( i ) = prob%B( i ) - prob%A%val( l ) * xl
            END IF
          END DO
        END DO

!  original row-wise storage

      ELSE IF ( map%a_type == sparse_by_columns ) THEN
        DO j = 1, map%o
          xl = prob%X_l( j )
          IF ( xl == prob%X_u( j ) ) THEN
            DO l = prob%A%ptr( j ), prob%A%ptr( j + 1 ) - 1
              i = prob%A%row( l )
              prob%B( i ) = prob%B( i ) - prob%A%val( l ) * xl
            END DO
          END IF
        END DO

!  original dense row-wise storage

      ELSE  IF ( map%a_type == dense_by_rows .OR.                            &
                 map%a_type == dense_by_rows ) THEN
        l = 0
        DO j = 1, map%n
          xl = prob%X_l( j )
          IF ( xl == prob%X_u( j ) ) THEN
            DO i = 1, map%o
              l = l + 1
              prob%B( i ) = prob%B( i ) - prob%A%val( l ) * xl
            END DO
          ELSE
            l = l + map%o
          END IF
        END DO

!  original dense column-wise storage

      ELSE
        l = 0
        DO i = 1, map%o
          DO j = 1, map%n
            l = l + 1
            xl = prob%X_l( j )
            IF ( xl == prob%X_u( j ) ) THEN
              prob%B( i ) = prob%B( i ) - prob%A%val( l ) * xl
            END IF
          END DO
        END DO
      END IF

!  subtract l_fx x_fx from constraint bounds

!  original co-ordinate storage

      IF ( map%l_type == coordinate ) THEN
        DO l = 1, map%l_ne
          j = prob%L%col( l ) ; xl = prob%X_l( j )
          IF ( xl == prob%X_u( j ) ) THEN
            i = prob%L%row( l ) ; val = prob%L%val( l ) * xl
            IF ( prob%C_l( i ) > - control%infinity )                          &
              prob%C_l( i ) = prob%C_l( i ) - val
            IF ( prob%C_u( i ) < control%infinity )                            &
              prob%C_u( i ) = prob%C_u( i ) - val
          END IF
        END DO

!  original row-wise storage

      ELSE IF ( map%l_type == sparse_by_rows ) THEN
        DO i = 1, map%m
          DO l = prob%L%ptr( i ), prob%L%ptr( i + 1 ) - 1
            j = prob%L%col( l ) ; xl = prob%X_l( j )
            IF ( xl == prob%X_u( j ) ) THEN
              val = prob%L%val( l ) * xl
              IF (prob%C_l( i ) > - control%infinity )                         &
                prob%C_l( i ) = prob%C_l( i ) - val
              IF ( prob%C_u( i ) < control%infinity )                          &
                prob%C_u( i ) = prob%C_u( i ) - val
            END IF
          END DO
        END DO

!  original row-wise storage

      ELSE IF ( map%l_type == sparse_by_columns ) THEN
        DO j = 1, map%m
          xl = prob%X_l( j )
          IF ( xl == prob%X_u( j ) ) THEN
            DO l = prob%L%ptr( j ), prob%L%ptr( j + 1 ) - 1
              i = prob%L%row( l )
                val = prob%L%val( l ) * xl
                IF (prob%C_l( i ) > - control%infinity )                       &
                  prob%C_l( i ) = prob%C_l( i ) - val
                IF ( prob%C_u( i ) < control%infinity )                        &
                  prob%C_u( i ) = prob%C_u( i ) - val
            END DO
          END IF
        END DO

!  original dense row-wise storage

      ELSE  IF ( map%l_type == dense_by_rows .OR.                            &
                 map%l_type == dense_by_rows ) THEN
        l = 0
        DO j = 1, map%n
          xl = prob%X_l( j )
          IF ( xl == prob%X_u( j ) ) THEN
            DO i = 1, map%m
              l = l + 1
              val = prob%L%val( l ) * xl
              IF (prob%C_l( i ) > - control%infinity )                         &
                prob%C_l( i ) = prob%C_l( i ) - val
              IF ( prob%C_u( i ) < control%infinity )                          &
                prob%C_u( i ) = prob%C_u( i ) - val
            END DO
          ELSE
            l = l + map%m
          END IF
        END DO

!  original dense column-wise storage

      ELSE
        l = 0
        DO i = 1, map%m
          DO j = 1, map%n
            l = l + 1
            xl = prob%X_l( j )
            IF ( xl == prob%X_u( j ) ) THEN
              val = prob%L%val( l ) * xl
              IF (prob%C_l( i ) > - control%infinity )                         &
                prob%C_l( i ) = prob%C_l( i ) - val
              IF ( prob%C_u( i ) < control%infinity )                          &
                prob%C_u( i ) = prob%C_u( i ) - val
            END IF
          END DO
        END DO
      END IF

!  =======================================================================
!                         Reorder variables
!  =======================================================================

!  run through the bounds to see how many fall into each of the
!  categories:  free(free), non-negativity(nonneg), lower(lower), range(range),
!  upper(upper), non-positivity (nonpos) and fixed (fixed)

      free = 0 ; nonneg = 0 ; lower = 0
      range = 0 ; upper = 0 ; nonpos = 0 ; fixed = 0

      DO i = 1, map%n
        xl = prob%X_l( i ) ; xu = prob%X_u( i )

!  fixed variable

        IF ( xu == xl ) THEN
          fixed = fixed + 1
          prob%X( i ) = xl
          IF ( get_z ) THEN
            IF ( apz ) prob%Z( i ) = one
            IF ( apzl ) prob%Z_l( i ) = one
            IF ( apzu ) prob%Z_u( i ) = zero
          END IF
        ELSE IF ( xl <= - control%infinity ) THEN

!  free variable

          IF ( xu >= control%infinity ) THEN
            free = free + 1
            IF ( get_x ) prob%X( i ) = zero
            IF ( get_z ) THEN
              IF ( apz ) prob%Z( i ) = zero
              IF ( apzl ) prob%Z_l( i ) = zero
              IF ( apzu ) prob%Z_u( i ) = zero
            END IF
          ELSE

!  non-positivity

            IF ( xu == zero .AND.                                              &
                .NOT. control% treat_zero_bounds_as_general ) THEN
              nonpos = nonpos + 1
              IF ( get_x ) prob%X( i ) = - one
              IF ( get_z ) THEN
                IF ( apz ) prob%Z( i ) = - one
                IF ( apzl ) prob%Z_l( i ) = zero
                IF ( apzu ) prob%Z_u( i ) = - one
              END IF

!  upper bounded variable

            ELSE
              upper = upper + 1
              IF ( get_x ) prob%X( i ) = xu - one
              IF ( get_z ) THEN
                IF ( apz ) prob%Z( i ) = - one
                IF ( apzl ) prob%Z_l( i ) = zero
                IF ( apzu ) prob%Z_u( i ) = - one
              END IF
            END IF
          END IF
        ELSE
          IF ( xu < control%infinity ) THEN

!  inconsistent bounds

            IF ( xu < xl ) THEN
              inform%status = GALAHAD_error_primal_infeasible
              RETURN

!  range bounded variable

            ELSE
              range = range + 1
              IF ( get_x ) prob%X( i ) = half * ( xl + xu )
              IF ( get_z ) THEN
                IF ( apz ) prob%Z( i ) = zero
                IF ( apzl ) prob%Z_l( i ) = zero
                IF ( apzu ) prob%Z_u( i ) = zero
              END IF
            END IF
          ELSE

!  non-negativity

            IF ( xl == zero .AND.                                              &
                .NOT. control% treat_zero_bounds_as_general ) THEN
              nonneg = nonneg + 1
              IF ( get_x ) prob%X( i ) = one
              IF ( get_z ) THEN
                IF ( apz ) prob%Z( i ) = one
                IF ( apzl ) prob%Z_l( i ) = one
                IF ( apzu ) prob%Z_u( i ) = zero
              END IF

!  lower bounded variable

            ELSE
              lower = lower + 1
              IF ( get_x ) prob%X( i ) = xl + one
              IF ( get_z ) THEN
                IF ( apz ) prob%Z( i ) = one
                IF ( apzl ) prob%Z_l( i ) = one
                IF ( apzu ) prob%Z_u( i ) = zero
              END IF
            END IF
          END IF
        END IF
      END DO

!  now set starting addresses for each division of the variables

      d_free = 0
      d_nonneg = d_free + free
      d_lower = d_nonneg + nonneg
      d_range = d_lower + lower
      d_upper = d_range + range
      d_nonpos = d_upper + upper
      d_fixed = d_nonpos + nonpos

!  also set the starting and ending addresses as required

      dims%x_free = d_nonneg
      dims%x_l_start = d_lower + 1
      dims%x_u_start = d_range + 1
      dims%x_l_end = d_upper
      dims%x_u_end = d_nonpos
      prob%n = d_fixed
      map%n_reordered = prob%n

!  allocate workspace arrays and permutation map for the variables

      array_name = 'lsp: map%x_map'
      CALL SPACE_resize_array( map%n, map%x_map,                               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  run through the variable bounds for a second time, this time building
!  the mapping array

      DO i = 1, map%n
        xl = prob%X_l( i ) ; xu = prob%X_u( i )

!  fixed variable

        IF ( xu == xl ) THEN
          d_fixed = d_fixed + 1
          map%x_map( i ) = d_fixed
        ELSE IF ( xl <= - control%infinity ) THEN

!  free variable

          IF ( xu >= control%infinity ) THEN
            d_free = d_free + 1
            map%x_map( i ) = d_free

!  non-positivity

          ELSE
            IF ( xu == zero .AND.                                              &
                .NOT. control% treat_zero_bounds_as_general ) THEN
              d_nonpos = d_nonpos + 1
              map%x_map( i ) = d_nonpos

!  upper bounded variable

            ELSE
              d_upper = d_upper + 1
              map%x_map( i ) = d_upper
            END IF
          END IF

!  range bounded variable

        ELSE
          IF ( xu < control%infinity ) THEN
            d_range = d_range + 1
            map%x_map( i ) = d_range

!  non-negativity

          ELSE
            IF ( xl == zero .AND.                                              &
                .NOT. control% treat_zero_bounds_as_general ) THEN
              d_nonneg = d_nonneg + 1
              map%x_map( i ) = d_nonneg

!  lower bounded variable

            ELSE
              d_lower = d_lower + 1
              map%x_map( i ) = d_lower
            END IF
          END IF
        END IF
      END DO

!  move the bounds on variables

      CALL SORT_inplace_permute( map%n, map%x_map, X = prob%X_l( : map%n ) )
      CALL SORT_inplace_permute( map%n, map%x_map, X = prob%X_u( : map%n ) )
      CALL SORT_inplace_permute( map%n, map%x_map, X = prob%X( : map%n ) )
      IF ( apz )                                                               &
        CALL SORT_inplace_permute( map%n, map%x_map, X = prob%Z( : map%n ) )
      IF ( apzl )                                                              &
        CALL SORT_inplace_permute( map%n, map%x_map, X = prob%Z_l( : map%n ) )
      IF ( apzu )                                                              &
        CALL SORT_inplace_permute( map%n, map%x_map, X = prob%Z_u( : map%n ) )
      IF ( PRESENT( parametric ) ) THEN
        IF ( ALLOCATED( prob%DX_l ) ) THEN
          IF ( SIZE( prob%DX_l ) == map%n )                                    &
            CALL SORT_inplace_permute( map%n, map%x_map, X = prob%DX_l(:map%n) )
        END IF
        IF ( ALLOCATED( prob%DX_u ) ) THEN
          IF ( SIZE( prob%DX_u ) == map%n )                                    &
            CALL SORT_inplace_permute( map%n, map%x_map, X = prob%DX_u(:map%n) )
        END IF
      END IF

!  =======================================================================
!                         Reorder objective Jacobian
!  =======================================================================

!  allocate the inverse mapping array for A

      array_name = 'lsp: map%l_map_inverse'
      CALL SPACE_resize_array( MAX( map%o, map%a_ne ), map%a_map_inverse,      &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  count how many entries there are in each row of A. map%IW(i) gives
!  the number of entries in row i

!  original dense storage; record the column indices

      IF ( map%a_type == dense .OR. map%a_type == dense_by_rows .OR.       &
           map%a_type == dense_by_columns ) THEN
        map%IW( : map%o ) = prob%n

!  sparse storage formats

      ELSE
        map%IW( : map%o ) = 0

!  original co-ordinate storage

        IF ( map%a_type == coordinate ) THEN
          DO l = 1, map%a_ne
            IF ( map%x_map( prob%A%col( l ) ) <= prob%n ) THEN
              i = prob%A%row( l )
              map%IW( i ) = map%IW( i ) + 1
            END IF
          END DO

!  original row-wise storage

        ELSE IF ( map%a_type == sparse_by_rows ) THEN
          DO i = 1, map%o
            DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
              IF ( map%x_map( prob%A%col( l ) ) <= prob%n )                    &
                map%IW( i ) = map%IW( i ) + 1
            END DO
          END DO

!  original column-wise storage

        ELSE
          DO j = 1, map%n
            IF ( map%x_map( j ) <= prob%n ) THEN
              DO l = prob%A%ptr( j ), prob%A%ptr( j + 1 ) - 1
                i = prob%A%row( l )
                map%IW( i ) = map%IW( i ) + 1
              END DO
            END IF
          END DO
        END IF
      END IF

!  count the number of terms in the objective once fixed variables have been
!  excluded

      prob%o = COUNT( map%IW( : map%o ) > 0 )
      map%o_reordered = prob%o

!  allocate permutation map for the constraints

      array_name = 'lsp: map%c_map'
      CALL SPACE_resize_array( map%o, map%a_map,                               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  run through the rows of A to build the mapping array, and set the constant 
!  term in the objective for rows with no free variables

      a_free = 0 ; a_fixed = prob%o ; prob%f = zero
      DO i = 1, map%o
        IF ( map%IW( i ) > 0 ) THEN
          a_free = a_free + 1
          map%a_map( i ) = a_free
        ELSE
          a_fixed = a_fixed + 1
          map%a_map( i ) = a_fixed
          prob%f = prob%f + half * prob%B( i ) ** 2
        END IF
      END DO

!  move the observations

      CALL SORT_inplace_permute( map%o, map%a_map, X = prob%B )
      IF ( PRESENT( parametric ) )                                             &
        CALL SORT_inplace_permute( map%o, map%a_map, X = prob%DB )

!  now permute the rows and columns of A. Start by counting how many entries
!  will be required for each column. map%IW(j) gives the number of entries in 
!  column j

!  original dense storage; record the column indices

      IF ( map%a_type == dense .OR. map%a_type == dense_by_rows .OR.           &
           map%a_type == dense_by_columns ) THEN
        map%IW( : map%n ) = prob%o
      ELSE
        map%IW( : map%n ) = 0

!  original co-ordinate storage

        IF ( map%a_type == coordinate ) THEN
          DO l = 1, map%a_ne
            j = map%x_map( prob%A%col( l ) )
            map%IW( j ) = map%IW( j ) + 1
          END DO

!  original row-wise storage

        ELSE IF ( map%a_type == sparse_by_rows ) THEN
          DO k = 1, map%o
            DO l = prob%A%ptr( k ), prob%A%ptr( k + 1 ) - 1
              j = map%x_map( prob%A%col( l ) )
              map%IW( j ) = map%IW( j ) + 1
            END DO
          END DO

!  original column-wise storage

        ELSE
          DO k = 1, map%n
            j = map%x_map( k )
            map%IW( j ) = prob%A%ptr( k + 1 ) - prob%A%ptr( k )
          END DO
        END IF
      END IF

!  set starting addresses prior to making the permutation

      i = 1
      DO j = 1, map%n
        k = i ; i = i + map%IW( i )
        map%IW( j ) = k
      END DO

!  allocate the inverse mapping array for A

      array_name = 'lsp: map%a_map_inverse'
      CALL SPACE_resize_array( map%a_ne, map%a_map_inverse,                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  reorder the rows; compute the mapping array for A and renumber its rows.
!  Ensure that there is enough space to store the result in a row-wise scheme

      IF ( map%a_type == coordinate .OR. map%a_type == dense_by_rows .OR.  &
           map%a_type == dense_by_columns .OR. map%a_type == dense .OR.    &
           map%a_type == sparse_by_rows ) THEN
        array_name = 'lsp: prob%A%row'
        CALL SPACE_resize_array( map%a_ne, prob%A%row,                         &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

      IF ( map%a_type == coordinate .OR. map%a_type == dense_by_rows .OR.  &
           map%a_type == dense_by_columns .OR. map%a_type == dense ) THEN
        array_name = 'lsp: prob%A%ptr'
        CALL SPACE_resize_array( map%n + 1, prob%A%ptr,                        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

!  original dense by rows storage

      IF ( map%a_type == dense .OR. map%a_type == dense_by_rows ) THEN
        l = 0
        DO ll = 1, map%o
          i = map%a_map( ll ) 
          DO k = 1, map%n
            l = l + 1 ; j = map%x_map( k )
            map%a_map_inverse( map%IW( j ) ) = l
            prob%A%row( l ) = i
            map%IW( j ) = map%IW( j ) + 1
          END DO
        END DO

!  original dense by columns storage

      ELSE IF ( map%a_type == dense_by_columns ) THEN
        l = 0
        DO k = 1, map%n
          j = map%x_map( k )
          DO ll = 1, map%o
            l = l + 1 ; i = map%a_map( ll )
            map%a_map_inverse( map%IW( j ) ) = l
            prob%A%row( l ) = i
            map%IW( j ) = map%IW( j ) + 1
          END DO
        END DO

!  original co-ordinate storage

      ELSE IF ( map%a_type == coordinate ) THEN
        DO l = 1, map%a_ne
          i = map%a_map( prob%A%row( l ) ) ; j = map%x_map( prob%A%col( l ) )
          map%a_map_inverse( map%IW( j ) ) = l
          prob%A%row( l ) = i
          map%IW( j ) = map%IW( j ) + 1
        END DO

!  original row-wise storage

      ELSE IF ( map%a_type == sparse_by_rows ) THEN
        DO k = 1, map%o
          i = map%a_map( k )
          DO l = prob%A%ptr( k ), prob%A%ptr( k + 1 ) - 1
            j = map%x_map( prob%A%col( l ) )
            map%a_map_inverse( map%IW( j ) ) = l
            prob%A%row( l ) = i
            map%IW( j ) = map%IW( j ) + 1
          END DO
        END DO

!  original column-wise storage

      ELSE
        DO k = 1, map%n
          j = map%x_map( k )
          DO l = prob%A%ptr( k ), prob%A%ptr( k + 1 ) - 1
            i = map%a_map( prob%A%row( l ) )
            map%a_map_inverse( map%IW( j ) ) = l
            prob%A%row( l ) = i
            map%IW( j ) = map%IW( j ) + 1
          END DO
        END DO
      END IF

!  set the starting addresses for each column in the permuted matrix

      prob%A%ptr( 1 ) = 1
      DO i = 1, map%n
        prob%A%ptr( i + 1 ) = map%IW( i )
      END DO

!  apply the reordering to A

      CALL SORT_inverse_permute( map%a_ne, map%a_map_inverse, X = prob%A%val,  &
                                 IX = prob%A%row )

!  reorder the rows in each column so that they appear in increasing order

      CALL LSP_order_rows( map%n, prob%A%val, prob%A%row, prob%A%ptr,          &
                           map%a_map_inverse )

!  if the original storage scheme was by co-ordinates, record the
!  final column indices

      IF ( map%a_type == coordinate ) THEN
        DO j = 1, prob%n
          prob%A%col( prob%A%ptr( j ) : prob%A%ptr( j + 1 ) - 1 ) = j
        END DO
      END IF

!  =======================================================================
!                         Reorder constraints
!  =======================================================================

!  allocate the inverse mapping array for L

      array_name = 'lsp: map%l_map_inverse'
      CALL SPACE_resize_array( MAX( map%m, map%l_ne ), map%l_map_inverse,      &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  count how many entries there are in each constraint. map%IW(i) gives
!  the number of entries in row i

!  original dense storage; record the column indices

      IF ( map%l_type == dense .OR. map%l_type == dense_by_rows .OR.       &
           map%l_type == dense_by_columns ) THEN
        map%IW( : map%m ) = prob%n
      ELSE
        map%IW( : map%m ) = 0

!  original co-ordinate storage

        IF ( map%l_type == coordinate ) THEN
          DO l = 1, map%l_ne
            IF ( map%x_map( prob%L%col( l ) ) <= prob%n ) THEN
              i = prob%L%row( l )
              map%IW( i ) = map%IW( i ) + 1
            END IF
          END DO

!  original row-wise storage

        ELSE IF ( map%l_type == sparse_by_rows ) THEN
          DO i = 1, map%m
            DO l = prob%L%ptr( i ), prob%L%ptr( i + 1 ) - 1
              IF ( map%x_map( prob%L%col( l ) ) <= prob%n )                    &
                map%IW( i ) = map%IW( i ) + 1
            END DO
          END DO

!  original column-wise storage

        ELSE
          DO j = 1, map%n
            IF ( map%x_map( j ) <= prob%n ) THEN
              DO l = prob%L%ptr( j ), prob%L%ptr( j + 1 ) - 1
                i = prob%L%row( l )
                map%IW( i ) = map%IW( i ) + 1
              END DO
            END IF
          END DO
        END IF
      END IF

!  run through the constraint bounds to see how many fall into each of the
!  categories:  free, lower, range, upper and equality

      free = 0 ; lower = 0 ; range = 0 ; upper = 0 ; equality = 0

      DO i = 1, map%m
        cl = prob%C_l( i ) ; cu = prob%C_u( i )

!  equality constraint

        IF ( cu == cl ) THEN
          IF ( map%IW( i ) > 0 ) THEN
            equality = equality + 1
            IF ( get_y ) THEN
              IF ( apy ) prob%Y( i ) = one
              IF ( apyl ) prob%Y_l( i ) = one
              IF ( apyu ) prob%Y_u( i ) = zero
            END IF
          ELSE

!  deal with null equality constraint

            IF ( cu == zero ) THEN
              free = free + 1
              IF ( get_y ) THEN
                IF ( apy ) prob%Y( i ) = zero
                IF ( apyl ) prob%Y_l( i ) = zero
                IF ( apyu ) prob%Y_u( i ) = zero
              END IF
            ELSE
              inform%status = GALAHAD_error_primal_infeasible
              RETURN
            END IF
          END IF

        ELSE IF ( cl <= - control%infinity ) THEN

!  free constraint

          IF ( cu >= control%infinity ) THEN
            free = free + 1
            IF ( get_y ) THEN
              IF ( apy ) prob%Y( i ) = zero
              IF ( apyl ) prob%Y_l( i ) = zero
              IF ( apyu ) prob%Y_u( i ) = zero
            END IF
          ELSE

!  upper bounded constraint

            IF ( map%IW( i ) > 0 ) THEN
              upper = upper + 1
              IF ( get_y ) THEN
                IF ( apy ) prob%Y( i ) = - one
                IF ( apyl ) prob%Y_l( i ) = zero
                IF ( apyu ) prob%Y_u( i ) = - one
              END IF
            ELSE

!  deal with null upper bounded constraint

              IF ( cu >= zero ) THEN
                free = free + 1
                IF ( get_y ) THEN
                  IF ( apy ) prob%Y( i ) = zero
                  IF ( apyl ) prob%Y_l( i ) = zero
                  IF ( apyu ) prob%Y_u( i ) = zero
                END IF
              ELSE
                inform%status = GALAHAD_error_primal_infeasible
                RETURN
              END IF
            END IF
          END IF
        ELSE
          IF ( cu < control%infinity ) THEN

!  inconsistent constraints

            IF ( cu < cl ) THEN
              inform%status = GALAHAD_error_primal_infeasible
              RETURN

!  range bounded constraint

            ELSE
              IF ( map%IW( i ) > 0 ) THEN
                range = range + 1
                IF ( get_y ) THEN
                  IF ( apy ) prob%Y( i ) = one
                  IF ( apyl ) prob%Y_l( i ) = one
                  IF ( apyu ) prob%Y_u( i ) = zero
                END IF
              ELSE

!  deal with null range bounded constraint

                IF ( cl <= zero .AND. cu >= zero ) THEN
                  free = free + 1
                  IF ( get_y ) THEN
                    IF ( apy ) prob%Y( i ) = zero
                    IF ( apyl ) prob%Y_l( i ) = zero
                    IF ( apyu ) prob%Y_u( i ) = zero
                  END IF
                ELSE
                  inform%status = GALAHAD_error_primal_infeasible
                  RETURN
                END IF
              END IF
            END IF
          ELSE

!  lower bounded constraint

            IF ( map%IW( i ) > 0 ) THEN
              lower = lower + 1
              IF ( get_y ) THEN
                IF ( apy ) prob%Y( i ) = one
                IF ( apyl ) prob%Y_l( i ) = one
                IF ( apyu ) prob%Y_u( i ) = zero
              END IF
            ELSE

!  deal with null lower bounded constraint

              IF ( cl <= zero ) THEN
                free = free + 1
                IF ( get_y ) THEN
                  IF ( apy ) prob%Y( i ) = zero
                  IF ( apyl ) prob%Y_l( i ) = zero
                  IF ( apyu ) prob%Y_u( i ) = zero
                END IF
              ELSE
                inform%status = GALAHAD_error_primal_infeasible
                RETURN
              END IF
            END IF
          END IF
        END IF
      END DO

!  now set starting addresses for each division of the constraints

      l_equality = 0
      l_lower = l_equality + equality
      l_range = l_lower + lower
      l_upper = l_range + range
      l_free = l_upper + upper

!  also set the starting and ending addresses as required

      dims%c_equality = equality
      dims%c_l_start = l_lower + 1
      dims%c_u_start = l_range + 1

      dims%c_l_end = l_upper
      dims%c_u_end = l_free
      prob%m = l_free
      map%m_reordered = prob%m

!  allocate permutation map for the constraints

      array_name = 'lsp: map%c_map'
      CALL SPACE_resize_array( map%m, map%c_map,                               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  run through the bounds for a second time, this time building the mapping
!  array

      DO i = 1, map%m
        cl = prob%C_l( i ) ; cu = prob%C_u( i )

!  equality constraint

        IF ( cu == cl ) THEN
          IF ( map%IW( i ) > 0 ) THEN
            l_equality = l_equality + 1
            map%c_map( i ) = l_equality
          ELSE
            l_free = l_free + 1
            map%c_map( i ) = l_free
          END IF
        ELSE IF ( cl <= - control%infinity ) THEN

!  free constraint

          IF ( cu >= control%infinity ) THEN
            l_free = l_free + 1
            map%c_map( i ) = l_free
          ELSE

!  upper bounded constraint

            IF ( map%IW( i ) > 0 ) THEN
              l_upper = l_upper + 1
              map%c_map( i ) = l_upper
            ELSE
              l_free = l_free + 1
              map%c_map( i ) = l_free
            END IF
          END IF
        ELSE

          IF ( cu < control%infinity ) THEN

!  range bounded constraint

            IF ( map%IW( i ) > 0 ) THEN
              l_range = l_range + 1
              map%c_map( i ) = l_range
            ELSE
              l_free = l_free + 1
              map%c_map( i ) = l_free
            END IF
          ELSE

!  lower bounded constraint

            IF ( map%IW( i ) > 0 ) THEN
              l_lower = l_lower + 1
              map%c_map( i ) = l_lower
            ELSE
              l_free = l_free + 1
              map%c_map( i ) = l_free
            END IF
          END IF
        END IF
      END DO

!  move the constraint values and bounds

      CALL SORT_inplace_permute( map%m, map%c_map, X = prob%C_l )
      CALL SORT_inplace_permute( map%m, map%c_map, X = prob%C_u )
      IF ( apy ) CALL SORT_inplace_permute( map%m, map%c_map, X = prob%Y )
      IF ( apyl ) CALL SORT_inplace_permute( map%m, map%c_map, X = prob%Y_l )
      IF ( apyu ) CALL SORT_inplace_permute( map%m, map%c_map, X = prob%Y_u )
      IF ( PRESENT( parametric ) ) THEN
        IF ( ALLOCATED( prob%DC_l ) ) THEN
          IF ( SIZE( prob%DC_l ) == map%m )                                    &
            CALL SORT_inplace_permute( map%m, map%c_map, X = prob%DC_l )
        END IF
        IF ( ALLOCATED( prob%DC_u ) ) THEN
          IF ( SIZE( prob%DC_u ) == map%m )                                    &
            CALL SORT_inplace_permute( map%m, map%c_map, X = prob%DC_u )
        END IF
      END IF

!  now permute the rows and columns of L. Start by counting how many entries
!  will be required for each row. map%IW(i) gives the number of entries in row i
!  Also record the number of nonzeros in each COLUMN corresponding to
!  fixed variables. map%ptr_l_fixed(j) gives the number in permuted column j

      IF ( prob%n < map%n ) THEN
        array_name = 'lsp: map%ptr_l_fixed'
        CALL SPACE_resize_array(prob%n + 1, map%n + 1, map%ptr_l_fixed,        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error,              &
               exact_size = .TRUE. )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

!  original dense storage; record the column indices

      IF ( map%l_type == dense .OR. map%l_type == dense_by_rows .OR.       &
           map%l_type == dense_by_columns ) THEN
        map%IW( : map%m ) = prob%n
        IF ( prob%n < map%n ) map%ptr_l_fixed( prob%n + 1 : map%n ) = map%m
      ELSE
        map%IW( : map%m ) = 0
        IF ( prob%n < map%n ) map%ptr_l_fixed( prob%n + 1 : map%n ) = 0

!  original co-ordinate storage

        IF ( map%l_type == coordinate ) THEN
          DO l = 1, map%l_ne
            j = map%x_map( prob%L%col( l ) )
            IF ( j <= prob%n ) THEN
              i = map%c_map( prob%L%row( l ) )
              map%IW( i ) = map%IW( i ) + 1
            ELSE
              map%ptr_l_fixed( j ) = map%ptr_l_fixed( j ) + 1
            END IF
          END DO

!  original row-wise storage

        ELSE IF ( map%l_type == sparse_by_rows ) THEN
          DO k = 1, map%m
            i = map%c_map( k )
            DO l = prob%L%ptr( k ), prob%L%ptr( k + 1 ) - 1
              j = map%x_map( prob%L%col( l ) )
              IF ( j <= prob%n ) THEN
                map%IW( i ) = map%IW( i ) + 1
              ELSE
                map%ptr_l_fixed( j ) = map%ptr_l_fixed( j ) + 1
              END IF
            END DO
          END DO

!  original column-wise storage

        ELSE
          DO k = 1, map%n
            j = map%x_map( k )
            IF ( j <= prob%n ) THEN
              DO l = prob%L%ptr( k ), prob%L%ptr( k + 1 ) - 1
                i = map%c_map( prob%L%row( l ) )
                map%IW( i ) = map%IW( i ) + 1
              END DO
            ELSE
              map%ptr_l_fixed( j ) = map%ptr_l_fixed( j )                      &
                + prob%L%ptr( k + 1 ) -  prob%L%ptr( k  )
            END IF
          END DO
        END IF
      END IF

!  set starting addresses prior to making the permutation

      j = 1
      DO i = 1, map%m
        k = j
        j = j + map%IW( i )
        map%IW( i ) = k
      END DO

      DO i = prob%n + 1, map%n
        k = j
        j = j + map%ptr_l_fixed( i )
        map%ptr_l_fixed( i ) = k
      END DO

!  allocate the inverse mapping array for L

      array_name = 'lsp: map%l_map_inverse'
      CALL SPACE_resize_array( map%l_ne, map%l_map_inverse,                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  reorder the rows; compute the mapping array for L and renumber its columns
!  NB. Any columns corresponding to FIXED variables, will have been
!  moved to the end of L, and will be stored by COLUMN not by row. In
!  particular, l_col for these entries gives the row and not the column
!  number

!  ensure that there is enough space to store the result in a row-wise scheme

      IF ( map%l_type == coordinate .OR. map%l_type == dense_by_rows .OR.  &
           map%l_type == dense_by_columns .OR. map%l_type == dense .OR.    &
           map%l_type == sparse_by_columns ) THEN
        array_name = 'lsp: prob%L%col'
        CALL SPACE_resize_array( map%l_ne, prob%L%col,                         &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

      IF ( map%l_type == coordinate .OR. map%l_type == dense_by_rows .OR.  &
           map%l_type == dense_by_columns .OR. map%l_type == dense ) THEN
        array_name = 'lsp: prob%L%ptr'
        CALL SPACE_resize_array( map%m + 1, prob%L%ptr,                        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

!  original dense by rows storage

      IF ( map%l_type == dense .OR. map%l_type == dense_by_rows ) THEN
        l = 0
        DO ll = 1, map%m
          i = map%c_map( ll ) 
          DO k = 1, map%n
            l = l + 1
            j = map%x_map( k )
            IF ( j <= prob%n ) THEN
              map%l_map_inverse( map%IW( i ) ) = l
              prob%L%col( l ) = j
              map%IW( i ) = map%IW( i ) + 1
            ELSE
              map%l_map_inverse( map%ptr_l_fixed( j ) ) = l
              prob%L%col( l ) = i
              map%ptr_l_fixed( j ) = map%ptr_l_fixed( j ) + 1
            END IF
          END DO
        END DO

!  original dense by columns storage

      ELSE IF ( map%l_type == dense_by_columns ) THEN
        l = 0
        DO k = 1, map%n
          j = map%x_map( k )
          DO ll = 1, map%m
            l = l + 1
            i = map%c_map( ll )
            IF ( j <= prob%n ) THEN
              map%l_map_inverse( map%IW( i ) ) = l
              prob%L%col( l ) = j
              map%IW( i ) = map%IW( i ) + 1
            ELSE
              map%l_map_inverse( map%ptr_l_fixed( j ) ) = l
              prob%L%col( l ) = i
              map%ptr_l_fixed( j ) = map%ptr_l_fixed( j ) + 1
            END IF
          END DO
        END DO

!  original co-ordinate storage

      ELSE IF ( map%l_type == coordinate ) THEN
        DO l = 1, map%l_ne
          i = map%c_map( prob%L%row( l ) ) ; j = map%x_map( prob%L%col( l ) )
          IF ( j <= prob%n ) THEN
            map%l_map_inverse( map%IW( i ) ) = l
            prob%L%col( l ) = j
            map%IW( i ) = map%IW( i ) + 1
          ELSE
            map%l_map_inverse( map%ptr_l_fixed( j ) ) = l
            prob%L%col( l ) = i
            map%ptr_l_fixed( j ) = map%ptr_l_fixed( j ) + 1
          END IF
        END DO

!  original row-wise storage

      ELSE IF ( map%l_type == sparse_by_rows ) THEN
        DO k = 1, map%m
          i = map%c_map( k )
          DO l = prob%L%ptr( k ), prob%L%ptr( k + 1 ) - 1
            j = map%x_map( prob%L%col( l ) )
            IF ( j <= prob%n ) THEN
              map%l_map_inverse( map%IW( i ) ) = l
              prob%L%col( l ) = j
              map%IW( i ) = map%IW( i ) + 1
            ELSE
              map%l_map_inverse( map%ptr_l_fixed( j ) ) = l
              prob%L%col( l ) = i
              map%ptr_l_fixed( j ) = map%ptr_l_fixed( j ) + 1
            END IF
          END DO
        END DO

!  original column-wise storage

      ELSE
        DO k = 1, map%n
          j = map%x_map( k )
          DO l = prob%L%ptr( k ), prob%L%ptr( k + 1 ) - 1
            i = map%c_map( prob%L%row( l ) )
            IF ( j <= prob%n ) THEN
              map%l_map_inverse( map%IW( i ) ) = l
              prob%L%col( l ) = j
              map%IW( i ) = map%IW( i ) + 1
            ELSE
              map%l_map_inverse( map%ptr_l_fixed( j ) ) = l
              prob%L%col( l ) = i
              map%ptr_l_fixed( j ) = map%ptr_l_fixed( j ) + 1
            END IF
          END DO
        END DO
      END IF

!  set the starting addresses for each row in the permuted matrix

      prob%L%ptr( 1 ) = 1
      DO i = 1, map%m
        prob%L%ptr( i + 1 ) = map%IW( i )
      END DO

      IF ( prob%n < map%n ) THEN
        DO i = map%n, prob%n + 1, - 1
          map%ptr_l_fixed( i + 1 ) = map%ptr_l_fixed( i )
        END DO
        map%ptr_l_fixed( prob%n + 1 ) = prob%L%ptr( map%m + 1 )
      END IF

!  apply the reordering to A

      CALL SORT_inverse_permute( map%l_ne, map%l_map_inverse, X = prob%L%val,  &
                                 IX = prob%L%col )

!  reorder the columns in each row so that they appear in increasing order

      CALL LSP_order_rows( map%m, prob%L%val, prob%L%col, prob%L%ptr,          &
                           map%l_map_inverse )

!  if the original storage scheme was by co-ordinates, record the
!  final row indices

      IF ( map%l_type == coordinate ) THEN
        DO i = 1, prob%m
          prob%L%row( prob%L%ptr( i ) : prob%L%ptr( i + 1 ) - 1 ) = i
        END DO
      END IF

!  now evaluate c = A x corresponding to the free variables

      prob%C( : map%m ) = zero
      CALL LSP_LX( map, prob%X( : map%n ), prob%L%val, prob%L%col,             &
                   prob%L%ptr, map%m, prob%C( : map%m ) )

      map%set = .TRUE.
      map%a_perm = .TRUE. ; map%l_perm = .TRUE.
      prob%A%ne = prob%A%ptr( prob%n + 1 ) - 1
      prob%L%ne = prob%L%ptr( prob%m + 1 ) - 1
      CALL QPT_put_A( prob%A%type, 'SPARSE_BY_COLUMNS' )
      CALL QPT_put_A( prob%L%type, 'SPARSE_BY_ROWS' )

      inform%status = GALAHAD_ok
      RETURN

!  Error returns

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
!     IF ( control%error > 0 )                                                 &
!       WRITE( control%error, 2900 ) inform%bad_alloc, inform%alloc_status
      RETURN

!  Non-executable statements

!2900 FORMAT( ' ** Message from -LSP_reorder-', /,                             &
!             ' Allocation error, for ', A20, /, ' status = ', I6 )

!  End of LSP_reorder

      END SUBROUTINE LSP_reorder

!-*-*-*-*-*-*-*-*-   L S P _ a p p l y   S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE LSP_apply( map, inform, prob, get_all,                        &
                            get_all_parametric, get_b, get_db,                 &
                            get_x, get_y, get_z, get_x_bounds,                 &
                            get_dx_bounds, get_c, get_c_bounds,                &
                            get_dc_bounds, get_A, get_L )

!      .........................................
!      .                                       .
!      .  Apply the permutations computed in   .
!      .  LSP_reorder to the data prob as      .
!      .  appropriate                          .
!      .                                       .
!      .........................................

!  Arguments:
!  =========
!
!   prob    see Subroutine LSP_reorder
!   map     see Subroutine LSP_initialize
!
!   inform is a structure of type LSP_inform_type that provides
!    information on exit from LSP_apply. The component status
!    has possible values:
!
!     0 Normal termination.
!
!     1 The mapping arrays have not yet been set. Either LSP_reorder
!       has not yet been called, or the call was unsuccessful.
!
!   get_all       LOGICAL, OPTIONAL. If present, process the entire problem
!   get_all_parametric  LOGICAL, OPTIONAL. If present, process the entire
!                 problem including parametric parts
!   get_b         LOGICAL, OPTIONAL. If present, process b
!   get_db        LOGICAL, OPTIONAL. If present, process db
!   get_c         LOGICAL, OPTIONAL. If present, process c
!   get_x_bounds  LOGICAL, OPTIONAL. If present, process x_l and x_u
!   get_dx_bounds LOGICAL, OPTIONAL. If present, process dx_l and dx_u
!   get_c_bounds  LOGICAL, OPTIONAL. If present, process c_l and c_u
!   get_dc_bounds LOGICAL, OPTIONAL. If present, process dc_l and dc_u
!   get_x         LOGICAL, OPTIONAL. If present, process x
!   get_y         LOGICAL, OPTIONAL. If present, process y
!   get_z         LOGICAL, OPTIONAL. If present, process z
!   get_A         LOGICAL, OPTIONAL. If present, process A
!   get_L         LOGICAL, OPTIONAL. If present, process L

!  Dummy arguments

      TYPE ( LSP_map_type ), INTENT( INOUT ) :: map
      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LSP_inform_type ), INTENT( OUT ) :: inform
      LOGICAL, OPTIONAL, INTENT( IN ) :: get_all, get_b, get_db,               &
                                         get_c, get_x_bounds, get_c_bounds,    &
                                         get_dx_bounds, get_dc_bounds,         &
                                         get_x, get_y, get_z, get_A, get_L,    &
                                         get_all_parametric

!  Local variables

      INTEGER :: i, j, k, l, ll
      LOGICAL :: apy, apyl, apyu, apz, apzl, apzu

!  check that Y or Y_l/Y_u and Z or Z_l/Z_u has been allocated

      IF ( ALLOCATED( prob%Y ) ) THEN
        apy = SIZE( prob%Y ) >= prob%m
      ELSE
        apy = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Y_l ) ) THEN
        apyl = SIZE( prob%Y_l ) >= prob%m
      ELSE
        apyl = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Y_u ) ) THEN
        apyu = SIZE( prob%Y_u ) >= prob%m
      ELSE
        apyu = .FALSE.
      END IF
      IF ( .NOT. ( apy .OR. ( apyl .AND. apyu ) ) ) THEN
        inform%status = GALAHAD_error_y_unallocated ; RETURN
      END IF

      IF ( ALLOCATED( prob%Z ) ) THEN
        apz = SIZE( prob%Z ) >= prob%n
      ELSE
        apz = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Z_l ) ) THEN
        apzl = SIZE( prob%Z_l ) >= prob%n
      ELSE
        apzl = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Z_u ) ) THEN
        apzu = SIZE( prob%Z_u ) >= prob%n
      ELSE
        apzu = .FALSE.
      END IF
      IF ( .NOT. ( apz .OR. ( apzl .AND. apzu ) ) ) THEN
        inform%status = GALAHAD_error_z_unallocated ; RETURN
      END IF

!  check to see that the mapping arrays have been set

      IF ( .NOT. map%set ) THEN
        inform%status = GALAHAD_error_call_order
        RETURN
      END IF

!  check that the variable and constraint bounds are consistent

      DO i = 1, map%n
        IF ( prob%X_l( i ) > prob%X_u( i ) ) THEN
          inform%status = GALAHAD_error_primal_infeasible
          RETURN
        END IF
      END DO

      DO i = 1, map%m
        IF ( prob%C_l( i ) > prob%C_u( i ) ) THEN
          inform%status = GALAHAD_error_primal_infeasible
          RETURN
        END IF
      END DO

!  check to see that storage formats have not changed

      IF ( ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.        &
             PRESENT( get_A ) ) .AND. .NOT. map%a_perm ) THEN
        IF ( ( map%a_type == dense .AND.                                       &
                 ( SMT_get( prob%A%type ) /= 'DENSE' .OR.                      &
                   SMT_get( prob%A%type ) /= 'DENSE_BY_ROWS' ) )  .OR.         &
             ( map%a_type == dense_by_columns .AND.                            &
                   SMT_get( prob%A%type ) /= 'DENSE_BY_COLUMNS' )  .OR.        &
             ( map%a_type == sparse_by_rows .AND.                              &
                 SMT_get( prob%A%type ) /= 'SPARSE_BY_ROWS' ) .OR.             &
             ( map%a_type == sparse_by_columns .AND.                           &
                 SMT_get( prob%A%type ) /= 'SPARSE_BY_COLUMNS' ) .OR.          &
             ( map%a_type == coordinate .AND.                                  &
                 ( SMT_get( prob%A%type ) /= 'COORDINATE' .OR.                 &
                   prob%A%ne /= map%a_ne ) ) ) THEN
          inform%status = GALAHAD_error_reformat
          RETURN
        END IF
      END IF

      IF ( ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.        &
             PRESENT( get_L ) ) .AND. .NOT. map%l_perm ) THEN
        IF ( ( map%l_type == dense .AND.                                       &
                 ( SMT_get( prob%L%type ) /= 'DENSE' .OR.                      &
                   SMT_get( prob%L%type ) /= 'DENSE_BY_ROWS' ) )  .OR.         &
             ( map%l_type == dense_by_columns .AND.                            &
                   SMT_get( prob%L%type ) /= 'DENSE_BY_COLUMNS' )  .OR.        &
             ( map%l_type == sparse_by_rows .AND.                              &
                 SMT_get( prob%L%type ) /= 'SPARSE_BY_ROWS' ) .OR.             &
             ( map%l_type == sparse_by_columns .AND.                           &
                 SMT_get( prob%L%type ) /= 'SPARSE_BY_COLUMNS' ) .OR.          &
             ( map%l_type == coordinate .AND.                                  &
                 ( SMT_get( prob%L%type ) /= 'COORDINATE' .OR.                 &
                   prob%A%ne /= map%a_ne ) ) ) THEN
          inform%status = GALAHAD_error_reformat
          RETURN
        END IF
      END IF

!  pick up the correct dimensions

      prob%n = map%n_reordered
      prob%m = map%m_reordered
      prob%o = map%o_reordered

!  map A

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_A ) ) THEN

!  the row/column permutations have already been made

        IF ( map%a_perm ) THEN
          CALL SORT_inverse_permute( map%a_ne, map%a_map_inverse,              &
                                     X = prob%A%val )

!  the row/column indices are in their original order
!  compute the number of entries in each column of A, and renumber its rows

        ELSE

!  original dense by rows storage

          IF ( map%a_type == dense .OR. map%a_type == dense_by_rows ) THEN
            l = 0
            DO ll = 1, map%o
              i = map%a_map( ll ) 
              DO k = 1, map%n
                l = l + 1 ; j = map%x_map( k )
                map%a_map_inverse( map%IW( j ) ) = l
                prob%A%row( l ) = i
                map%IW( j ) = map%IW( j ) + 1
              END DO
            END DO

!  original dense by columns storage

          ELSE IF ( map%a_type == dense_by_columns ) THEN
            l = 0
            DO k = 1, map%n
              j = map%x_map( k )
              DO ll = 1, map%o
                l = l + 1 ; i = map%a_map( ll )
                map%a_map_inverse( map%IW( j ) ) = l
                prob%A%row( l ) = i
                map%IW( j ) = map%IW( j ) + 1
              END DO
            END DO

!  original co-ordinate storage

          ELSE IF ( map%a_type == coordinate ) THEN
            DO l = 1, map%a_ne
              i = map%a_map( prob%A%row( l ) )
              j = map%x_map( prob%A%col( l ) )
              map%a_map_inverse( map%IW( j ) ) = l
              prob%A%row( l ) = i
              map%IW( j ) = map%IW( j ) + 1
            END DO

!  original row-wise storage

          ELSE IF ( map%a_type == sparse_by_rows ) THEN
            DO k = 1, map%o
              i = map%a_map( k )
              DO l = prob%A%ptr( k ), prob%A%ptr( k + 1 ) - 1
                j = map%x_map( prob%A%col( l ) )
                map%a_map_inverse( map%IW( j ) ) = l
                prob%A%row( l ) = i
                map%IW( j ) = map%IW( j ) + 1
              END DO
            END DO

!  original column-wise storage

          ELSE
            DO k = 1, map%n
              j = map%x_map( k )
              DO l = prob%A%ptr( k ), prob%A%ptr( k + 1 ) - 1
                i = map%a_map( prob%A%row( l ) )
                map%a_map_inverse( map%IW( j ) ) = l
                prob%A%row( l ) = i
                map%IW( j ) = map%IW( j ) + 1
              END DO
            END DO
          END IF

!  set the starting addresses for each column in the permuted matrix

          prob%A%ptr( 1 ) = 1
          DO i = 1, map%n
            prob%A%ptr( i + 1 ) = map%IW( i )
          END DO

!  apply the reordering to A

          CALL SORT_inverse_permute( map%a_ne, map%a_map_inverse,              &
                                     X = prob%A%val, IX = prob%A%row )

!  if the original storage scheme was by co-ordinates, record the
!  final column indices

          IF ( map%a_type == coordinate ) THEN
            DO j = 1, prob%n
              prob%A%col( prob%A%ptr( j ) : prob%A%ptr( j + 1 ) - 1 ) = j
            END DO
          END IF

          map%a_perm = .TRUE.
          prob%A%ne = prob%A%ptr( prob%m + 1 ) - 1
        END IF
      END IF

!  map L

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_L ) ) THEN

!  the row/column permutations have already been made

        IF ( map%l_perm ) THEN
          CALL SORT_inverse_permute( map%l_ne, map%l_map_inverse,              &
                                     X = prob%L%val )

!  the row/column indices are in their original order

        ELSE

!  original dense row storage; record the column indices

          IF ( map%l_type == dense .OR. map%l_type == dense_by_rows ) THEN

!  compute the number of entries in each row of A, and renumber its columns.
!  NB. Any columns corresponding to FIXED variables, will have been
!  moved to the end of A, and will be stored by COLUMN not by row. In
!  particular, l_col for these entries gives the row and not the column
!  number

            l = 0
            DO ll = 1, map%m
              i = map%c_map( ll )
              map%IW( ll ) = prob%n
              DO k = 1, map%n
                j = map%x_map( k )
                l = l + 1
                IF ( j <= prob%n ) THEN
                  prob%L%col( l ) = j
                ELSE
                  prob%L%col( l ) = i
                END IF
              END DO
            END DO

!  original dense column storage

          ELSE IF ( map%l_type == dense_by_columns ) THEN
            l = 0
            DO k = 1, map%n
              j = map%x_map( k )
              DO ll = 1, map%m
                i = map%c_map( ll )
                l = l + 1
                IF ( j <= prob%n ) THEN
                  prob%L%col( l ) = j
                  map%IW( ll ) = map%IW( ll ) + 1
                ELSE
                  prob%L%col( l ) = i
                END IF
              END DO
            END DO

!  original co-ordinate storage

          ELSE
            map%IW( : map%m ) = 0
            IF ( map%l_type == coordinate ) THEN
              DO l = 1, map%l_ne
                i = map%c_map( prob%L%row( l ) )
                j = map%x_map( prob%L%col( l ) )
                IF ( j <= prob%n ) THEN
                  map%IW( i ) = map%IW( i ) + 1
                  prob%L%col( l ) = j
                ELSE
                  prob%L%col( l ) = i
                END IF
              END DO

!  original row-wise storage

            ELSE IF ( map%l_type == sparse_by_rows ) THEN
              DO k = 1, map%m
                i = map%c_map( k )
                DO l = prob%L%ptr( k ), prob%L%ptr( k + 1 ) - 1
                  j = map%x_map( prob%L%col( l ) )
                  IF ( j <= prob%n ) THEN
                    map%IW( i ) = map%IW( i ) + 1
                    prob%L%col( l ) = j
                  ELSE
                    prob%L%col( l ) = i
                  END IF
                END DO
              END DO

!  original column-wise storage

            ELSE
              DO k = 1, map%n
                j = map%x_map( k )
                DO l = prob%L%ptr( k ), prob%L%ptr( k + 1 ) - 1
                  i = map%c_map( prob%L%row( l ) )
                  IF ( j <= prob%n ) THEN
                    map%IW( i ) = map%IW( i ) + 1
                    prob%L%col( l ) = j
                  ELSE
                    prob%L%col( l ) = i
                  END IF
                END DO
              END DO
            END IF
          END IF

!  set the starting addresses for each row in the permuted matrix

          prob%L%ptr( 1 ) = 1
          DO i = 1, map%m
            prob%L%ptr( i + 1 ) = prob%L%ptr( i ) + map%IW( i )
          END DO

!  apply the reordering to L

          CALL SORT_inverse_permute( map%l_ne, map%l_map_inverse,              &
                                     X = prob%L%val, IX = prob%L%col )

!  if the original storage scheme was by co-ordinates, record the
!  final row indices

          IF ( map%l_type == coordinate ) THEN
            DO i = 1, prob%m
              prob%L%row( prob%L%ptr( i ) : prob%L%ptr( i + 1 ) - 1 ) = i
            END DO
          END IF
          map%l_perm = .TRUE.
          prob%L%ne = prob%L%ptr( prob%m + 1 ) - 1
        END IF
      END IF

!  permute the bounds on x

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.        &
           PRESENT( get_x_bounds ) ) THEN
        CALL SORT_inplace_permute( map%n, map%x_map, X = prob%X_l )
        CALL SORT_inplace_permute( map%n, map%x_map, X = prob%X_u )
      END IF

!  permute the bounds on dx

      IF ( PRESENT( get_all_parametric ) .OR. PRESENT( get_dx_bounds ) ) THEN
        IF ( ALLOCATED( prob%DX_l ) ) THEN
          IF ( SIZE( prob%DX_l ) == map%n )                                    &
            CALL SORT_inplace_permute( map%n, map%x_map, X = prob%DX_l )
        END IF
        IF ( ALLOCATED( prob%DX_u ) ) THEN
          IF ( SIZE( prob%DX_u ) == map%n )                                    &
            CALL SORT_inplace_permute( map%n, map%x_map, X = prob%DX_u )
        END IF
      END IF

!  permute the bounds on x

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_x ) ) THEN
        CALL SORT_inplace_permute( map%n, map%x_map, X = prob%X )
      END IF

!  permute the bounds on y

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_y ) ) THEN
        IF ( apy ) CALL SORT_inplace_permute( map%m, map%c_map, X = prob%Y )
        IF ( apyl ) CALL SORT_inplace_permute( map%m, map%c_map, X = prob%Y_l )
        IF ( apyu ) CALL SORT_inplace_permute( map%m, map%c_map, X = prob%Y_u )
      END IF

!  permute the bounds on z

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_z ) ) THEN
        IF ( apz ) CALL SORT_inplace_permute( map%n, map%x_map, X = prob%Z )
        IF ( apzl ) CALL SORT_inplace_permute( map%n, map%x_map, X = prob%Z_l )
        IF ( apzu ) CALL SORT_inplace_permute( map%n, map%x_map, X = prob%Z_u )
      END IF

!  permute the bounds on b

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_b ) ) THEN
        CALL SORT_inplace_permute( map%o, map%a_map, X = prob%B )
      END IF

!  permute the bounds on db

      IF ( PRESENT( get_all_parametric ) .OR. PRESENT( get_db ) ) THEN
        IF ( ALLOCATED( prob%DB ) ) THEN
          IF ( SIZE( prob%DB ) == map%o )                                      &
            CALL SORT_inplace_permute( map%m, map%a_map, X = prob%DB )
        END IF
      END IF

!  permute the bounds on c

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_c_bounds ) ) THEN
        CALL SORT_inplace_permute( map%m, map%c_map, X = prob%C_l )
        CALL SORT_inplace_permute( map%m, map%c_map, X = prob%C_u )
      END IF

!  permute the bounds on dc

      IF ( PRESENT( get_all_parametric ) .OR. PRESENT( get_dc_bounds ) ) THEN
        IF ( ALLOCATED( prob%DC_l ) ) THEN
          IF ( SIZE( prob%DC_l ) == map%m )                                    &
            CALL SORT_inplace_permute( map%m, map%c_map, X = prob%DC_l )
        END IF
        IF ( ALLOCATED( prob%DC_u ) ) THEN
          IF ( SIZE( prob%DC_u ) == map%m )                                    &
            CALL SORT_inplace_permute( map%m, map%c_map, X = prob%DC_u )
        END IF
      END IF

!  form c = A * x

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_c ) ) THEN
        prob%C( : map%m ) = zero
        CALL LSP_LX( map, prob%X, prob%L%val, prob%L%col, prob%L%ptr,          &
                      map%m, prob%C( : map%m ) )

      END IF

!  transform f, g and the bounds on the constraints to account for
!  fixed variables

      IF ( ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) ) .AND.     &
             prob%n < map%n ) THEN
        CALL LSP_remove_fixed( map, prob,                                      &
                               c_bounds = .TRUE. )
      ELSE IF ( PRESENT( get_c_bounds ) .AND. prob%n < map%n ) THEN
        CALL LSP_remove_fixed( map, prob,                                      &
                               c_bounds = PRESENT( get_c_bounds ) )
      END IF

      inform%status = GALAHAD_ok
      RETURN

!  End of LSP_apply

      END SUBROUTINE LSP_apply

!-*-*-*-*-*-   L S P _ g e t _ v a l u e s  S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE LSP_get_values( map, inform, prob, X_val, Y_val, Z_val )

!      ................................................
!      .                                              .
!      .  Recover the values of x (primal variables), .
!      .  y (Lagrange multipliers for constraints),   .
!      .  and z (dual variables)                      .
!      .                                              .
!      ................................................

!  Arguments:
!  =========
!
!   map     see Subroutine LSP_initialize
!   prob    see Subroutine LSP_reorder
!   X       REAL, OPTIONAL. If present, returns the value of x
!   Y       REAL, OPTIONAL. If present, returns the value of y
!   Z       REAL, OPTIONAL. If present, returns the value of z
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LSP_map_type ), INTENT( IN ) :: map
      TYPE ( LSP_inform_type ), INTENT( OUT ) :: inform
      TYPE ( QPT_problem_type ), INTENT( IN ) :: prob

      REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ), DIMENSION( map%n ) :: X_val
      REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ), DIMENSION( map%n ) :: Z_val
      REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ), DIMENSION( map%m ) :: Y_val

!  check to see that the mapping arrays have been set

      IF ( .NOT. map%set ) THEN
        inform%status = GALAHAD_error_call_order
        RETURN
      END IF

!  recover the appropriate array(s)

      IF ( PRESENT( X_val ) ) X_val( : map%n ) = prob%X( map%x_map( : map%n ) )
      IF ( PRESENT( Y_val ) ) THEN
        IF ( ALLOCATED( prob%Y ) ) THEN
          Y_val( : map%m ) = prob%Y( map%c_map( : map%m ) )
        ELSE IF ( ALLOCATED( prob%Y_l ) .AND. ALLOCATED( prob%Y_u ) ) THEN
          Y_val( : map%m ) = prob%Y_l( map%c_map( : map%m ) ) +                &
                             prob%Y_u( map%c_map( : map%m ) )
        END IF
      END IF
      IF ( PRESENT( Z_val ) ) THEN
        IF ( ALLOCATED( prob%Z ) ) THEN
          Z_val( : map%n ) = prob%Z( map%x_map( : map%n ) )
        ELSE IF ( ALLOCATED( prob%Z_l ) .AND. ALLOCATED( prob%Z_u ) ) THEN
          Z_val( : map%n ) = prob%Z_l( map%x_map( : map%n ) ) +                &
                             prob%Z_u( map%x_map( : map%n ) )
        END IF
      END IF
      inform%status = GALAHAD_ok
      RETURN

!  End of LSP_get_values

      END SUBROUTINE LSP_get_values

!-*-*-*-*-*-*-*-   L S P _ r e s t o r e  S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE LSP_restore( map, inform, prob, get_all,                      &
                              get_all_parametric, get_b, get_db,               &
                              get_x, get_y, get_z, get_x_bounds,               &
                              get_dx_bounds, get_c, get_c_bounds,              &
                              get_dc_bounds, get_A, get_L )

!      ..............................................
!      .                                            .
!      .  Apply the inverse of the permutations     .
!      .  computed in LSP_reorder to the            .
!      .  data prob as appropriate                  .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   prob    see Subroutine LSP_reorder
!   map     see Subroutine LSP_initialize
!
!  inform is a structure of type LSP_inform_type that provides
!    information on exit from LSP_apply. The component status
!    has possible values:
!
!     0 Normal termination.
!
!     1 The mapping arrays have not yet been set. Either LSP_reorder
!      has not yet been called, or the call was unsuccessful.
!
!   get_all       LOGICAL, OPTIONAL. If present, process the entire problem
!   get_all_parametric    LOGICAL, OPTIONAL. If present, process the entire
!                 problem including parametric parts
!   get_b         LOGICAL, OPTIONAL. If present, process b
!   get_db        LOGICAL, OPTIONAL. If present, process db
!   get_x         LOGICAL, OPTIONAL. If present, process x
!   get_y         LOGICAL, OPTIONAL. If present, process y
!   get_z         LOGICAL, OPTIONAL. If present, process z
!   get_x_bounds  LOGICAL, OPTIONAL. If present, process x_l and x_u
!   get_dx_bounds LOGICAL, OPTIONAL. If present, process dx_l and dx_u
!   get_c         LOGICAL, OPTIONAL. If present, process c
!   get_c_bounds  LOGICAL, OPTIONAL. If present, process c_l and c_u
!   get_dc_bounds LOGICAL, OPTIONAL. If present, process dc_l and dc_u
!   get_A         LOGICAL, OPTIONAL. If present, process A
!   get_L         LOGICAL, OPTIONAL. If present, process L

!  Dummy arguments

      TYPE ( LSP_map_type ), INTENT( INOUT ) :: map
      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LSP_inform_type ), INTENT( OUT ) :: inform
      LOGICAL, OPTIONAL, INTENT( IN ) :: get_all, get_b, get_db,               &
                                         get_c, get_x_bounds, get_c_bounds,    &
                                         get_dx_bounds, get_dc_bounds,         &
                                         get_x, get_y, get_z, get_A, get_L,    &
                                         get_all_parametric

!  Local variables

      INTEGER :: i, j, k, l
      LOGICAL :: apy, apyl, apyu, apz, apzl, apzu

!  check that Y or Y_l/Y_u and Z or Z_l/Z_u has been allocated

      IF ( ALLOCATED( prob%Y ) ) THEN
        apy = SIZE( prob%Y ) >= prob%m
      ELSE
        apy = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Y_l ) ) THEN
        apyl = SIZE( prob%Y_l ) >= prob%m
      ELSE
        apyl = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Y_u ) ) THEN
        apyu = SIZE( prob%Y_u ) >= prob%m
      ELSE
        apyu = .FALSE.
      END IF
      IF ( .NOT. ( apy .OR. ( apyl .AND. apyu ) ) ) THEN
        inform%status = GALAHAD_error_y_unallocated ; RETURN
      END IF

      IF ( ALLOCATED( prob%Z ) ) THEN
        apz = SIZE( prob%Z ) >= prob%n
      ELSE
        apz = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Z_l ) ) THEN
        apzl = SIZE( prob%Z_l ) >= prob%n
      ELSE
        apzl = .FALSE.
      END IF
      IF ( ALLOCATED( prob%Z_u ) ) THEN
        apzu = SIZE( prob%Z_u ) >= prob%n
      ELSE
        apzu = .FALSE.
      END IF
      IF ( .NOT. ( apz .OR. ( apzl .AND. apzu ) ) ) THEN
        inform%status = GALAHAD_error_z_unallocated ; RETURN
      END IF

!  check to see that the mapping arrays have been set

      IF ( .NOT. map%set ) THEN
        inform%status = GALAHAD_error_call_order
        RETURN
      END IF

!  check to see if permuted A and L are available

      IF ( ( ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.      &
               PRESENT( get_c ) .OR. PRESENT( get_A ) )                        &
               .AND. .NOT. map%l_perm ) ) THEN
        inform%status = GALAHAD_error_ah_unordered
        RETURN
      END IF
      IF ( ( PRESENT( get_c_bounds ) ) .AND. map%n_reordered < map%n           &
           .AND. .NOT. ( map%l_perm .AND. map%a_perm ) ) THEN
        inform%status = GALAHAD_error_ah_unordered
        RETURN
      END IF

!  if there are fixed variables, compute suitable dual variables

      IF ( ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.        &
             PRESENT( get_z ) ) .AND. map%n_reordered < map%n ) THEN

        IF ( apz ) THEN
          prob%Z( map%n_reordered + 1 : map%n ) =                              &
            prob%G( map%n_reordered + 1 : map%n )
        ELSE
          prob%Z_l( map%n_reordered + 1 : map%n ) =                            &
            prob%G( map%n_reordered + 1 : map%n )
        END IF

!  copy to Z_l and Z_u if required

        IF ( apz ) THEN
          IF ( apzl .AND. apzu ) THEN
            DO i = map%n_reordered + 1, map%n
              IF ( prob%Z( i ) > 0 ) THEN
                prob%Z_l( i ) = prob%Z( i )
                prob%Z_u( i ) = zero
              ELSE
                prob%Z_u( i ) = prob%Z( i )
                prob%Z_l( i ) = zero
              END IF
            END DO
          END IF
        ELSE
          IF ( apzl .AND. apzu ) THEN
            DO i = map%n_reordered + 1, map%n
              IF ( prob%Z_l( i ) > 0 ) THEN
                prob%Z_u( i ) = zero
              ELSE
                prob%Z_u( i ) = prob%Z_l( i )
                prob%Z_l( i ) = zero
              END IF
            END DO
          END IF
        END IF
      END IF

!  form c = A * x

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_c ) ) THEN
        prob%C( : map%m ) = zero
        CALL LSP_LX( map, prob%X, prob%L%val, prob%L%col, prob%L%ptr,          &
                      map%m, prob%C( : map%m ) )
      END IF

!  transform the bounds on the constraints to account for fixed variables

      IF ( ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) )           &
             .AND. map%n_reordered < map%n ) THEN
          CALL LSP_add_fixed( map, prob, c = .TRUE., c_bounds = .TRUE. )
      ELSE IF ( ( PRESENT( get_c ) .OR. PRESENT( get_c_bounds ) )              &
                 .AND. map%n_reordered < map%n ) THEN
        CALL LSP_add_fixed( map, prob, c =  PRESENT( get_c ),                  &
                            c_bounds = PRESENT( get_c_bounds ) )
      END IF

!  see if we need to invert the mappings

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_A ) .OR. PRESENT( get_L ) )                            &
        CALL LSP_invert_mapping( map%n, map%x_map, map%IW( : map%n ) )
      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_A ) )                                                  &
        CALL LSP_invert_mapping( map%o, map%c_map, map%IW( : map%o ) )
      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_L ) )                                                  &
        CALL LSP_invert_mapping( map%m, map%c_map, map%IW( : map%m ) )





!  restore A

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_A ) ) THEN

!  original dense storage

        IF ( map%a_type == dense .OR. map%a_type == dense_by_rows .OR.         &
             map%a_type == dense_by_columns ) THEN
          CALL SORT_inplace_permute(                                           &
              map%a_ne, map%a_map_inverse, X = prob%A%val )

!  original row-wise storage

        ELSE IF ( map%a_type == sparse_by_rows ) THEN

!  compute the number of entries in each row of A, and renumber its columns

          map%IW( : map%o ) = 0
          DO l = 1, map%n
            j = map%x_map( l )
            DO k = map%ptr_l_fixed( l ), map%ptr_l_fixed( l + 1 ) - 1
              i = map%a_map( prob%A%row( k ) )
              map%IW( i ) = map%IW( i ) + 1
              prob%A%col( l ) = j
            END DO
          END DO

!  set the starting addresses for each row in the permuted matrix

          prob%A%ptr( 1 ) = 1
          DO i = 1, map%o
            prob%A%ptr( i + 1 ) = prob%A%ptr( i ) + map%IW( i )
          END DO

!  undo the reordering of A

          CALL SORT_inplace_permute( map%a_ne, map%a_map_inverse,              &
                                     X = prob%A%val, IX = prob%A%col )

!  original column-wise storage

        ELSE IF ( map%a_type == sparse_by_columns ) THEN

!  compute the number of entries in each column of A, and renumber its rows.
!  NB. Any columns corresponding to FIXED variables, will have been
!  moved to the end of A, and will be stored by COLUMN not by row. In
!  particular, l_col for these entries gives the row and not the column
!  number

          map%IW( : map%n ) = 0
          DO l = 1, map%n
            j = map%x_map( l )
            DO k = map%ptr_l_fixed( l ), map%ptr_l_fixed( l + 1 ) - 1
              i = map%a_map( prob%A%row( k ) )
              map%IW( j ) = map%IW( j ) + 1
              prob%A%row( l ) = i
            END DO
          END DO

!  set the starting addresses for each row in the permuted matrix

          prob%A%ptr( 1 ) = 1
          DO i = 1, map%n
            prob%A%ptr( i + 1 ) = prob%A%ptr( i ) + map%IW( i )
          END DO

!  undo the reordering of A

          CALL SORT_inplace_permute( map%a_ne, map%a_map_inverse,              &
                                     X = prob%A%val, IX = prob%A%row )

!  original co-ordinate storage

        ELSE

!  renumber the rows and columns

          DO l = 1, map%n
            j = map%x_map( l )
            DO k = map%ptr_l_fixed( l ), map%ptr_l_fixed( l + 1 ) - 1
              prob%A%row( k ) = map%a_map( prob%A%row( k ) )
              prob%A%col( k ) = j
            END DO
          END DO

!  undo the reordering of A

          CALL SORT_inplace_permute( map%a_ne, map%a_map_inverse,              &
              X = prob%A%val, IX = prob%A%row, IY = prob%A%col )
        END IF

!  record that A had been restored

        map%a_perm = .FALSE.
      END IF

!  restore L

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_L ) ) THEN

!  original dense storage

        IF ( map%l_type == dense .OR. map%l_type == dense_by_rows .OR.         &
             map%l_type == dense_by_columns ) THEN
          CALL SORT_inplace_permute(                                           &
              map%l_ne, map%l_map_inverse, X = prob%L%val )

!  original row-wise storage

        ELSE IF ( map%l_type == sparse_by_rows ) THEN

!  compute the number of entries in each row of L, and renumber its columns.
!  NB. Any columns corresponding to FIXED variables, will have been
!  moved to the end of A, and will be stored by COLUMN not by row. In
!  particular, l_col for these entries gives the row and not the column
!  number

          map%IW( : map%m ) = 0
          DO k = 1, map%m
            i = map%c_map( k )
            DO l = prob%L%ptr( k ), prob%L%ptr( k + 1 ) - 1
              map%IW( i ) = map%IW( i ) + 1
              prob%L%col( l ) = map%x_map( prob%L%col( l ) )
            END DO
          END DO

          DO l = map%n_reordered + 1, map%n
            j = map%x_map( l )
            DO k = map%ptr_l_fixed( l ), map%ptr_l_fixed( l + 1 ) - 1
              i =  map%c_map( prob%L%col( k ) )
              map%IW( i ) = map%IW( i ) + 1
              prob%L%col( k ) = j
            END DO
          END DO

!  set the starting addresses for each row in the permuted matrix

          prob%L%ptr( 1 ) = 1
          DO i = 1, map%m
            prob%L%ptr( i + 1 ) = prob%L%ptr( i ) + map%IW( i )
          END DO

!  undo the reordering of L

          CALL SORT_inplace_permute( map%l_ne, map%l_map_inverse,              &
                                     X = prob%L%val, IX = prob%L%col )

!  original column-wise storage

        ELSE IF ( map%l_type == sparse_by_columns ) THEN

!  compute the number of entries in each column of L, and renumber its rows.
!  NB. Any columns corresponding to FIXED variables, will have been
!  moved to the end of L, and will be stored by COLUMN not by row. In
!  particular, l_col for these entries gives the row and not the column
!  number

          map%IW( : map%n ) = 0
          DO k = 1, map%m
            i = map%c_map( k )
            DO l = prob%L%ptr( k ), prob%L%ptr( k + 1 ) - 1
              j = map%x_map( prob%L%col( l ) )
              map%IW( j ) = map%IW( j ) + 1
              prob%L%row( l ) = i
            END DO
          END DO

          DO l = map%n_reordered + 1, map%n
            j = map%x_map( l )
            DO k = map%ptr_l_fixed( l ), map%ptr_l_fixed( l + 1 ) - 1
              i = map%c_map( prob%L%col( k ) )
              map%IW( j ) = map%IW( j ) + 1
              prob%L%row( k ) = i
            END DO
          END DO

!  set the starting addresses for each row in the permuted matrix

          prob%L%ptr( 1 ) = 1
          DO i = 1, map%n
            prob%L%ptr( i + 1 ) = prob%L%ptr( i ) + map%IW( i )
          END DO

!  undo the reordering of L

          CALL SORT_inplace_permute( map%l_ne, map%l_map_inverse,              &
                                     X = prob%L%val, IX = prob%L%row )

!  original co-ordinate storage

        ELSE

!  renumber the rows and columns

          DO k = 1, map%m
            i = map%c_map( k )
            DO l = prob%L%ptr( k ), prob%L%ptr( k + 1 ) - 1
              prob%L%row( l ) = i
              prob%L%col( l ) = map%x_map( prob%L%col( l ) )
            END DO
          END DO

          DO l = map%n_reordered + 1, map%n
            j = map%x_map( l )
            DO k = map%ptr_l_fixed( l ), map%ptr_l_fixed( l + 1 ) - 1
              prob%L%row( k ) = map%c_map( prob%L%col( k ) )
              prob%L%col( k ) = j
            END DO
          END DO

!  undo the reordering of L

          CALL SORT_inplace_permute( map%l_ne, map%l_map_inverse,              &
              X = prob%L%val, IX = prob%L%row, IY = prob%L%col )
        END IF

!  record that A had been restored

        map%l_perm = .FALSE.
      END IF

!  see if we need to reinvert the mappings

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_A ) .OR. PRESENT( get_L ) )                            &
        CALL LSP_invert_mapping( map%n, map%x_map, map%IW( : map%n ) )
      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_A ) )                                                  &
        CALL LSP_invert_mapping( map%m, map%a_map, map%IW( : map%o ) )
      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_L ) )                                                  &
        CALL LSP_invert_mapping( map%m, map%c_map, map%IW( : map%m ) )

!  restore b

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_b ) ) THEN
        CALL SORT_inverse_permute( map%o, map%a_map, X = prob%B )
      END IF

!  restore db

      IF ( PRESENT( get_all_parametric ) .OR. PRESENT( get_db ) ) THEN
        IF ( ALLOCATED( prob%DB ) ) THEN
          IF ( SIZE( prob%DB ) == map%o )                                      &
            CALL SORT_inverse_permute( map%o, map%a_map, X = prob%DB )
        END IF
      END IF

!  restore c

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_c ) ) THEN
        CALL SORT_inverse_permute( map%m, map%c_map, X = prob%C )
      END IF

!  restore the bounds on c

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_c_bounds ) ) THEN
        CALL SORT_inverse_permute( map%m, map%c_map, X = prob%C_l )
        CALL SORT_inverse_permute( map%m, map%c_map, X = prob%C_u )
      END IF

!  restore the bounds on dc

      IF ( PRESENT( get_all_parametric ) .OR. PRESENT( get_dc_bounds ) ) THEN
        IF ( ALLOCATED( prob%DC_l ) ) THEN
          IF ( SIZE( prob%DC_l ) == map%m )                                    &
            CALL SORT_inverse_permute( map%m, map%c_map, X = prob%DC_l )
        END IF
        IF ( ALLOCATED( prob%DC_u ) ) THEN
          IF ( SIZE( prob%DC_u ) == map%m )                                    &
            CALL SORT_inverse_permute( map%m, map%c_map, X = prob%DC_u )
        END IF
      END IF

!  restore x

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_x ) ) THEN
        CALL SORT_inverse_permute( map%n, map%x_map, X = prob%X )
      END IF

!  restore the bounds on x

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_x_bounds ) ) THEN
        CALL SORT_inverse_permute( map%n, map%x_map, X = prob%X_l )
        CALL SORT_inverse_permute( map%n, map%x_map, X = prob%X_u )
      END IF

!  restore the bounds on dx

      IF ( PRESENT( get_all_parametric ) .OR. PRESENT( get_dx_bounds ) ) THEN
        IF ( ALLOCATED( prob%DX_l ) ) THEN
          IF ( SIZE( prob%DX_l ) == map%n )                                    &
            CALL SORT_inverse_permute( map%n, map%x_map, X = prob%DX_l )
        END IF
        IF ( ALLOCATED( prob%DX_u ) ) THEN
          IF ( SIZE( prob%DX_u ) == map%n )                                    &
            CALL SORT_inverse_permute( map%n, map%x_map, X = prob%DX_u )
        END IF
      END IF

!  restore the dual variables z

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_z ) ) THEN
        IF ( apz ) CALL SORT_inverse_permute( map%n, map%x_map, X = prob%Z )
        IF ( apzl ) CALL SORT_inverse_permute( map%n, map%x_map, X = prob%Z_l )
        IF ( apzu ) CALL SORT_inverse_permute( map%n, map%x_map, X = prob%Z_u )
      END IF

!  restore the Lagrange multipliers y

      IF ( PRESENT( get_all ) .OR. PRESENT( get_all_parametric ) .OR.          &
           PRESENT( get_y ) ) THEN
        IF ( apy ) CALL SORT_inverse_permute( map%m, map%c_map, X = prob%Y )
        IF ( apyl ) CALL SORT_inverse_permute( map%m, map%c_map, X = prob%Y_l )
        IF ( apyu ) CALL SORT_inverse_permute( map%m, map%c_map, X = prob%Y_u )
      END IF

!  pick up the correct dimensions

      prob%n = map%n ; prob%o = map%o ; prob%m = map%m
      CALL QPT_put_A( prob%A%type, SMT_get( map%a_type_original ) )
      CALL QPT_put_A( prob%L%type, SMT_get( map%l_type_original ) )
      IF ( SMT_get( prob%A%type ) == 'COORDINATE' )                            &
         prob%A%ne = map%a_ne_original
      IF ( SMT_get( prob%L%type ) == 'COORDINATE' )                            &
         prob%L%ne = map%l_ne_original

      inform%status = GALAHAD_ok
      RETURN

!  End of LSP_restore

      END SUBROUTINE LSP_restore

!-*-*-*-*-*-*-*-   L S P _ f i n a l i z e   S U B R O U T I N E  -*-*-*-*-*-*-

      SUBROUTINE LSP_terminate( map, control, inform )

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   map     see Subroutine LSP_initialize
!   control see Subroutine LSP_initialize
!   inform  see Subroutine LSP_reorder

!  Dummy arguments

      TYPE ( LSP_map_type ), INTENT( INOUT ) :: map
      TYPE ( LSP_control_type ), INTENT( IN ) :: control
      TYPE ( LSP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

      inform%status = GALAHAD_ok
      map%set = .FALSE.

!  Deallocate all mapping arrays

      array_name = 'lsp: map%IW'
      CALL SPACE_dealloc_array( map%IW,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsp: map%x_map'
      CALL SPACE_dealloc_array( map%x_map,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsp: map%c_map'
      CALL SPACE_dealloc_array( map%c_map,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsp: map%h_map_inverse'
      CALL SPACE_dealloc_array( map%h_map_inverse,                             &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsp: map%l_map_inverse'
      CALL SPACE_dealloc_array( map%l_map_inverse,                             &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsp: map%ptr_l_fixed'
      CALL SPACE_dealloc_array( map%ptr_l_fixed,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsp: map%l_type_original'
      CALL SPACE_dealloc_array( map%l_type_original,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsp: map%h_type_original'
      CALL SPACE_dealloc_array( map%h_type_original,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsp: map%W'
      CALL SPACE_dealloc_array( map%W,                                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine LSP_terminate

      END SUBROUTINE LSP_terminate

!-*-*-*-*-*-*-   L S P _ o r d e r _ r o w s   S U B R O U T I N E  -*-*-*-*-*-

      SUBROUTINE LSP_order_rows( n_rows, VAL, COL, PTR, MAP_inverse )

!  Reorder the entries in each row so that the column entries appear
!  in order of increasing index; also update (the inverse of) a map from
!  a previous reordering

!  Arguments:
!  =========
!
!   n_rows      number of rows
!   PTR         starting addresses of each row, as well as 1 beyond the last row
!   VAL         values of entries in each row
!   COL         column indices corresponding to values
!   map         a permutation of the numbers 1, .., Ptr( n_rows + 1 ) - 1
!   map_inverse the inverse permutation of the above

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n_rows
      INTEGER, INTENT( IN ), DIMENSION( n_rows + 1 ) :: PTR
      INTEGER, INTENT( INOUT ),                                                &
               DIMENSION( Ptr( n_rows + 1 ) - 1 ) :: COL, MAP_inverse
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
               DIMENSION( Ptr( n_rows + 1 ) - 1 ) :: VAL

!  Local variables

      INTEGER :: i, current, col_current, inverse, inverse_current
      INTEGER :: previous, next, row_start, row_end, inrow, inform_quicksort
      REAL ( KIND = wp ) :: val_current
      INTEGER, PARAMETER :: do_quicksort = 10

!  loop over the rows

      DO i = 1, n_rows
        row_start = PTR( i )
        row_end = PTR( i + 1 ) - 1

!  for the current row

        inrow = row_end - row_start + 1
        IF ( inrow <= 0 ) CYCLE
        IF ( inrow > do_quicksort ) THEN

!  if the row contains enough entries, use quicksort ...

          DO current = row_start + 1, row_end

!  skip if the value in the current position is already in order

            IF ( COL( current ) < COL( current - 1 ) ) THEN
              CALL SORT_quicksort( inrow, COL( row_start : row_end ),          &
                                   inform_quicksort,                           &
                                   ix = MAP_inverse( row_start : row_end ),    &
                                   rx = VAL( row_start : row_end ) )

              EXIT
            END IF
          END DO
        ELSE

!  ... else use a simple entry shuffle

          DO current = row_start + 1, row_end
            col_current = COL( current )

!  skip if the value in the current position is already in order

            IF ( col_current < COL( current - 1 ) ) THEN
              val_current = VAL( current )
              inverse_current = MAP_inverse( current )

!  the value in the current position is out of order, but those in
!  positions row_start, current - 1 are now in order

              DO previous = row_start, current - 1
                IF ( col_current < COL( previous ) ) THEN

!  the current value should be inserted at position previous

                  DO next = current, previous + 1, - 1

!  shift values previous, ... , current + 1 one place to the right

                    COL( next ) = COL( next - 1 )
                    VAL( next ) = VAL( next - 1 )

!  update the inverse map

                    inverse = MAP_inverse( next - 1 )
                    MAP_inverse( next ) = inverse

                  END DO

!  insert the current value in its correct position

                  VAL( previous ) = val_current
                  COL( previous ) = col_current

!  update the inverse map

                  MAP_inverse( previous ) = inverse_current
                  EXIT

                END IF
              END DO
            END IF
          END DO
        END IF
!       write(45, "( ' after  ', /, ( 10I6 ) )" ) COL( row_start : row_end )
!       write(47, "( '        ', /, ( 10I6 ) )" ) COL( row_start : row_end )
      END DO

      RETURN

!  End of subroutine LSP_order_rows

      END SUBROUTINE LSP_order_rows

!!-*-*-*-*-   L S P _ i n p l a c e _ p e r m u t e  S U B R O U T I N E  -*-*-*-
!
!!      SUBROUTINE LSP_inplace_permute( n, MAP, X, IX, IY )
!
!!  Permute the entries of X so that x_i appears in position map_i
!!  Do this without resorting to extra vector storage. Optionally,
!!  permute the entries of IX and IY so that ixc_i and iy_i
!!  appear in positions map_i
!
!!  Arguments:
!!  =========
!!
!!   n           number of components of x
!!   X           the array x
!!   MAP         the permutation map
!!   IX          the array ix
!!   IY          the array iy
!
!!  Dummy arguments
!
!      INTEGER, INTENT( IN ) :: n
!      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: MAP
!      INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( n ) :: IX, IY
!      REAL ( KIND = wp ), INTENT( INOUT ), OPTIONAL, DIMENSION( n ) :: X
!
!!  Local variables
!
!      INTEGER :: i, mi, mi_old, iymi, iymi_old, ixmi, ixmi_old
!      REAL ( KIND = wp ) :: xmi, xmi_old
!
!!  for X, IX and IY:
!
!      IF ( PRESENT( IX ) .AND. PRESENT( IY ) ) THEN
!
!!  loop over the entries of X, IX and IY
!
!        DO i = 1, n
!          mi = MAP( i )
!
!!  skip any entry which is already in place
!
!          IF ( mi == i ) THEN
!            CYCLE
!
!!  skip any entry which has already been moved into place, remembering
!!  to un-negate the relevant entry in MAP
!
!          ELSE IF ( mi < 0 ) THEN
!            MAP( i ) = - mi
!
!!  the i-th entry is not in place. Chase through the list of entries
!!  i, MAP( i ), MAP( MAP( i ) ), ... until MAP( ... ( MAP( i ) ) ... ) = i
!!  moving entries into place. Negate the relevant entries in MAP so that
!!  these entries will not be moved again
!
!          ELSE
!            xmi_old = X( i )
!            iymi_old = IY( i )
!            ixmi_old = IX( i )
!            DO
!              xmi = X( mi )
!              iymi = IY( mi )
!              ixmi = IX( mi )
!              X( mi ) = xmi_old
!              IY( mi ) = iymi_old
!              IX( mi ) = ixmi_old
!              xmi_old = xmi
!              iymi_old = iymi
!              ixmi_old = ixmi
!              mi_old = mi
!              mi = MAP( mi_old )
!              MAP( mi_old ) = - mi
!              IF ( mi == i ) EXIT
!            END DO
!            X( i ) = xmi_old
!            IY( i ) = iymi_old
!            IX( i ) = ixmi_old
!          END IF
!        END DO
!
!!  for X and IX:
!
!      ELSE IF ( PRESENT( IX ) ) THEN
!
!!  loop over the entries of X and IX
!
!        DO i = 1, n
!          mi = MAP( i )
!
!!  skip any entry which is already in place
!
!          IF ( mi == i ) THEN
!            CYCLE
!
!!  skip any entry which has already been moved into place, remembering
!!  to un-negate the relevant entry in MAP
!
!          ELSE IF ( mi < 0 ) THEN
!            MAP( i ) = - mi
!
!!  the i-th entry is not in place. Chase through the list of entries
!!  i, MAP( i ), MAP( MAP( i ) ), ... until MAP( ... ( MAP( i ) ) ... ) = i
!!  moving entries into place. Negate the relevant entries in MAP so that
!!  these entries will not be moved again
!
!          ELSE
!            xmi_old = X( i )
!            ixmi_old = IX( i )
!            DO
!              xmi = X( mi )
!              ixmi = IX( mi )
!              X( mi ) = xmi_old
!              IX( mi ) = ixmi_old
!              xmi_old = xmi
!              ixmi_old = ixmi
!              mi_old = mi
!              mi = MAP( mi_old )
!              MAP( mi_old ) = - mi
!              IF ( mi == i ) EXIT
!            END DO
!            X( i ) = xmi_old
!            IX( i ) = ixmi_old
!          END IF
!        END DO
!
!!  for just X:
!
!      ELSE
!
!!  loop over the entries of X
!
!        DO i = 1, n
!          mi = MAP( i )
!
!!  skip any entry which is already in place
!
!          IF ( mi == i ) THEN
!            CYCLE
!
!!  skip any entry which has already been moved into place, remembering
!!  to un-negate the relevant entry in MAP
!
!          ELSE IF ( mi < 0 ) THEN
!            MAP( i ) = - mi
!
!!  the i-th entry is not in place. Chase through the list of entries
!!  i, MAP( i ), MAP( MAP( i ) ), ... until MAP( ... ( MAP( i ) ) ... ) = i
!!  moving entries into place. Negate the relevant entries in MAP so that
!!  these entries will not be moved again
!
!          ELSE
!            xmi_old = X( i )
!            DO
!              xmi = X( mi )
!              X( mi ) = xmi_old
!              xmi_old = xmi
!              mi_old = mi
!              mi = MAP( mi_old )
!              MAP( mi_old ) = - mi
!              IF ( mi == i ) EXIT
!            END DO
!            X( i ) = xmi_old
!          END IF
!        END DO
!
!      END IF
!
!      RETURN
!
!!  End of subroutine SORT_inplace_permute
!
!      END SUBROUTINE LSP_inplace_permute

!!-   L S P _ i n v e r s e _ p e r m u t e  S U B R O U T I N E  -
!
!      SUBROUTINE LSP_inverse_permute( n, MAP_inverse, X, IX )
!
!!  Permute the entries of X so that x(map_i) appears in position i
!!  Do this without resorting to extra vector storage. Optionally,
!!  permute the entries of IX so that ix(map_i) appears in position i
!
!!  Arguments:
!!  =========
!
!!   n           number of components of x
!!   X           the array x
!!   MAP_inverse the permutation map
!!   IX          the array IX
!
!!  Dummy arguments
!
!      INTEGER, INTENT( IN ) :: n
!      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: MAP_inverse
!      INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( n ) :: IX
!      REAL ( KIND = wp ), INTENT( INOUT ), OPTIONAL, DIMENSION( n ) :: X
!
!!  Local variables
!
!      INTEGER :: i, mi, mi_old, ixi
!      REAL ( KIND = wp ) :: xi
!
!!  For both X and IX:
!
!      IF ( PRESENT( IX ) ) THEN
!
!!  loop over the entries of X and IX
!
!        DO i = 1, n
!          mi = MAP_inverse( i )
!
!!  skip any entry which is already in place
!
!          IF ( mi == i ) THEN
!            CYCLE
!
!!  skip any entry which has already been moved into place, remembering
!!  to un-negate the relevant entry in MAP_inverse
!
!          ELSE IF ( mi < 0 ) THEN
!            MAP_inverse( i ) = - mi
!
!!  the i-th entry is not in place. Chase through the list of entries
!!  i, MAP_inverse( i ), MAP_inverse( MAP_inverse( i ) ), ... until
!!  MAP_inverse( ... ( MAP_inverse( i ) ) ... ) = i, moving entries into place.
!!  Negate the relevant entries in MAP_inverse so that these entries will
!!  not be moved again
!
!          ELSE
!            xi = X( i )
!            ixi = IX( i )
!            mi_old = i
!            DO
!              X( mi_old ) = X( mi )
!              IX( mi_old ) = IX( mi )
!              mi_old = mi
!              mi = MAP_inverse( mi_old )
!              MAP_inverse( mi_old ) = - mi
!              IF ( mi == i ) EXIT
!            END DO
!            X( mi_old ) = xi
!            IX( mi_old ) = ixi
!          END IF
!        END DO
!
!!  for just X:
!
!      ELSE
!
!!  loop over the entries of X
!
!        DO i = 1, n
!          mi = MAP_inverse( i )
!
!!  skip any entry which is already in place
!
!          IF ( mi == i ) THEN
!            CYCLE
!
!!  skip any entry which has already been moved into place, remembering
!!  to un-negate the relevant entry in MAP_inverse
!
!          ELSE IF ( mi < 0 ) THEN
!            MAP_inverse( i ) = - mi
!
!!  the i-th entry is not in place. Chase through the list of entries
!!  i, MAP_inverse( i ), MAP_inverse( MAP_inverse( i ) ), ... until
!!  MAP_inverse( ... ( MAP_inverse( i ) ) ... ) = i, moving entries into place.
!!  Negate the relevant entries in MAP_inverse so that these entries will
!!  not be moved again
!
!          ELSE
!            xi = X( i )
!            mi_old = i
!            DO
!              X( mi_old ) = X( mi )
!              mi_old = mi
!              mi = MAP_inverse( mi_old )
!              MAP_inverse( mi_old ) = - mi
!              IF ( mi == i ) EXIT
!            END DO
!            X( mi_old ) = xi
!          END IF
!        END DO
!
!      END IF
!
!      RETURN
!
!! End of subroutine LSP_inverse_permute
!
!      END SUBROUTINE LSP_inverse_permute

!-*-*-*-*-   L S P _ i n v e r t _ m a p p i n g  S U B R O U T I N E   -*-*-*-

      SUBROUTINE LSP_invert_mapping( n, MAP, MAP_inverse )

!  Place the inverse mapping to MAP in MAP_inverse, and then use this
!  to overwrite MAP

!  Arguments:
!  =========
!
!   n           number of components of x
!   MAP         the mapping (and, on exit, its inverse)
!   MAP_inverse its inverse

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: MAP, MAP_inverse

!  Local variables

      INTEGER :: i

!  invert the mapping, MAP

      DO i = 1, n
        MAP_inverse( MAP( i ) ) = i
      END DO

!  copy this back to MAP

      MAP = MAP_inverse

      RETURN

!  End of subroutine LSP_invert_mapping

      END SUBROUTINE LSP_invert_mapping

!-*-*-*-*-   L S P _ r e m o v e _ f i x e d   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LSP_remove_fixed( map, prob, c_bounds )

!      ................................................
!      .                                              .
!      .  Transform the bounds on the constraints     .
!      .  to account for fixed variables              .
!      .                                              .
!      ................................................

!  Arguments:
!  =========
!
!   map      see Subroutine LSP_initialize
!   prob     see Subroutine LSP_reorder
!   c_bounds LOGICAL. If true, adjust c_l and c_u to account for fixed
!            variables via
!              c_l <- c_l - l_x x_x and c_u <- c_u - l_x x_x

!  Dummy arguments

      TYPE ( LSP_map_type ), INTENT( INOUT ) :: map
      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      LOGICAL, OPTIONAL, INTENT( IN ) :: c_bounds

!  Local variables

      INTEGER :: i, j, l
      REAL ( KIND = wp ) :: x, c
      LOGICAL :: yes_f, yes_g, yes_c_bounds

      IF ( prob%n >= map%n ) RETURN

      IF ( PRESENT ( c_bounds ) ) THEN
        yes_c_bounds = c_bounds
      ELSE
        yes_c_bounds = .FALSE.
      END IF

!  process the bounds on c

      IF ( yes_c_bounds ) THEN
        DO j = prob%n + 1, map%n
          x = prob%X( j )
          IF ( x /= zero ) THEN
            DO l = map%ptr_l_fixed( j ), map%ptr_l_fixed( j + 1 ) - 1
              i = prob%L%col( l )   !  NB: this is the row number
              c = prob%L%val( l ) * x
              prob%C_l( i ) = prob%C_l( i ) - c
              prob%C_u( i ) = prob%C_u( i ) - c
            END DO
          END IF
        END DO
      END IF

      RETURN

!  End of LSP_remove_fixed

      END SUBROUTINE LSP_remove_fixed

!-*-*-*-*-*-*-   L S P _ a d d _ f i x e d   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE LSP_add_fixed( map, prob, c, c_bounds )

!      .............................................
!      .                                           .
!      .  Transform the bounds on the constraints  .
!      .  when reintroducing fixed variables       .
!      .                                           .
!      .............................................

!  Arguments:
!  =========
!
!   map      see Subroutine LSP_initialize
!   prob%    see Subroutine LSP_reorder
!   c        LOGICAL. If true, adjust c to account for fixed variables via
!              c <- c + l_x x_x
!   c_bounds LOGICAL. If true, adjust c_l and c_u to account for fixed
!            variables via
!              c_l <- c_l + l_x x_x and c_u <- c_u + l_x x_x

!  Dummy arguments

      TYPE ( LSP_map_type ), INTENT( INOUT ) :: map
      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      LOGICAL, OPTIONAL, INTENT( IN ) :: c, c_bounds

!  Local variables

      INTEGER :: i, j, l
      REAL ( KIND = wp ) :: x, c_val
      LOGICAL :: yes_f, yes_g, yes_c, yes_c_bounds

      IF ( map%n_reordered >= map%n ) RETURN

      IF ( PRESENT ( c ) ) THEN
        yes_c = c
      ELSE
        yes_c = .FALSE.
      END IF

      IF ( PRESENT ( c_bounds ) ) THEN
        yes_c_bounds = c_bounds
      ELSE
        yes_c_bounds = .FALSE.
      END IF

      IF ( yes_c .AND. yes_c_bounds ) THEN

!  process c and its bounds

        DO j = map%n_reordered + 1, map%n
          x = prob%X( j )
          IF ( x /= zero ) THEN
            DO l = map%ptr_l_fixed( j ), map%ptr_l_fixed( j + 1 ) - 1
              i = prob%L%col( l )   !  NB: this is the row number
              c_val = prob%L%val( l ) * x
              prob%C( i ) = prob%C( i ) + c_val
              prob%C_l( i ) = prob%C_l( i ) + c_val
              prob%C_u( i ) = prob%C_u( i ) + c_val
            END DO
          END IF
        END DO

!  process the bounds on c

      ELSE IF ( yes_c_bounds ) THEN
        DO j = map%n_reordered + 1, map%n
          x = prob%X( j )
          IF ( x /= zero ) THEN
            DO l = map%ptr_l_fixed( j ), map%ptr_l_fixed( j + 1 ) - 1
              i = prob%L%col( l )   !  NB: this is the row number
              c_val = prob%L%val( l ) * x
              prob%C_l( i ) = prob%C_l( i ) + c_val
              prob%C_u( i ) = prob%C_u( i ) + c_val
            END DO
          END IF
        END DO

!  process c

      ELSE IF ( yes_c ) THEN
        DO j = map%n_reordered + 1, map%n
          x = prob%X( j )
          IF ( x /= zero ) THEN
            DO l = map%ptr_l_fixed( j ), map%ptr_l_fixed( j + 1 ) - 1
              i = prob%L%col( l )   !  NB: this is the row number
              prob%C( i ) = prob%C( i ) + prob%L%val( l ) * x
            END DO
          END IF
        END DO
      END IF

      RETURN

!  End of LSP_add_fixed

      END SUBROUTINE LSP_add_fixed

!-*-*-*-*-*-*-*-*-*-*-*-   L S P _ L x  S U B R O U T I N E  -*-*-*-*-*-*-*-*-*-

      SUBROUTINE LSP_LX( map, prob_x, prob_l_val, prob_l_col, prob_l_ptr, m, Lx)

!      ..............................................
!      .                                            .
!      .  Perform the operation Lx := Lx + L * x    .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   map     see Subroutine LSP_initialize
!   prob_   see Subroutine LSP_reorder
!   m       row dimension of L
!   Lx      the result of adding L * x to Lx
!

!  Dummy arguments

      TYPE ( LSP_map_type ), INTENT( IN ) :: map
      INTEGER, INTENT( IN ), DIMENSION( map%m + 1 ) :: prob_l_ptr
      INTEGER, INTENT( IN ), DIMENSION( map%l_ne ) ::  prob_l_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( map%n ) :: prob_x
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( map%l_ne ) :: prob_l_val
      INTEGER, INTENT( IN ) :: m
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Lx

!  Local variables

      INTEGER :: i, l

      DO i = 1, m
        DO l = prob_l_ptr( i ), prob_l_ptr( i + 1 ) - 1
         Lx( i ) = Lx( i ) + prob_l_val( l ) * prob_x( prob_l_col( l ) )
        END DO
      END DO

      RETURN

!  End of subroutine LSP_Lx

      END SUBROUTINE LSP_Lx

!  End of module LSP

   END MODULE GALAHAD_LSP_double
