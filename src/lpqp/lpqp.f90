! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ L P Q P   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started July 30th 2002
!   originally released GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_LPQP_double

!     ----------------------------------------------------
!     |                                                  |
!     | Convert an ordinary quadratic program into       |
!     | an l_p quadratic program                         |
!     |                                                  |
!     |    minimize     1/2 x(T) H x + g(T) x + f        |
!     |     + rho || max( 0, c_l - A x, A x - c_u ) ||_p |
!     |    subject to       x_l <= x <= x_u              |
!     |                                                  |
!     ----------------------------------------------------

!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPACE_double
     USE GALAHAD_QPT_double
     USE GALAHAD_SORT_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_SMT_double, ONLY: SMT_put, SMT_get

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: LPQP_initialize, LPQP_read_specfile, LPQP_formulate,            &
               LPQP_restore, LPQP_terminate, QPT_problem_type

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

     INTEGER, PARAMETER :: a_output_sparse_by_rows = 0
     INTEGER, PARAMETER :: a_output_coordinate = 1
     INTEGER, PARAMETER :: a_output_dense = 2
     INTEGER, PARAMETER :: h_output_sparse_by_rows = 0
     INTEGER, PARAMETER :: h_output_coordinate = 1
     INTEGER, PARAMETER :: h_output_dense = 2
     INTEGER, PARAMETER :: h_output_diagonal = 3
     INTEGER, PARAMETER :: h_output_none = 4
     INTEGER, PARAMETER :: h_output_lbfgs = 5
     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenm2 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: tenm4 = ten ** ( - 4 )
     REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

     TYPE, PUBLIC :: LPQP_control_type

!  error and warning diagnostics occur on stream error

       INTEGER :: error = 6

!  general output occurs on stream out

       INTEGER :: out = 6

!  the level of output required. <= 0 gives no output, = 1 gives a one-line
!   summary for every iteration, = 2 gives a summary of the inner iteration
!   for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER :: print_level = 0

!  any bound larger than infinity in modulus will be regarded as infinite

       REAL ( KIND = wp ) :: infinity = ten ** 19

!   if space_critical is true, every effort will be made to use as little
!    space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!   if deallocate_error_fatal is true, any array/pointer deallocation error
!    will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!  output format for problem Hessian, H; possible values are
!    'COORDINATE'
!    'DENSE'   (not yet implemented)
!    'SPARSE_BY_ROWS'
!    'DIAGONAL'
!    'NONE'
!  any other value defaults to 'SPARSE_BY_ROWS', except if input format is
!  'LBFGS' which defaults to 'LBFGS'. 'DIAGONAL' will result in an error
!  if the matrix is not diagonal, while 'NONE' will do the same if the matrix
!  has any nonzeros

       CHARACTER ( LEN = 30 ) :: h_output_format = 'SPARSE_BY_ROWS'

!  output format for problem constraint Jacobian, A; possible values are
!    'COORDINATE'
!    'DENSE'   (not yet implemented)
!    'SPARSE_BY_ROWS'
!  any other value defaults to 'SPARSE_BY_ROWS'

       CHARACTER ( LEN = 30 ) :: a_output_format = 'SPARSE_BY_ROWS'

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '
     END TYPE

     TYPE, PUBLIC :: LPQP_inform_type

!  return status. See FISQP_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total CPU time spent in the package

       REAL ( KIND = wp ) :: time = 0.0

!  the total clock time spent in the package

       REAL ( KIND = wp ) :: clock_time = 0.0

     END TYPE

     TYPE, PUBLIC :: LPQP_data_type
       INTEGER :: m, n, m_b, a_ne, h_ne, h_output_format, a_output_format
       INTEGER :: Hessian_kind, gradient_kind
       LOGICAL :: one_norm
       CHARACTER, ALLOCATABLE, DIMENSION( : ) :: a_type, h_type
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
       INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: BOTH
     END TYPE

   CONTAINS

!-*-*-*-*-*-   L P Q P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

!    SUBROUTINE LPQP_initialize( data, control, VNAME_lpqp, CNAME_lpqp )
!    SUBROUTINE LPQP_initialize( control, VNAME_lpqp, CNAME_lpqp )
     SUBROUTINE LPQP_initialize( control )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for LPQP. This routine should be called before
!  LPQP_solve
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data is a structure of type LPQP_data_type. On output,
!   pointer array components will have been nullified.
!
!  control is a structure of type LPQP_control_type that contains
!   control parameters. Components are -
!
!  INTEGER control parameters:
!
!   error. Error and warning diagnostics occur on stream error
!
!   out. General output occurs on stream out
!
!   print_level. The level of output required is specified by print_level
!
!  REAL control parameters:
!
!   infinity. Any bound larger than infinity in modulus will be regarded as
!    infinite
!
!  LOGICAL control parameters:
!
!  VNAME_lpqp is an optional pointer array of character strings of length
!   10 that may be used in LPQP_formulate to store names for any additional
!   variables that are introduced when formulating the lp_qp problem. If
!   VNAME_lpqp is present, it will be nullified on exit.
!
!  CNAME_lpqp is an optional pointer array of character strings of length
!   10 that may be used in LPQP_formulate to store names for any additional
!   constraints that are introduced when formulating the lp_qp problem. If
!   CNAME_lpqp is present, it will be nullified on exit.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!    TYPE ( LPQP_data_type ), INTENT( INOUT ) :: data
     TYPE ( LPQP_control_type ), INTENT( OUT ) :: control
!    CHARACTER ( LEN = 10 ), OPTIONAL, ALLOCATABLE,                            &
!      DIMENSION( : ) :: VNAME_lpqp, CNAME_lpqp

!  Set control parameters

     control%h_output_format = 'SPARSE_BY_ROWS'
     control%a_output_format = 'SPARSE_BY_ROWS'

!  End of LPQP_initialize

     END SUBROUTINE LPQP_initialize

!-*-*-*-*-   L P Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

     SUBROUTINE LPQP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LPQP_initialize could (roughly)
!  have been set as:

! BEGIN LPQP SPECIFICATIONS (DEFAULT)
!  error-printout-device                          6
!  printout-device                                6
!  print-level                                    0
!  infinity-value                                 1.0D+19
!  space-critical                                 no
!  deallocate-error-fatal                         no
!  H-output-format                                SPARSE_BY_ROWS
!  A-output-format                                SPARSE_BY_ROWS
!  output-line-prefix                             ""
! END LPQP SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LPQP_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: infinity = print_level + 1
     INTEGER, PARAMETER :: space_critical = infinity + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: h_output_format = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: a_output_format = h_output_format + 1
     INTEGER, PARAMETER :: prefix = a_output_format + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 16 ), PARAMETER :: specname = 'LPQP           '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'

!  Real key-words

     spec( infinity )%keyword = 'infinity-value'

!  Logical key-words

     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

     spec( h_output_format )%keyword = 'H-output-format'
     spec( a_output_format )%keyword = 'A-output-format'
     spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
     ELSE
       CALL SPECFILE_read( device, specname, spec, lspec, control%error )
     END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_string( spec( h_output_format ),                     &
                                  control%h_output_format,                     &
                                  control%error )
     CALL SPECFILE_assign_string( spec( a_output_format ),                     &
                                  control%a_output_format,                     &
                                  control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

     RETURN

     END SUBROUTINE LPQP_read_specfile

!-*-*-*-*-*-*-   L P Q P _ F O R M U L A T E   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE LPQP_formulate( prob, rho, one_norm, data, control, inform,    &
                                VNAME_lpqp, CNAME_lpqp, B_stat, C_stat, cold )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find a quadratic program that is equivalent to the l_1 penalty function
!  for the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     subject to    (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!        and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ), const is a
!  constant, g is an n-vector, H is a symmetric matrix,
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %new_problem_structure is a LOGICAL variable, which must be set to
!    .TRUE. by the user if this is the first problem with this "structure"
!    to be solved since the last call to LPQP_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %Hessian_kind is an INTEGER variable which defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero, and the analytic centre of the
!        feasible region (if gradient_kind = 0) or the the minimizer of
!        the L1LP
!           g^T x + f + rho || min( A x - c_l, c_u - A x, 0 )||_1
!        (if gradient kind /= 0) will be found. %WEIGHT (see below)
!        need not be set
!
!     1  all the weights will be one. %WEIGHT (see below) need not be set
!
!     2  the weights will be those given by %WEIGHT (see below)
!
!    <0  the Hessian H will be used
!
!   %H is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of H (except for the L-BFGS case). Eight storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       %H%type( 1 : 10 ) = 'COORDINATE'
!       %H%val( : )  the values of the components of H
!       %H%row( : )  the row indices of the components of H
!       %H%col( : )  the column indices of the components of H
!       %H%ne        the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of H
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       %H%type( 1 : 14 ) = 'SPARSE_BY_ROWS'
!       %H%val( : )  the values of the components of H, stored row by row
!       %H%col( : )  the column indices of the components of H
!       %H%ptr( : )  pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       %H%type( 1 : 5 ) = 'DENSE'
!       %H%val( : )  the values of the components of H, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!   iv) diagonal
!
!       In this case, the following must be set:
!
!       %H%type( 1 : 8 ) = 'DIAGONAL'
!       %H%val( : )  the values of the %n diagonal components of H
!
!   v) scaled identity
!
!       In this case, the following must be set:
!
!       %H%type( 1 : 15 ) = 'SCALED-IDENTITY'
!       %H%val( 1 )  the value assigned to each diagonal of H
!
!   vi) identity
!
!       In this case, the following must be set:
!
!       %H%type( 1 : 8 ) = 'IDENTITY'
!
!   vii) no Hessian
!
!       In this case, the following must be set:
!
!       %H%type( 1 : 4 ) = 'ZERO' or 'NONE'
!
!   viii) L-BFGS Hessian
!
!       In this case, the following must be set:
!
!        %H%type( 1 : 5 ) = 'LBFGS'

!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored either by coordinates according to
!    sceme (i) above or by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output %H%row will contain
!    the row numbers corresponding to the values in %H%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %H_lm is a structure of type LMS_data_type, whose components hold the
!     L-BFGS Hessian. Access to this structure is via the module GALAHAD_LMS,
!     and this component needs only be set if %H%type( 1 : 5 ) = 'LBFGS.'
!
!   %target_kind is an INTEGER variable which defines possible special
!     targets X0. Possible values are
!
!     0  X0 will be a vector of zeros.
!        %X0 (see below) need not be set
!
!     1  X0 will be a vector of ones.
!        %X0 (see below) need not be set
!
!     any other value - the values of X0 will be those given by %X0 (see below)
!
!   %X0 is a REAL array, which need only be set if %Hessian_kind is larger
!    that 0 and %target_kind /= 0,1. If this is so, it must be of length at
!    least %n, and contain the targets X^0 for the objective function.
!
!   %gradient_kind is an INTEGER variable which defines the type of linear
!    term of the objective function to be used. Possible values are
!
!     0  the linear term g will be zero, and the analytic centre of the
!        feasible region will be found if in addition %Hessian_kind is 0.
!        %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by %G (see below)
!
!   %G is a REAL array of length %n, which must be set by
!    the user to the value of the gradient, g, of the linear term of the
!    quadratic objective function. The i-th component of G, i = 1, ....,
!    n, should contain the value of g_i.
!    On exit, G will most likely have been reordered.
!
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
!
!   %A is a structure of type SMT_type used to hold the matrix A.
!    Three storage formats are permitted:
!    are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       %A%type( 1 : 10 ) = 'COORDINATE'
!       %A%val( : )   the values of the components of A
!       %A%row( : )   the row indices of the components of A
!       %A%col( : )   the column indices of the components of A
!       %A%ne         the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       %A%type( 1 : 14 ) = 'SPARSE_BY_ROWS'
!       %A%val( : )   the values of the components of A, stored row by row
!       %A%col( : )   the column indices of the components of A
!       %A%ptr( : )   pointers to the start of each row, and past the end of
!                     the last row
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       %A%type( 1 : 5 ) = 'DENSE'
!       %A%val( : )   the values of the components of A, stored row by row,
!                     with each the entries in each row in order of
!                     increasing column indicies.
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored either by coordinates according to
!    sceme (i) above or by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output %A%row will contain
!    the row numbers corresponding to the values in %A%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %C is a REAL array of length %m, which is used to store the values of
!    A x. It need not be set on entry. On exit, it will have been filled
!    with appropriate values.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to an estimate of the solution x. On successful exit, it will contain
!    the required solution.
!
!   %C_l, %C_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays c_l and c_u of lower and upper bounds on A x.
!    Any bound c_l_i or c_u_i larger than or equal to control%infinity in
!    absolute value will be regarded as being infinite (see the entry
!    control%infinity). Thus, an infinite lower bound may be specified by
!    setting the appropriate component of %C_l to a value smaller than
!    -control%infinity, while an infinite upper bound can be specified by
!    setting the appropriate element of %C_u to a value larger than
!    control%infinity. On exit, %C_l and %C_u will most likely have been
!    reordered.
!
!   %Y is a REAL array of length %m, which must be set by the user to
!    appropriate estimates of the values of the Lagrange multipliers
!    corresponding to the general constraints c_l <= A x <= c_u.
!    On successful exit, it will contain the required vector of Lagrange
!    multipliers.
!
!   %X_l, %X_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays x_l and x_u of lower and upper bounds on x.
!    Any bound x_l_i or x_u_i larger than or equal to control%infinity in
!    absolute value will be regarded as being infinite (see the entry
!    control%infinity). Thus, an infinite lower bound may be specified by
!    setting the appropriate component of %X_l to a value smaller than
!    -control%infinity, while an infinite upper bound can be specified by
!    setting the appropriate element of %X_u to a value larger than
!    control%infinity. On exit, %X_l and %X_u will most likely have been
!    reordered.
!
!   %Z is a REAL array of length %n, which must be set by the user to
!    appropriate estimates of the values of the dual variables
!    (Lagrange multipliers corresponding to the simple bound constraints
!    x_l <= x <= x_u). On successful exit, it will contain
!   the required vector of dual variables.
!
!  rho is a REAL variable that holds the required value of the penalty
!   parameter for the l_p qp.
!
!  one-norm is a LOGICAL variable that is true if the l_1 norm is to be
!   used and false if the l_infinity norm is to be used.
!
!  control is a structure of type LPQP_control_type that contains
!   control parameters. See LPQP_initialize for details.
!
!  inform is a structure of type LPQP_inform_type that provides
!    information on exit from LPQP_formulate. The component status
!    has possible values:
!
!     0 Normal termination with the problem reformulated.
!
!   - 1 one of the restrictions
!          prob%n     >=  1
!          prob%m     >=  0
!          prob%A%ne  >=  -2
!          prob%H%ne  >=  -2
!       has been violated.
!
!    -2 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -3 One of more of the components A/H_val,  A/H_row,  A/H_col is not
!       large enough to hold the given matrices.
!
!    -4 an entry from the strict upper triangle of H has been input.
!
!    -86 a diagonal Hessian output is requested while H is not diagonal.
!
!    -87 no Hessian output is requested while H is available.
!
!  On exit from QPB_solve, other components of inform give the
!  following:
!
!     alloc_status = The status of the last attempted allocation/deallocation
!     time = the total time spent in the package.
!
!  VNAME_lpqp is an optional pointer array of character strings of length
!   10 that may be used in LPQP_formulate to store names for any additional
!   variables that are introduced when formulating the lp_qp problem. If
!   VNAME_lpqp is present, it must have previouly been nullified or allocated
!   on entry, and will be filled with appropriate variable names on exit.
!
!  CNAME_lpqp is an optional pointer array of character strings of length
!   10 that may be used in LPQP_formulate to store names for any additional
!   constraints that are introduced when formulating the lp_qp problem. If
!   CNAME_lpqp is present, it must have previouly been nullified or allocated
!   on entry, and will be filled with appropriate constraint names on exit.
!
!  B_stat and C_stat are optional INTEGER pointer arrays of length n
!  and m (respectively) that may be extended to accomodate additonal
!  variables and constraints. If the optional INTEGER variable cold has
!  the value 0, the contents of B_stat and C_stat will be preserved when
!  the arrays are extended. This functionality is primarily for use by QPA.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
     TYPE ( LPQP_data_type ), INTENT( INOUT ) :: data
     TYPE ( LPQP_control_type ), INTENT( IN ) :: control
     TYPE ( LPQP_inform_type ), INTENT( OUT ) :: inform
     REAL ( KIND = wp ), INTENT( IN ) :: rho
     LOGICAL, INTENT( IN ) :: one_norm
     CHARACTER ( LEN = 10 ), OPTIONAL, ALLOCATABLE,                            &
       DIMENSION( : ) :: VNAME_lpqp, CNAME_lpqp
     INTEGER, OPTIONAL, ALLOCATABLE, DIMENSION( : ) :: B_stat, C_stat
     INTEGER, OPTIONAL :: cold

!  Local variables

     INTEGER :: m, n, a_ne, h_ne, i, j, l, ll, l1, l2, la, mm, alloc_status
     INTEGER :: m_orig, n_orig, a_ne_orig, h_ne_orig, n_s, n_c, n_r, liw, lw
     INTEGER :: out, error
     REAL :: time_start, time_now, clock_start, clock_now
     REAL ( KIND = wp ) :: cl, cu, infinity
     LOGICAL :: reallocate, vname, cname, lcold, printi, printd, printe
     LOGICAL :: h_row_available, h_row_wanted, a_row_available, a_row_wanted
     LOGICAL :: h_col_available, h_col_wanted, a_col_available, a_col_wanted
     LOGICAL :: h_ptr_available, h_ptr_wanted, a_ptr_available, a_ptr_wanted
     LOGICAL :: h_val_available, h_val_wanted, h_diagonal, h_none
     CHARACTER ( LEN = 80 ) :: array_name

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

     printi = control%print_level > 0
     IF ( printi ) THEN
       out = control%out ; error = control%error
       printd = control%print_level > 4 .AND. out > 0 ; printe = error > 0
     ELSE
       out = 0 ; error = 0
       printd = .FALSE. ; printe = .FALSE.
     END IF

     IF ( printd ) WRITE( out, "( A, ' entering LPQP_fomulate ' )" ) prefix

!  initialize time

     inform%time = 0.0
     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  compute how many variables and constraints there will be

     IF ( prob%n < 1 ) THEN
       IF ( printe ) WRITE( error, "( A, ' LPQP error: prob%n = ', I0 )" )     &
         prefix, prob%n
       inform%status = GALAHAD_error_restrictions ; GO TO 800
     ELSE
       n = prob%n
     END IF
     n_orig = n ; data%n = n_orig

     IF ( prob%m < 0 ) THEN
       IF ( printe ) WRITE( error, "( A, ' LPQP error: prob%m = ', I0 )" )     &
         prefix, prob%m
       inform%status = GALAHAD_error_restrictions ; GO TO 800
     ELSE
       m = prob%m
     END IF
     m_orig = m ; data%m = m_orig

!  record the Jacobian input array type

     IF ( ALLOCATED( data%A_type ) ) DEALLOCATE( data%A_type )
     CALL SMT_put( data%A_type, SMT_get( prob%A%type ), alloc_status )
     IF ( ALLOCATED( data%H_type ) ) DEALLOCATE( data%H_type )
     IF ( prob%Hessian_kind < 0 )                                              &
       CALL SMT_put( data%H_type, SMT_get( prob%H%type ), alloc_status )

     IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
       a_ne = n * m
       a_row_available = .FALSE. ; a_col_available = .FALSE.
       a_ptr_available = .FALSE.
     ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
       a_ne = prob%A%ptr( m + 1 ) - 1
       a_row_available = .FALSE. ; a_col_available = .TRUE.
       a_ptr_available = .TRUE.
     ELSE IF ( SMT_get( prob%A%type ) == 'COORDINATE' ) THEN
       a_ne = prob%A%ne
       a_row_available = .TRUE. ; a_col_available = .TRUE.
       a_ptr_available = .FALSE.
     ELSE
       IF ( printe ) WRITE( error, "( A, ' LPQP error: prob%A%type = ', A )" ) &
         prefix, SMT_get( prob%A%type )
       inform%status = GALAHAD_error_restrictions ; GO TO 800
     END IF
     a_ne_orig = a_ne
     data%A_ne = prob%A%ne

!  record the required Jacobian output format

     SELECT CASE ( TRIM( control%a_output_format ) )
     CASE ( 'COORDINATE' )
       data%a_output_format = a_output_coordinate
       a_row_wanted = .TRUE. ; a_col_wanted = .TRUE. ; a_ptr_wanted = .FALSE.
!    CASE ( 'DENSE' )
!      data%a_output_format = a_output_dense
!      a_row_wanted = .FALSE. ; a_col_wanted = .FALSE. ; a_ptr_wanted = .FALSE.
     CASE DEFAULT
       data%a_output_format = a_output_sparse_by_rows
       a_row_wanted = .FALSE. ; a_col_wanted = .TRUE. ; a_ptr_wanted = .TRUE.
     END SELECT

!  record the Hessian input array type

     IF ( prob%Hessian_kind < 0 ) THEN
       IF ( SMT_get( prob%H%type ) == 'LBFGS' ) THEN
         data%h_output_format = h_output_lbfgs
         h_row_wanted = .FALSE. ; h_col_wanted = .FALSE. ; h_ptr_wanted =.FALSE.
         h_val_wanted = .FALSE.
         h_diagonal = .FALSE. ; h_none = .FALSE.
       ELSE
         IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
           h_ne_orig = ( n * ( n + 1 ) ) / 2 ; h_ne = h_ne_orig
           h_row_available = .FALSE. ; h_col_available = .FALSE.
           h_ptr_available = .FALSE. ; h_val_available = .TRUE.
           h_diagonal = .FALSE.
           h_none = .FALSE.
         ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
           h_ne_orig = prob%H%ptr( n + 1 ) - 1 ; h_ne = h_ne_orig
           h_row_available = .FALSE. ; h_col_available = .TRUE.
           h_ptr_available = .TRUE. ; h_val_available = .TRUE.
           h_diagonal = .TRUE.
           DO i = 1, n
             IF ( COUNT( prob%H%col( prob%H%ptr( i ) :                         &
                                     prob%H%ptr( i + 1 ) - 1 ) /= i ) > 0 ) THEN
               h_diagonal = .FALSE. ; EXIT
             END IF
           END DO
           h_none = prob%H%ptr( n + 1 ) == 1
       ELSE IF ( SMT_get( prob%H%type ) == 'COORDINATE' ) THEN
           h_ne_orig = prob%H%ne ; h_ne = h_ne_orig
           h_row_available = .TRUE. ; h_col_available = .TRUE.
           h_ptr_available = .FALSE. ; h_val_available = .TRUE.
           h_diagonal = .TRUE.
           DO i = 1, prob%H%ne
             IF ( prob%H%row( i ) /= prob%H%col( i ) ) THEN
               h_diagonal = .FALSE. ; EXIT
             END IF
           END DO
           h_none = prob%H%ne == 0
         ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
           h_ne_orig = n ; h_ne = h_ne_orig
           h_row_available = .FALSE. ; h_col_available = .FALSE.
           h_ptr_available = .FALSE. ; h_val_available = .TRUE.
           h_diagonal = .TRUE.
           h_none = .FALSE.
         ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
           h_ne_orig = 1 ; h_ne = n
           h_row_available = .FALSE. ; h_col_available = .FALSE.
           h_ptr_available = .FALSE. ; h_val_available = .TRUE.
           h_diagonal = .TRUE. ; h_none = .FALSE.
         ELSE IF ( SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
           h_ne_orig = 0 ; h_ne = n
           h_row_available = .FALSE. ; h_col_available = .FALSE.
           h_ptr_available = .FALSE. ; h_val_available = .FALSE.
           h_diagonal = .TRUE. ; h_none = .FALSE.
         ELSE IF ( SMT_get( prob%H%type ) == 'NONE' .OR.                       &
                   SMT_get( prob%H%type ) == 'ZERO' ) THEN
           h_ne_orig = 0 ; h_ne = n
           h_row_available = .FALSE. ; h_col_available = .FALSE.
           h_ptr_available = .FALSE. ; h_val_available = .FALSE.
           h_diagonal = .TRUE. ; h_none = .TRUE.
         ELSE
           IF ( printe ) WRITE( error, "( A, ' LPQP error: prob%H%type = ',    &
          &   A )") prefix, SMT_get( prob%H%type )
           inform%status = GALAHAD_error_restrictions ; GO TO 800
         END IF
         data%H_ne = prob%H%ne
       END IF
     ELSE
       IF ( prob%Hessian_kind == 0 ) THEN
         h_ne = 0 ; h_none = .TRUE.
       ELSE
         h_ne = n ; h_none = .FALSE.
       END IF
       h_row_available = .FALSE. ; h_col_available = .FALSE.
       h_ptr_available = .FALSE. ; h_val_available = .FALSE.
       h_diagonal = .TRUE.
     END IF

!  record the required Hessian output format

     SELECT CASE ( TRIM( control%h_output_format ) )
     CASE ( 'COORDINATE' )
       data%h_output_format = h_output_coordinate
       h_row_wanted = .TRUE. ; h_col_wanted = .TRUE. ; h_ptr_wanted = .FALSE.
       h_val_wanted = .TRUE.
     CASE ( 'DENSE' )
       inform%status = GALAHAD_not_yet_implemented ; GO TO 800
!      data%h_output_format = h_output_dense
!      h_row_wanted = .FALSE. ; h_col_wanted = .FALSE. ; h_ptr_wanted =.FALSE.
!      h_val_wanted = .TRUE.
     CASE ( 'DIAGONAL' )
       IF ( h_diagonal ) THEN
         data%h_output_format = h_output_diagonal
         h_row_wanted = .FALSE. ; h_col_wanted = .FALSE.
         h_ptr_wanted = .FALSE. ; h_val_wanted = .TRUE.
         h_ne = n
       ELSE
         inform%status = GALAHAD_error_h_not_diagonal ; GO TO 800
       END IF
     CASE ( 'NONE' )
       IF ( h_none ) THEN
         data%h_output_format = h_output_none
         h_row_wanted = .FALSE. ; h_col_wanted = .FALSE.
         h_ptr_wanted = .FALSE. ; h_val_wanted = .FALSE.
         h_ne = 0
       ELSE
         inform%status = GALAHAD_error_h_not_permitted ; GO TO 800
       END IF
     CASE DEFAULT
       data%h_output_format = h_output_sparse_by_rows
       h_row_wanted = .FALSE. ; h_col_wanted = .TRUE. ; h_ptr_wanted = .TRUE.
       h_val_wanted = .TRUE.
     END SELECT

     data%one_norm = one_norm

     IF ( PRESENT( cold ) ) THEN
       IF ( cold == 0 ) THEN ; lcold = .TRUE. ; ELSE ; lcold = .FALSE. ; END IF
     ELSE
       lcold = .FALSE.
     END IF

!  ---------------------------------------------------------------------------

!  Total number of
!    variables (n_lpqp)
!    constraints (m_lpqp) &
!    nonzeros in the Jacobian (a_ne_lpqp)

!  1_1 problem -
!  -----------

!    n_lpqp = n + 2 * ( # equalities + # two-sided inequalities )
!               + # one-sided inequalities
!    m_lpqp = m + 2 * # two-sided inequalities
!    a_ne_lpqp = a_ne + 2 * # equalities + 5 * # two-sided inequalities )
!                     + # one-sided inequalities
!  [used to be
!    m_lpqp = m + 2 * ( # equalities + # two-sided inequalities )
!    a_ne_lpqp = a_ne + 5 * ( # equalities + # two-sided inequalities )
!                     + # one-sided inequalities ]

!  1_infty problem -
!  ---------------

!    n_lpqp = n + 1 + # equalities + # two-sided inequalities
!    m_lpqp = m + 2 * ( # equalities + # two-sided inequalities )
!    a_ne_lpqp = a_ne + 5 * ( # equalities + # two-sided inequalities )
!                     + # one-sided inequalities

!  Details of how constraints are handled

!  ========================================
!  For equality constraints: c_i(x) = c^e_i
!  ========================================

!  1_1 problem -
!  -----------

!    Define extra variables c_i and s_i

!          c_i(x) - c_i + s_i = c_i^e
!          c_i >= 0, s_i >= 0

!    Term rho ( s_i + c_i ) in objective for each constraint

!  1_infty problem -
!  ---------------

!    Define extra variables s and c_i

!          c_i(x) - c_i = c_i^e
!          s - c_i >= 0
!          s + c_i >= 0
!  (maybe) s >= 0

!    Single term rho s in objective

!  ====================================================
!  For inequality constraints: c_i^l <= c_i(x) <= c^u_i
!  ====================================================

!  1_1 problem -
!  -----------

!    Define extra variables s_i and c_i

!          c_i(x) - c_i = 0
!          c_i + s_i >= c_i^l
!          c_i - s_i <= c_i^u
!          s_i >= 0

!    Term rho s_i in objective for each constraint ]

!  1_infty problem -
!  ---------------

!    Define extra variables s and c_i

!          c_i(x) - c_i = 0
!          c_i + s >= c_i^l
!          c_i - s <= c_i^u
!          s >= 0

!    Single term rho s in objective

!  ===========================================
!  For inequality constraints: c_i^l <= c_i(x)
!  ===========================================

!  1_1 problem -
!  -----------

!    Define extra variables s_i

!          c_i(x) + s_i >= c_i^l
!          s_i >= 0

!    Term rho s_i in objective for each constraint

!  1_infty problem -
!  ---------------

!    Define extra variable s

!          c_i(x) + s >= c_i^l
!          s >= 0

!    Single term rho s in objective

!  ===========================================
!  For inequality constraints: c_i(x) <= c^u_i
!  ===========================================

!  1_1 problem -
!  -----------

!    Define extra variables s_i

!          c_i(x) - s_i <= c_i^u
!          s_i >= 0

!    Term rho s_i in objective for each constraint

!  1_infty problem -
!  ---------------

!    Define extra variable s

!          c_i(x) - s <= c_i^u
!          s >= 0

!    Single term rho s in objective

!  ---------------------------------------------------------------------------

     la = a_ne ; data%m_b = 0
     IF ( .NOT. one_norm .AND. m_orig > 0 ) n = n + 1

     DO i = 1, m
       cl = prob%c_l( i ) ; cu = prob%c_u( i )

!  equality constraint

       IF ( cu == cl ) THEN
         la = la + 2
         a_ne = a_ne + 2
         IF ( one_norm ) THEN
           n = n + 2
         ELSE
           n = n + 1
         END IF

!  constraint bounded on both sides

       ELSE IF ( cl > - control%infinity ) THEN
         IF ( cu < control%infinity ) THEN
           IF ( cl <= cl ) THEN
             m = m + 2
             la = la + 1
             a_ne = a_ne + 5
             data%m_b = data%m_b + 1
             IF ( one_norm ) THEN
               n = n + 2
             ELSE
               n = n + 1
             END IF

!  inconsistent constraints

           ELSE
             IF ( printe ) WRITE( error, "( A, ' LPQP error:  inconsistent',   &
            &                                  ' constraints' )" ) prefix
             inform%status = GALAHAD_error_primal_infeasible ; GO TO 800
           END IF

!  constraint bounded from below

         ELSE
           la = la + 1
           a_ne = a_ne + 1
           IF ( one_norm ) n = n + 1
         END IF

!  constraint bounded from above

       ELSE
         IF ( cu < control%infinity ) THEN
           la = la + 1
           a_ne = a_ne + 1
           IF ( one_norm ) n = n + 1

!  free constraint

         ELSE
         END IF
       END IF
     END DO

!  record problem dimensions

     prob%n = n
     prob%m = m
     IF ( TRIM( control%h_output_format ) == 'DIAGONAL' .OR.                   &
          prob%Hessian_kind >= 0 ) h_ne = n

!  ensure that there is sufficient space for the integer components

     liw = 0
     IF ( a_row_available .AND. a_row_wanted ) liw = MAX( liw,                 &
       LPQP_integer_required( a_ne_orig, a_ne, prob%A%row, data%IW ) )
     IF ( a_col_available .AND. a_col_wanted ) liw = MAX( liw,                 &
       LPQP_integer_required( a_ne_orig, a_ne, prob%A%col, data%IW ) )
     IF ( a_ptr_available .AND. a_ptr_wanted ) liw = MAX( liw,                 &
       LPQP_integer_required( a_ne_orig, a_ne, prob%A%ptr, data%IW ) )
     IF ( h_row_available .AND. h_row_wanted ) liw = MAX( liw,                 &
       LPQP_integer_required( h_ne_orig, h_ne, prob%H%row, data%IW ) )
     IF ( h_col_available .AND. h_col_wanted ) liw = MAX( liw,                 &
       LPQP_integer_required( h_ne_orig, h_ne, prob%H%col, data%IW ) )
     IF ( h_ptr_available .AND. h_ptr_wanted ) liw = MAX( liw,                 &
       LPQP_integer_required( h_ne_orig, h_ne, prob%H%ptr, data%IW ) )

     IF ( PRESENT( B_stat ) .AND. lcold )                                      &
       liw = MAX( liw, LPQP_integer_required( n_orig + 1, n + 1, B_stat,       &
                                              data%IW ) )
     IF ( PRESENT( C_stat ) .AND. lcold )                                      &
       liw = MAX( liw, LPQP_integer_required( m_orig + 1, m + 1, C_stat,       &
                                              data%IW ) )

!  provide integer workspace

     IF ( data%m_b > 0 ) THEN
       array_name = 'lpqp: data%BOTH'
       CALL SPACE_resize_array( 2, data%m_b, data%BOTH,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = error )
       IF ( inform%status /= 0 ) GO TO 900
     END IF

     IF ( liw > 0 ) THEN
       array_name = 'lpqp: data%IW'
       CALL SPACE_resize_array( liw, data%IW,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = error )
       IF ( inform%status /= 0 ) GO TO 900
     END IF

!  enlarge integer components if required

     IF ( a_row_wanted ) THEN
       array_name = 'lpqp: prob%A%row'
       IF ( a_row_available ) THEN
         CALL LPQP_integer_reallocate( a_ne_orig, a_ne, prob%A%row, data%IW,   &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc )
       ELSE
         CALL SPACE_resize_array( a_ne, prob%A%row,                            &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = error )
       END IF
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
     END IF

     IF ( a_col_wanted ) THEN
       array_name = 'lpqp: prob%A%col'
       IF ( a_col_available ) THEN
         CALL LPQP_integer_reallocate( a_ne_orig, a_ne, prob%A%col, data%IW,   &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc )
       ELSE
         CALL SPACE_resize_array( a_ne, prob%A%col,                            &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = error )
       END IF
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
     END IF

     IF ( a_ptr_wanted ) THEN
       array_name = 'lpqp: prob%A%ptr'
       IF ( a_ptr_available ) THEN
         CALL LPQP_integer_reallocate( m_orig + 1, m + 1, prob%A%ptr, data%IW, &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc )
       ELSE
         CALL SPACE_resize_array( m + 1, prob%A%ptr,                           &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = error )
       END IF
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
     END IF

     IF ( h_row_wanted ) THEN
       array_name = 'lpqp: prob%H%row'
       IF ( h_row_available ) THEN
         CALL LPQP_integer_reallocate( h_ne_orig, h_ne, prob%H%row, data%IW,   &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc )
       ELSE
         CALL SPACE_resize_array( h_ne, prob%H%row,                            &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = error )
       END IF
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
     END IF

     IF ( h_col_wanted ) THEN
       array_name = 'lpqp: prob%H%col'
       IF ( h_col_available ) THEN
         CALL LPQP_integer_reallocate( h_ne_orig, h_ne, prob%H%col, data%IW,   &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc )
       ELSE
         CALL SPACE_resize_array( h_ne, prob%H%col,                            &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = error )
       END IF
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
     END IF

     IF ( h_ptr_wanted ) THEN
       array_name = 'lpqp: prob%H%ptr'
       IF ( h_ptr_available ) THEN
         CALL LPQP_integer_reallocate( n_orig + 1, n + 1, prob%H%ptr, data%IW, &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc )
       ELSE
         CALL SPACE_resize_array( n + 1, prob%H%ptr,                           &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = error )
       END IF
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
     END IF

     IF ( PRESENT( B_stat ) ) THEN
       array_name = 'lpqp: B_stat'
       CALL LPQP_integer_reallocate_cold( n_orig + 1, n + 1, B_stat, lcold,    &
                      data%IW, inform%status, inform%alloc_status,             &
                      array_name = array_name, bad_alloc = inform%bad_alloc )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
     END IF

     IF ( PRESENT( C_stat ) ) THEN
       array_name = 'lpqp: C_stat'
       CALL LPQP_integer_reallocate_cold( m_orig + 1, m + 1, C_stat, lcold,    &
                      data%IW, inform%status, inform%alloc_status,             &
                      array_name = array_name, bad_alloc = inform%bad_alloc )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
     END IF

!  ensure that there is sufficient space for the real components

     lw = MAX( LPQP_real_required( a_ne_orig, a_ne, prob%A%val, data%W ),      &
               LPQP_real_required( n_orig, n, prob%G, data%W ),                &
               LPQP_real_required( m_orig, m, prob%C_l, data%W ),              &
               LPQP_real_required( m_orig, m, prob%C_u, data%W ),              &
               LPQP_real_required( n_orig, n, prob%X_l, data%W ),              &
               LPQP_real_required( n_orig, n, prob%X_u, data%W ),              &
               LPQP_real_required( n_orig, n, prob%X, data%W ),                &
               LPQP_real_required( m_orig, m, prob%Y, data%W ),                &
               LPQP_real_required( n_orig, n, prob%Z, data%W ),                &
               LPQP_real_required( m_orig, m, prob%C, data%W ) )
     IF ( h_val_available .AND. h_val_wanted ) lw                              &
        = MAX( LPQP_real_required( h_ne_orig, h_ne, prob%H%val, data%W ), lw )
     IF ( h_diagonal ) lw = MAX( lw, n )

!  provide real workspace

     IF ( lw > 0 ) THEN
       array_name = 'lpqp: data%W'
       CALL SPACE_resize_array( lw, data%W,                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = error )
       IF ( inform%status /= 0 ) GO TO 900

!  enlarge real components if required

       array_name = 'lpqp: prob%A%val'
       CALL LPQP_real_reallocate( a_ne_orig, a_ne, prob%A%val, data%W,         &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900

       IF ( h_val_wanted ) THEN
         array_name = 'lpqp: prob%H%val'
         IF ( h_val_available ) THEN
           CALL LPQP_real_reallocate( h_ne_orig, h_ne, prob%H%val, data%W,     &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc )
         ELSE
           CALL SPACE_resize_array( h_ne, prob%H%val,                          &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = error )
         END IF
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
       END IF

       array_name = 'lpqp: prob%G'
       IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
         CALL SPACE_resize_array( n, prob%G,                                   &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         IF ( prob%gradient_kind == 0 ) THEN
           prob%G( : n_orig ) = zero
         ELSE
           prob%G( : n_orig ) = one
         END IF
       ELSE
         CALL LPQP_real_reallocate( n_orig, n, prob%G, data%W,                 &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
       END IF

!  include gradient and function terms from original least-distance format

       IF ( prob%Hessian_kind == 1 ) THEN
         IF ( prob%target_kind == 1 ) THEN
           prob%G( : n_orig ) = prob%G( : n_orig ) - one
           prob%f = prob%f + half * REAL( n_orig, wp )
         ELSE IF ( prob%target_kind /= 0 ) THEN
           prob%G( : n_orig ) = prob%G( : n_orig ) - prob%X0( : n_orig )
           prob%f = prob%f + half * SUM( prob%X0( : n_orig ) ** 2 )
         END IF
       ELSE IF ( prob%Hessian_kind == 2 ) THEN
         IF ( prob%target_kind == 1 ) THEN
           prob%G( : n_orig ) = prob%G( : n_orig )                             &
             - prob%WEIGHT( : n_orig ) ** 2
           prob%f = prob%f + half * SUM( prob%WEIGHT( : n_orig ) ** 2 )
         ELSE IF ( prob%target_kind /= 0 ) THEN
           prob%G( : n_orig ) = prob%G( : n_orig )                             &
             - prob%X0( : n_orig ) * prob%WEIGHT( : n_orig ) ** 2
           prob%f = prob%f + half                                              &
             * SUM( ( prob%X0( : n_orig ) * prob%WEIGHT( : n_orig ) ) ** 2 )
         END IF
       END IF

       array_name = 'lpqp: prob%C_l'
       CALL LPQP_real_reallocate( m_orig, m, prob%C_l, data%W,                 &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900

       array_name = 'lpqp: prob%C_u'
       CALL LPQP_real_reallocate( m_orig, m, prob%C_u, data%W,                 &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900

       array_name = 'lpqp: prob%X_l'
       CALL LPQP_real_reallocate( n_orig, n, prob%X_l, data%W,                 &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900

       array_name = 'lpqp: prob%X_u'
       CALL LPQP_real_reallocate( n_orig, n, prob%X_u, data%W,                 &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900

       array_name = 'lpqp: prob%X'
       CALL LPQP_real_reallocate( n_orig, n, prob%X, data%W,                   &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900

       array_name = 'lpqp: prob%Y'
       CALL LPQP_real_reallocate( m_orig, m, prob%Y, data%W,                   &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900

       array_name = 'lpqp: prob%Z'
       CALL LPQP_real_reallocate( n_orig, n, prob%Z, data%W,                   &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900

       array_name = 'lpqp: prob%C'
       CALL LPQP_real_reallocate( m_orig, m, prob%C, data%W,                   &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
     END IF

!  if required, allocate space for names for additional variables/constraints

     vname = PRESENT( VNAME_lpqp )
     IF ( vname ) THEN
       reallocate = .TRUE.
       IF ( ALLOCATED( VNAME_lpqp ) ) THEN
         IF ( LBOUND( VNAME_lpqp, 1 ) /= n_orig + 1 .OR.                       &
              UBOUND( VNAME_lpqp, 1 ) < n ) THEN
           DEALLOCATE( VNAME_lpqp ) ; ELSE ; reallocate = .FALSE.
         END IF
       END IF
       IF ( reallocate ) THEN
         ALLOCATE( VNAME_lpqp( n_orig + 1 : n ), STAT = inform%alloc_status )
         IF ( inform%alloc_status /= 0 ) THEN
           inform%bad_alloc = 'VNAME_lpqp' ; GO TO 900 ; END IF
       END IF
     END IF

     cname = PRESENT( CNAME_lpqp )
     IF ( cname ) THEN
       reallocate = .TRUE.
       IF ( ALLOCATED( CNAME_lpqp ) ) THEN
         IF ( LBOUND( CNAME_lpqp, 1 ) /= m_orig + 1 .OR.                       &
              UBOUND( CNAME_lpqp, 1 ) < m ) THEN
           DEALLOCATE( CNAME_lpqp ) ; ELSE ; reallocate = .FALSE.
         END IF
       END IF
       IF ( reallocate ) THEN
         ALLOCATE( CNAME_lpqp( m_orig + 1 : m ), STAT = inform%alloc_status )
         IF ( inform%alloc_status /= 0 ) THEN
           inform%bad_alloc = 'CNAME_lpqp' ; GO TO 900 ; END IF
       END IF
     END IF

!  write the l_p Hessian in co-ordinate output format
!  --------------------------------------------------

     IF ( data%h_output_format == h_output_coordinate ) THEN
       prob%H%n = n ; prob%H%m = n

       IF ( prob%Hessian_kind < 0 ) THEN

!  original dense storage

         IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
           l = 0
           DO i = 1, n_orig
             DO j = 1, i
               l = l + 1 ; prob%H%row( l ) = i ; prob%H%col( l ) = j
             END DO
           END DO

!  original row-wise storage

         ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
           DO i = 1, n_orig
             DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
               prob%H%row( l ) = i
             END DO
           END DO

!  original diagonal storage

         ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
           DO i = 1, n_orig
             prob%H%row( i ) = i ; prob%H%col( i ) = i
           END DO

!  original scaled identity storage

         ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
           DO i = 1, n_orig
             prob%H%row( i ) = i ; prob%H%col( i ) = i
             prob%H%val( i ) = prob%H%val( 1 )
           END DO

!  original dentity storage

         ELSE IF ( SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
           DO i = 1, n_orig
             prob%H%row( i ) = i ; prob%H%col( i ) = i ; prob%H%val( i ) = one
           END DO

!  original no-Hessian storage

         ELSE IF ( SMT_get( prob%H%type ) == 'NONE' .OR.                       &
                   SMT_get( prob%H%type ) == 'ZERO' ) THEN

!  original co-ordinate storage

         ELSE
         END IF

!  original least-distance format, unit weights

       ELSE IF ( prob%Hessian_kind == 1 ) THEN
         DO i = 1, n_orig
           prob%H%row( i ) = i ; prob%H%col( i ) = i ; prob%H%val( i ) = one
         END DO
         h_ne = n_orig

!  original least-distance format, variable weights

       ELSE IF ( prob%Hessian_kind == 2 ) THEN
         DO i = 1, n_orig
           prob%H%row( i ) = i ; prob%H%col( i ) = i
           prob%H%val( i ) = prob%WEIGHT( i ) ** 2
         END DO
         h_ne = n_orig

!  linear program

       ELSE
         h_ne = 0
       END IF

       prob%H%ne = h_ne
       IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
       CALL SMT_put( prob%H%type, 'COORDINATE', alloc_status )

!  write the l_p Hessian in dense output format
!  --------------------------------------------

!    ELSE IF ( data%h_output_format == h_output_dense ) THEN

!  write the l_p Hessian in diagonal output format
!  -----------------------------------------------

     ELSE IF ( data%h_output_format == h_output_diagonal ) THEN

       IF ( prob%Hessian_kind < 0 ) THEN

!  original row-wise storage

         IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
           data%W( : n_orig ) = zero
           DO i = 1, n_orig
             DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
               data%W( i ) = data%W( i ) + prob%H%val( l )
             END DO
           END DO

!  original diagonal storage

         ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
           DO i = 1, n_orig
             data%W( i ) = prob%H%val( i )
           END DO

!  original scaled identity storage

         ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
           DO i = 1, n_orig
             data%W( i ) = prob%H%val( 1 )
           END DO

!  original dentity storage

         ELSE IF ( SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
           DO i = 1, n_orig
             data%W( i ) = one
           END DO

!  original no-Hessian storage

         ELSE IF ( SMT_get( prob%H%type ) == 'NONE' .OR.                       &
                   SMT_get( prob%H%type ) == 'ZERO' ) THEN
           data%W( : n_orig ) = zero

!  original co-ordinate storage

         ELSE
           DO l = 1, h_ne_orig
             data%W( prob%H%row( l ) )                                         &
               = data%W( prob%H%row( l ) ) + prob%H%val( l )
           END DO
         END IF

!  original least-distance format, unit weights

       ELSE IF ( prob%Hessian_kind == 1 ) THEN
         data%W( : n_orig ) = one
         h_ne = n

!  original least-distance format, variable weights

       ELSE IF ( prob%Hessian_kind == 2 ) THEN
         data%W( : n_orig ) = prob%WEIGHT( : n_orig ) ** 2
         h_ne = n

!  linear program

       ELSE
         h_ne = 0
       END IF

       prob%H%val( : n_orig ) = data%W( : n_orig )
       prob%H%val( n_orig + 1 : n ) = zero
       prob%H%n = n ; prob%H%ne = n
       IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
       CALL SMT_put( prob%H%type, 'DIAGONAL', alloc_status )

!  write the l_p Hessian in sparse row output format
!  --------------------------------------------------

     ELSE IF ( data%h_output_format == h_output_sparse_by_rows ) THEN

       IF ( prob%Hessian_kind < 0 ) THEN

!  first convert H into row order

!  original dense storage

         IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
           l = 1
           DO i = 1, n_orig
             prob%H%ptr( i ) = l
             DO j = 1, i
               prob%H%col( l ) = j
               l = l + 1
             END DO
           END DO
           prob%H%ptr( n_orig + 1 ) = l

!  original row-wise storage

         ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
           DO i = 1, n_orig
             DO j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
               IF ( i < prob%H%col( j ) ) THEN
                 IF ( printe ) WRITE( error, "( A, ' LPQP error:  H entry in', &
                &                              ' the upper triangle' )" ) prefix
                 inform%status = GALAHAD_error_upper_entry ; GO TO 800
               END IF
             END DO
           END DO

!  original diagonal storage

         ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
           DO i = 1, n_orig
             prob%H%col( i ) = i ; prob%H%ptr( i ) = i
           END DO
           prob%H%ptr( n_orig + 1 ) = n_orig + 1

!  original scaled identity storage

         ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
           DO i = 1, n_orig
             prob%H%col( i ) = i ; prob%H%ptr( i ) = i
             prob%H%val( i ) = prob%H%val( 1 )
           END DO
           prob%H%ptr( n_orig + 1 ) = n_orig + 1

!  original dentity storage

         ELSE IF ( SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
           DO i = 1, n_orig
             prob%H%col( i ) = i ; prob%H%ptr( i ) = i ; prob%H%val( i ) = one
           END DO
           prob%H%ptr( n_orig + 1 ) = n_orig + 1

!  original no-Hessian storage

         ELSE IF ( SMT_get( prob%H%type ) == 'NONE' .OR.                       &
                   SMT_get( prob%H%type ) == 'ZERO' ) THEN
           prob%H%ptr( : n_orig + 1 ) = 1

!  original co-ordinate storage

         ELSE
           DO l = 1, h_ne_orig
             i = prob%H%row( l )
             IF ( i < prob%H%col( l ) ) THEN
               IF ( printe ) WRITE( error, "( A, ' LPQP error:  H entry in',   &
              &                              ' the upper triangle' )" ) prefix
               inform%status = GALAHAD_error_upper_entry ; GO TO 800
             END IF
           END DO

!          IF ( a_ne_orig == - 2 ) THEN
!            liw = MAX( m_orig, n_orig ) + 1
!          ELSE
             liw = n_orig + 1
!          END IF
           reallocate = .TRUE.
           IF ( ALLOCATED( data%IW ) ) THEN
             IF ( SIZE( data%IW ) < liw ) THEN
               DEALLOCATE( data%IW ) ; ELSE ; reallocate = .FALSE.
             END IF
           END IF
           IF ( reallocate ) THEN
             ALLOCATE( data%IW( liw ), STAT = inform%alloc_status )
             IF ( inform%alloc_status /= 0 ) THEN
               inform%bad_alloc = 'data%IW' ; GO TO 900
             END IF
           END IF
           liw = n_orig + 1

           CALL SORT_reorder_by_rows( n_orig, n_orig, h_ne_orig,               &
                                      prob%H%row( : h_ne_orig ),               &
                                      prob%H%col( : h_ne_orig ),  h_ne_orig,   &
                                      prob%H%val( : h_ne_orig ),               &
                                      prob%H%ptr( : n_orig + 1 ), n_orig + 1,  &
                                      data%IW( : liw ), liw, error,            &
                                      out, inform%status )
           IF ( inform%status > 0 ) THEN
             IF ( printe ) WRITE( error, "( A, ' SORT error: status = ', I0 )")&
               prefix, inform%status
             inform%status = GALAHAD_error_sort ; GO TO 800
           END IF

         END IF

!  original least-distance format, unit weights

       ELSE IF ( prob%Hessian_kind == 1 ) THEN
         DO i = 1, n_orig
           prob%H%col( i ) = i ; prob%H%ptr( i ) = i ; prob%H%val( i ) = one
         END DO
         prob%H%ptr( n_orig + 1 ) = n_orig + 1
         h_ne = n_orig

!  original least-distance format, variable weights

       ELSE IF ( prob%Hessian_kind == 2 ) THEN
         DO i = 1, n_orig
           prob%H%col( i ) = i ; prob%H%ptr( i ) = i
           prob%H%val( i ) = prob%WEIGHT( i ) ** 2
         END DO
         prob%H%ptr( n_orig + 1 ) = n_orig + 1
         h_ne = n_orig

!  linear program

       ELSE
         prob%H%ptr( : n_orig + 1 ) = 1
         h_ne = 0
       END IF

!  record the pointers to the start of each row for the new variables

       prob%H%ptr( n_orig + 2 : n + 1 ) = prob%H%ptr( n_orig + 1 )

       IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
       CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', alloc_status )

!  write the l_p Hessian in zero output format
!  -------------------------------------------

     ELSE IF ( data%h_output_format == h_output_none ) THEN
       IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
       CALL SMT_put( prob%H%type, 'NONE', alloc_status )
     END IF

!  write the l_p constraints in co-ordinate output format
!  ------------------------------------------------------

     IF ( data%a_output_format == a_output_coordinate ) THEN

!  original dense storage

       IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
         prob%A%m = m ; prob%A%n = n ; l = 0
         DO i = 1, n_orig
           DO j = 1, m_orig
             l = l + 1
             prob%A%row( l ) = i ; prob%A%col( l ) = j
           END DO
         END DO

!  original row-wise storage

       ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
         prob%A%n = n ; prob%A%m = n
         DO i = 1, m_orig
           DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
             prob%A%row( l ) = i
           END DO
         END DO

!  original co-ordinate storage

       ELSE
       END IF

       IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
       CALL SMT_put( prob%A%type, 'COORDINATE', alloc_status )

!  compute the constraint residuals

      prob%C( : m_orig ) = zero
      DO l = 1, a_ne_orig
        i =  prob%A%row( l )
        prob%C( i ) = prob%C( i ) + prob%A%val( l ) * prob%X( prob%A%col( l ) )
      END DO

!  write the l_p Jacobian in row format

       n_r = m_orig ; la = a_ne_orig ; prob%A%ne = a_ne
       IF ( one_norm ) THEN
         n_c = n_orig + 1 ; n_s = n_orig + 1
       ELSE
         n_c = n_orig ; n_s = n
         IF ( m_orig > 0 ) prob%X( n_s ) = zero
       END IF
       infinity = ten * control%infinity

       data%m_b = 0
       DO i = 1, m_orig
         cl = prob%c_l( i ) ; cu = prob%c_u( i )

!  equality constraint

         IF ( cu == cl ) THEN
           la = la + 1
           prob%A%row( la ) = i
           prob%A%col( la ) = n_s
           prob%A%val( la ) = one
           n_c = n_c + 1 ; la = la + 1
           prob%A%row( la ) = i
           prob%A%col( la ) = n_c
           prob%A%val( la ) = - one
           prob%C_l( i ) = cl
           prob%C_u( i ) = cu

           prob%X_u( n_c ) = infinity
           IF ( vname ) VNAME_lpqp( n_c ) = 'C' // LPQP_char( i )

           IF ( one_norm ) THEN
             prob%G( n_c ) = rho
             prob%X( n_c ) = MAX( prob%C( i ) - cl, zero )
             prob%X_l( n_c ) = zero
             prob%G( n_s ) = rho
             prob%X( n_s ) = MAX( cl - prob%C( i ), zero )
             prob%X_l( n_s ) = zero
             prob%X_u( n_s ) = infinity
             IF ( vname ) VNAME_lpqp( n_s ) = 'S' // LPQP_char( i )
             n_s = n_s + 2 ; n_c = n_c + 1
           ELSE
             prob%G( n_c ) = zero
             prob%X( n_c ) = prob%C( i ) - cl
             prob%X_l( n_c ) = - infinity
             prob%X( n_s ) = MAX( prob%X( n_s ), ABS( prob%X( n_c ) ) )
           END IF

!  constraint bounded on both sides

         ELSE IF ( cl > - control%infinity ) THEN
           IF ( cu < control%infinity ) THEN
             IF ( cl <= cl ) THEN
               n_c = n_c + 1 ; la = la + 1
               prob%A%row( la ) = i
               prob%A%col( la ) = n_c
               prob%A%val( la ) = - one
               prob%C_l( i ) = zero
               prob%C_u( i ) = zero

               n_r = n_r + 1 ; la = la + 1
               prob%A%row( la ) = n_r
               prob%A%col( la ) = n_s
               prob%A%val( la ) = one
               la = la + 1
               prob%A%row( la ) = n_r
               prob%A%col( la ) = n_c
               prob%A%val( la ) = one
               prob%C_l( n_r ) = cl
               prob%C_u( n_r ) = infinity
               IF ( cname ) CNAME_lpqp( n_r ) = 'L' // LPQP_char( i )

               data%m_b = data%m_b + 1
               data%BOTH( 1, data%m_b ) = i
               data%BOTH( 2, data%m_b ) = n_r

               n_r = n_r + 1 ; la = la + 1
               prob%A%row( la ) = n_r
               prob%A%col( la ) = n_s
               prob%A%val( la ) = - one
               la = la + 1
               prob%A%row( la ) = n_r
               prob%A%col( la ) = n_c
               prob%A%val( la ) = one
               prob%C_l( n_r ) = - infinity
               prob%C_u( n_r ) = cu
               IF ( cname ) CNAME_lpqp( n_r ) = 'U' // LPQP_char( i )

               prob%G( n_c ) = zero
               prob%X( n_c ) = prob%C( i ) - cl
               prob%X_l( n_c ) = - infinity
               prob%X_u( n_c ) = infinity
               IF ( vname ) VNAME_lpqp( n_c ) = 'C' // LPQP_char( i )

               IF ( one_norm ) THEN
                 prob%G( n_s ) = rho
                 prob%X( n_s ) = MAX( zero, - prob%X( n_c ),                   &
                                      prob%X( n_c ) + cl - cu )
                 prob%X_l( n_s ) = zero
                 prob%X_u( n_s ) = infinity
                 IF ( vname ) VNAME_lpqp( n_s ) = 'S' // LPQP_char( i )
                 n_s = n_s + 2 ; n_c = n_c + 1
               ELSE
                 prob%X( n_s ) = MAX( prob%X( n_s ), - prob%X( n_c ),          &
                                      prob%X( n_c ) + cl - cu )
               END IF
             END IF

!  constraint bounded from below

           ELSE
             la = la + 1
             prob%A%row( la ) = i
             prob%A%col( la ) = n_s
             prob%A%val( la ) = one
             prob%C_l( i ) = cl
             prob%C_u( i ) = cu

             IF ( one_norm ) THEN
               prob%G( n_s ) = rho
               prob%X( n_s ) = MAX( zero, cl - prob%C( i ) )
               prob%X_l( n_s ) = zero
               prob%X_u( n_s ) = infinity
               IF ( vname ) VNAME_lpqp( n_s ) = 'S' // LPQP_char( i )
               n_s = n_s + 1 ; n_c = n_c + 1
             ELSE
               prob%X( n_s ) = MAX( prob%X( n_s ), cl - prob%C( i ) )
             END IF
           END IF

!  constraint bounded from above

         ELSE
           IF ( cu < control%infinity ) THEN
             la = la + 1
             prob%A%row( la ) = i
             prob%A%col( la ) = n_s
             prob%A%val( la ) = - one
             prob%C_l( i ) = cl
             prob%C_u( i ) = cu

             IF ( one_norm ) THEN
               prob%G( n_s ) = rho
               prob%X( n_s ) = MAX( zero, prob%C( i ) - cu )
               prob%X_l( n_s ) = zero
               prob%X_u( n_s ) = infinity
               IF ( vname ) VNAME_lpqp( n_s ) = 'S' // LPQP_char( i )
               n_s = n_s + 1 ; n_c = n_c + 1
             ELSE
               prob%X( n_s ) = MAX( prob%X( n_s ), prob%C( i ) - cu )
             END IF

!  free constraint

           ELSE
           END IF
         END IF
       END DO

!  write the l_p constraints in dense output format
!  ------------------------------------------------

!    ELSE IF ( data%a_output_format == a_output_dense ) THEN

!  write the l_p constraints in sparse row output format
!  ------------------------------------------------------

     ELSE

!  original dense storage

       IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
         l = 1
         DO i = 1, m_orig
           prob%A%ptr( i ) = l
           DO j = 1, n_orig
             prob%A%col( l ) = j
             l = l + 1
           END DO
         END DO
         prob%A%ptr( m_orig + 1 ) = l

!  original row-wise storage

       ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN

!  original co-ordinate storage

       ELSE
         liw = MAX( m_orig, n_orig ) + 1
         reallocate = .TRUE.
         IF ( ALLOCATED( data%IW ) ) THEN
           IF ( SIZE( data%IW ) < liw ) THEN
             DEALLOCATE( data%IW ) ; ELSE ; reallocate = .FALSE.
           END IF
         END IF
         IF ( reallocate ) THEN
           ALLOCATE( data%IW( liw ), STAT = inform%alloc_status )
           IF ( inform%alloc_status /= 0 ) THEN
             inform%bad_alloc = 'data%IW' ; GO TO 900
           END IF
         END IF

         IF ( m_orig > 0 ) THEN
           CALL SORT_reorder_by_rows( m_orig, n_orig, a_ne_orig,               &
                                    prob%A%row( : a_ne_orig ),                 &
                                    prob%A%col( : a_ne_orig ),  a_ne_orig,     &
                                    prob%A%val( : a_ne_orig ),                 &
                                    prob%A%ptr( : m_orig + 1 ), m_orig + 1,    &
                                    data%IW( : liw ), liw, error,              &
                                    out, inform%status )
           IF ( inform%status > 0 ) THEN
             IF ( printe ) WRITE( error, "( A, ' SORT error: status = ', I0)") &
               prefix, inform%status
             inform%status = GALAHAD_error_sort ; GO TO 800
           END IF
         END IF
       END IF

       IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
       CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', alloc_status )

!  compute the constraint residuals

      DO i = 1, m_orig
        prob%C( i ) = zero
        DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          prob%C( i ) = prob%C( i ) + prob%A%val( l ) * prob%X( prob%A%col( l ))
        END DO
      END DO

!  write the l_p Jacobian in row format

       mm = m + 1 ; ll = a_ne + 1
       n_s = n
       n_c = n_s - 1
!      n_c = n
       IF ( .NOT. one_norm .AND. m_orig > 0 ) prob%X( n_s ) = zero
!      IF ( one_norm ) THEN
!        n_s = n_c - 1
!      ELSE
!        n_s = n_orig + 1
!      END IF
       infinity = ten * control%infinity

!  march backwards filling in the Jacobian as we go

       data%m_b = 0
       DO i = m_orig, 1, - 1

         l1 = prob%A%ptr( i )
         l2 = prob%A%ptr( i + 1 ) - 1

         prob%A%ptr( i + 1 ) = la + 1

         cl = prob%c_l( i ) ; cu = prob%c_u( i )

!  equality constraint

         IF ( cu == cl ) THEN

           prob%A%val( la ) = one
           prob%A%col( la ) = n_s
           la = la - 1
           prob%A%val( la ) = - one
           prob%A%col( la ) = n_c
           prob%C_l( i ) = cl
           prob%C_u( i ) = cu
           la = la - 1

           prob%X_u( n_c ) = infinity
           IF ( vname ) VNAME_lpqp( n_c ) = 'C' // LPQP_char( i )
           IF ( one_norm ) THEN
             prob%G( n_c ) = rho
             prob%X( n_c ) = prob%C( i ) - cl
             prob%X_l( n_c ) = zero

             prob%G( n_s ) = rho
             prob%X( n_s ) = ABS( prob%X( n_c ) )
             prob%X_l( n_s ) = zero
             prob%X_u( n_s ) = infinity
             IF ( vname ) VNAME_lpqp( n_s ) = 'S' // LPQP_char( i )
             n_s = n_s - 2
             n_c = n_c - 2
           ELSE
             prob%G( n_c ) = zero
             prob%X( n_c ) = prob%C( i ) - cl
             prob%X_l( n_c ) = - infinity
             prob%X( n_s ) = MAX( prob%X( n_s ), ABS( prob%X( n_c ) ) )
             n_c = n_c - 1
           END IF

!  constraint bounded on both sides

         ELSE IF ( cl > - control%infinity ) THEN
           IF ( cu < control%infinity ) THEN
             IF ( cl <= cl ) THEN
               prob%A%val( la ) = - one
               prob%A%col( la ) = n_c
               prob%C_l( i ) = zero
               prob%C_u( i ) = zero
               la = la - 1

               prob%A%ptr( mm ) = ll
               mm = mm - 1
               ll = ll - 1
               prob%A%val( ll ) = - one
               prob%A%col( ll ) = n_s
               prob%C_l( mm ) = - infinity
               ll = ll - 1
               prob%A%val( ll ) = one
               prob%A%col( ll ) = n_c
               prob%C_u( mm ) = cu
               IF ( cname ) CNAME_lpqp( mm ) = 'U' // LPQP_char( i )

               prob%A%ptr( mm ) = ll
               mm = mm - 1
               ll = ll - 1
               prob%A%val( ll ) = one
               prob%A%col( ll ) = n_s
               ll = ll - 1
               prob%A%val( ll ) = one
               prob%A%col( ll ) = n_c
               prob%C_l( mm ) = cl
               prob%C_u( mm ) = infinity
               IF ( cname ) CNAME_lpqp( mm ) = 'L' // LPQP_char( i )

               data%m_b = data%m_b + 1
               data%BOTH( 1, data%m_b ) = i
               data%BOTH( 2, data%m_b ) = mm

               prob%G( n_c ) = zero
               prob%X( n_c ) = prob%C( i ) - cl
               prob%X_l( n_c ) = - infinity
               prob%X_u( n_c ) = infinity
               IF ( vname ) VNAME_lpqp( n_c ) = 'C' // LPQP_char( i )
               IF ( one_norm ) THEN
                 prob%G( n_s ) = rho
                 prob%X( n_s ) = MAX( zero, - prob%X( n_c ),                   &
                                      prob%X( n_c ) + cl - cu )
                 prob%X_l( n_s ) = zero
                 prob%X_u( n_s ) = infinity
                 IF ( vname ) VNAME_lpqp( n_s ) = 'S' // LPQP_char( i )
                 n_s = n_s - 2
                 n_c = n_c - 2
               ELSE
                 prob%X( n_s ) = MAX( prob%X( n_s ), - prob%X( n_c ),          &
                                      prob%X( n_c ) + cl - cu )
                 n_c = n_c - 1
               END IF
             END IF

!  constraint bounded from below

           ELSE
             prob%A%val( la ) = one
             prob%A%col( la ) = n_s
             prob%C_l( i ) = cl
             prob%C_u( i ) = cu
             la = la - 1

             IF ( one_norm ) THEN
               prob%G( n_s ) = rho
               prob%X( n_s ) = MAX( zero, cl - prob%C( i ) )
               prob%X_l( n_s ) = zero
               prob%X_u( n_s ) = infinity
               IF ( vname ) VNAME_lpqp( n_s ) = 'S' // LPQP_char( i )
               n_s = n_s - 1
               n_c = n_c - 1
             ELSE
               prob%X( n_s ) = MAX( prob%X( n_s ), cl - prob%C( i ) )
             END IF
           END IF

!  constraint bounded from above

         ELSE
           IF ( cu < control%infinity ) THEN
             prob%A%val( la ) = - one
             prob%A%col( la ) = n_s
             prob%C_l( i ) = cl
             prob%C_u( i ) = cu
             la = la - 1

             IF ( one_norm ) THEN
               prob%G( n_s ) = rho
               prob%X( n_s ) = MAX( zero, prob%C( i ) - cu )
               prob%X_l( n_s ) = zero
               prob%X_u( n_s ) = infinity
               IF ( vname ) VNAME_lpqp( n_s ) = 'S' // LPQP_char( i )
               n_s = n_s - 1
               n_c = n_c - 1
             ELSE
               prob%X( n_s ) = MAX( prob%X( n_s ), prob%C( i ) - cu )
             END IF

!  free constraint

           ELSE
           END IF
         END IF

         IF ( i > 1 ) THEN
           DO l = l2, l1, - 1
             prob%A%val( la ) = prob%A%val( l )
             prob%A%col( la ) = prob%A%col( l )
             la = la - 1
           END DO
         END IF
       END DO
     END IF

     IF ( .NOT. one_norm .AND. m_orig > 0 ) THEN
       prob%G( n_s ) = rho
       prob%X_l( n_s ) = zero
       prob%X_u( n_s ) = infinity
       IF ( vname ) VNAME_lpqp( n_s ) = 'S'
     END IF

!  give the extra variables initial values

!    prob%X( n_orig + 1 : n ) = zero
!    WRITE(6, "( ( 5ES12.4 ) )" ) prob%X( n_orig + 1 : n )
     prob%Y( m_orig + 1 : m ) = zero
     prob%Z( n_orig + 1 : n ) = zero
     IF ( PRESENT( B_stat ) .AND. lcold ) B_stat( n_orig + 1 : n ) = 0
     IF ( PRESENT( C_stat ) .AND. lcold ) C_stat( m_orig + 1 : m ) = 0

     prob%new_problem_structure = .TRUE.
     data%gradient_kind = prob%gradient_kind
     data%Hessian_kind = prob%Hessian_kind
     prob%gradient_kind = 2
     inform%status = GALAHAD_ok

 800 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time = time_now - time_start
     inform%clock_time = clock_now - clock_start

     IF ( printd ) WRITE( out, "( A, ' leaving LPQP_formulate ' )" ) prefix

     RETURN

!  allocation error

 900 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time = time_now - time_start
     inform%clock_time = clock_now - clock_start

     IF ( printi ) WRITE( out,                                                 &
       "( A, ' ** Message from -LPQP_formulate-', /,                           &
      &   A, ' Allocation error, for ', A, ' status = ', I0 )" )               &
         prefix, prefix, inform%bad_alloc, inform%alloc_status
     IF ( printd ) WRITE( out, "( A, ' leaving LPQP_formulate ' )" ) prefix

     RETURN

     CONTAINS

       FUNCTION LPQP_char( i )

!  Internal function to convert the integer i to a left-shifted character

       CHARACTER ( LEN = 9 ) :: LPQP_char
       INTEGER, INTENT( IN ) :: i
       IF ( i <= 9 ) THEN
         WRITE( LPQP_char, "( I1 )" ) i
       ELSE IF ( i <= 99 ) THEN
         WRITE( LPQP_char, "( I2 )" ) i
       ELSE IF ( i <= 999 ) THEN
         WRITE( LPQP_char, "( I3 )" ) i
       ELSE IF ( i <= 9999 ) THEN
         WRITE( LPQP_char, "( I4 )" ) i
       ELSE IF ( i <= 99999 ) THEN
         WRITE( LPQP_char, "( I5 )" ) i
       ELSE IF ( i <= 999999 ) THEN
         WRITE( LPQP_char, "( I6 )" ) i
       ELSE IF ( i <= 9999999 ) THEN
         WRITE( LPQP_char, "( I7 )" ) i
       ELSE IF ( i <= 99999999 ) THEN
         WRITE( LPQP_char, "( I8 )" ) i
       ELSE
         WRITE( LPQP_char, "( I9 )" ) i
       END IF

!  End of LPQP_char

       END FUNCTION LPQP_char

       FUNCTION LPQP_real_required( nnz_orig, nnz, W, W_temp )

!  If W is too small to accomodate nnz entries, find the required size of the
!  temporary workspace array W_temp needed to hold existing entries of W

       INTEGER :: LPQP_real_required
       INTEGER, INTENT( IN ) :: nnz_orig, nnz
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W, W_temp

       LPQP_real_required = 0
       IF ( ALLOCATED( W ) ) THEN
         IF ( SIZE( W ) < nnz ) THEN
           IF ( ALLOCATED( W_temp ) ) THEN
             IF ( SIZE( W_temp ) < nnz_orig ) LPQP_real_required = nnz_orig
           ELSE
             LPQP_real_required = nnz_orig
           END IF
         END IF
       END IF

!  End of LPQP_real_required

       END FUNCTION LPQP_real_required

       FUNCTION LPQP_integer_required( nnz_orig, nnz, W, W_temp )

!  If W is too small to accomodate nnz entries, find the required size of the
!  temporary workspace array W_temp needed to hold existing entries of W

       INTEGER :: LPQP_integer_required
       INTEGER, INTENT( IN ) :: nnz_orig, nnz
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: W, W_temp

       LPQP_integer_required = 0
       IF ( ALLOCATED( W ) ) THEN
         IF ( SIZE( W ) < nnz ) THEN
           IF ( ALLOCATED( W_temp ) ) THEN
             IF ( SIZE( W_temp ) < nnz_orig ) LPQP_integer_required = nnz_orig
           ELSE
             LPQP_integer_required = nnz_orig
           END IF
         END IF
       END IF

!  End of LPQP_integer_required

       END FUNCTION LPQP_integer_required

       SUBROUTINE LPQP_real_reallocate( nnz_orig, nnz, W, W_temp, status,      &
                                        alloc_status, array_name, bad_alloc )

!  If W is too small to accomodate nnz entries, reallocate it
!  so that it is, while maintaining the first nnz_orig entries.

       INTEGER, INTENT( IN ) :: nnz_orig, nnz
       INTEGER, INTENT( OUT ) :: status, alloc_status
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
       REAL ( KIND = wp ), DIMENSION( : ) :: W_temp
       CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
       CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

       status = GALAHAD_ok ; alloc_status = 0
       IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
       IF ( SIZE( W ) < nnz ) THEN
         W_temp( : nnz_orig ) = W( : nnz_orig )
         CALL SPACE_dealloc_array( W, status, alloc_status,                    &
                                   array_name, bad_alloc, out )
         IF ( alloc_status /= 0 ) THEN
           status = GALAHAD_error_deallocate ; RETURN ; END IF
         ALLOCATE( W( nnz ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN
           status = GALAHAD_error_allocate ;
           IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )             &
             bad_alloc = array_name
         END IF
         W( : nnz_orig ) = W_temp( : nnz_orig )
       END IF

       RETURN

!  End of LPQP_real_reallocate

       END SUBROUTINE LPQP_real_reallocate

       SUBROUTINE LPQP_integer_reallocate( nnz_orig, nnz, IW, IW_temp, status, &
                                           alloc_status, array_name, bad_alloc )

!  If IW is too small to accomodate nnz entries, reallocate it
!  so that it is, while maintaining the first nnz_orig entries.

       INTEGER, INTENT( IN ) :: nnz_orig, nnz
       INTEGER, INTENT( OUT ) :: status, alloc_status
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
       INTEGER, DIMENSION( : ) :: IW_temp
       CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
       CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

       status = GALAHAD_ok ; alloc_status = 0
       IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
       IF ( SIZE( IW ) < nnz ) THEN
         IW_temp( : nnz_orig ) = IW( : nnz_orig )
         CALL SPACE_dealloc_array( IW, status, alloc_status,                   &
                                   array_name, bad_alloc, out )
         IF ( alloc_status /= 0 ) THEN
           status = GALAHAD_error_deallocate ; RETURN ; END IF
         ALLOCATE( IW( nnz ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN
           status = GALAHAD_error_allocate ;
           IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )             &
             bad_alloc = array_name
         END IF
         IW( : nnz_orig ) = IW_temp( : nnz_orig )
       END IF

       RETURN

!  End of LPQP_integer_reallocate

       END SUBROUTINE LPQP_integer_reallocate

       SUBROUTINE LPQP_integer_reallocate_cold( nnz_orig, nnz, IW, cold,       &
                         IW_temp, status, alloc_status, array_name, bad_alloc )

!  If IW is too small to accomodate nnz entries, reallocate it
!  so that it is, while maintaining the first nnz_orig entries if cold=.TRUE.

       INTEGER, INTENT( IN ) :: nnz_orig, nnz
       INTEGER, INTENT( OUT ) :: status, alloc_status
       LOGICAL, INTENT( IN ) :: cold
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
       INTEGER, DIMENSION( : ) :: IW_temp
       CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
       CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

       status = GALAHAD_ok ; alloc_status = 0
       IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
       IF ( SIZE( IW ) < nnz ) THEN
         IF ( cold ) IW_temp( : nnz_orig ) = IW( : nnz_orig )
         CALL SPACE_dealloc_array( IW, status, alloc_status,                   &
                                   array_name, bad_alloc, out )
         IF ( alloc_status /= 0 ) THEN
           status = GALAHAD_error_deallocate ; RETURN ; END IF
         ALLOCATE( IW( nnz ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN
           status = GALAHAD_error_allocate ;
           IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )             &
             bad_alloc = array_name
         END IF
         IF ( cold ) IW( : nnz_orig ) = IW_temp( : nnz_orig )
       END IF

       RETURN

!  End of LPQP_integer_reallocate_cold

       END SUBROUTINE LPQP_integer_reallocate_cold

!  End of LPQP_formulate

     END SUBROUTINE LPQP_formulate

!-*-*-*-*-*-*-*-   L P Q P _ R E S T O R E   S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE LPQP_restore( prob, data )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Restore the quadratic program from the equivalent l_p penalty function
!  version
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
     TYPE ( LPQP_data_type ), INTENT( IN ) :: data

!  Local variables

     INTEGER :: i, j, k, l, alloc_status

!  Restore H

     IF ( data%Hessian_kind < 0 ) THEN
       CALL SMT_put( prob%H%type, SMT_get( data%H_type ), alloc_status )

!  only needed if H has been transformed to row storage

       IF ( data%h_output_format == h_output_sparse_by_rows ) THEN

!  Original co-ordinate storage

         IF ( SMT_get( prob%H%type ) == 'COORDINATE' ) THEN
           DO i = 1, data%n
             DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
               prob%H%row( l ) = i
             END DO
           END DO
         END IF
       END IF
     END IF

!  Restore A

     CALL SMT_put( prob%A%type, SMT_get( data%A_type ), alloc_status )

!  only needed if A has been transformed to row storage

     IF ( data%a_output_format == a_output_sparse_by_rows ) THEN

!  Original dense storage

       IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
         k = 1
         DO i = 1, data%m
           DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
             IF ( prob%A%col( l ) <= data%n ) THEN
               prob%A%val( k ) = prob%A%val( l )
               k = k + 1
             END IF
           END DO
         END DO

!  Original row-wise storage

       ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
         k = 1
         DO i = 1, data%m
           j = prob%A%ptr( i )
           prob%A%ptr( i ) = k
           DO l = j, prob%A%ptr( i + 1 ) - 1
             IF ( prob%A%col( l ) <= data%n ) THEN
               prob%A%col( k ) = prob%A%col( l )
               prob%A%val( k ) = prob%A%val( l )
               k = k + 1
             END IF
           END DO
         END DO
         prob%A%ptr( data%m + 1 ) = k

!  Original co-ordinate storage

       ELSE IF ( SMT_get( prob%A%type ) == 'COORDINATE' ) THEN
         k = 1
         DO i = 1, data%m
           DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
             IF ( prob%A%col( l ) <= data%n ) THEN
               prob%A%row( k ) = i
               prob%A%col( k ) = prob%A%col( l )
               prob%A%val( k ) = prob%A%val( l )
               k = k + 1

!  Remove the contributions from the slacks from the constraint values

             ELSE
               prob%C( i ) =                                                   &
                 prob%C( i ) - prob%A%val( l ) * prob%X( prob%A%col( l ) )
             END IF
           END DO
         END DO
       END IF
     END IF

!  Restore the constraint bounds

     DO l = 1, data%m_b
       i = data%BOTH( 1, data%m_b )
       k = data%BOTH( 2, data%m_b )
       prob%C_l( i ) =  prob%C_l( k )
       prob%C_u( i ) =  prob%C_u( k + 1 )
     END DO

!  Restore the problem dimensions

     prob%n = data%n
     prob%m = data%m
     prob%A%ne = data%A_ne
     prob%H%ne = data%H_ne
     prob%gradient_kind = data%gradient_kind
     prob%Hessian_kind = data%Hessian_kind

!  remove gradient and function terms from original least-distance format

     IF ( prob%Hessian_kind == 1 ) THEN
       IF ( prob%target_kind == 1 ) THEN
         prob%G( : prob%n ) = prob%G( : prob%n ) + one
         prob%f = prob%f - half * REAL( prob%n, wp )
       ELSE IF ( prob%target_kind /= 0 ) THEN
         prob%G( : prob%n ) = prob%G( : prob%n ) + prob%X0( : prob%n )
         prob%f = prob%f - half * SUM( prob%X0( : prob%n ) ** 2 )
       END IF
     ELSE IF ( prob%Hessian_kind == 2 ) THEN
       IF ( prob%target_kind == 1 ) THEN
         prob%G( : prob%n ) = prob%G( : prob%n )                               &
           + prob%WEIGHT( : prob%n ) ** 2
         prob%f = prob%f - half * SUM( prob%WEIGHT( : prob%n ) ** 2 )
       ELSE IF ( prob%target_kind /= 0 ) THEN
         prob%G( : prob%n ) = prob%G( : prob%n )                               &
           + prob%X0( : prob%n ) * prob%WEIGHT( : prob%n ) ** 2
         prob%f = prob%f - half                                                &
           * SUM( ( prob%X0( : prob%n ) * prob%WEIGHT( : prob%n ) ) ** 2 )
       END IF
     END IF

!  End of LPQP_restore

     END SUBROUTINE LPQP_restore

!-*-*-*-*-*-*-   L P Q P _ F I N A L I Z E   S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE LPQP_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    see Subroutine LPQP_initialize
!   control see Subroutine LPQP_initialize
!   inform  see Subroutine LPQP_formulate

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( LPQP_data_type ), INTENT( INOUT ) :: data
     TYPE ( LPQP_control_type ), INTENT( IN ) :: control
     TYPE ( LPQP_inform_type ), INTENT( INOUT ) :: inform

!  Deallocate all allocated arrays

     IF ( ALLOCATED( data%BOTH ) ) THEN
       DEALLOCATE( data%BOTH, STAT = inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_deallocate
         IF ( control%error > 0 ) WRITE( control%error, 2900 )                 &
           'data%BOTH', inform%alloc_status
       END IF
     END IF

     IF ( ALLOCATED( data%IW ) ) THEN
       DEALLOCATE( data%IW, STAT = inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_deallocate
         IF ( control%error > 0 ) WRITE( control%error, 2900 )                 &
           'data%IW', inform%alloc_status
       END IF
     END IF

     IF ( ALLOCATED( data%W ) ) THEN
       DEALLOCATE( data%W, STAT = inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_deallocate
         IF ( control%error > 0 ) WRITE( control%error, 2900 )                 &
           'data%W', inform%alloc_status
       END IF
     END IF

     RETURN

!  Non-executable statement

 2900 FORMAT( ' ** Message from -LPQP_terminate-', /,                          &
              ' Deallocation error, for ', A, /, ' status = ', I0 )

!  End of subroutine LPQP_terminate

     END SUBROUTINE LPQP_terminate

!  End of module LPQP

   END MODULE GALAHAD_LPQP_double
