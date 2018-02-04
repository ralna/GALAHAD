! THIS VERSION: GALAHAD 2.4 - 10/01/2011 AT 13:30 GMT

!-*-*-*-*-*-*-*-*-  G A L A H A D _ S C A L E    M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History - released, pre GALAHAD 1.0, September 2nd 1999 as SCALING 
!  Updated for GALAHAD 2.4 January 10th, 2011 by adding parts of old QTRANS

   MODULE GALAHAD_SCALE_double

!  -------------------------------------------------------------------------
!  | Compute and apply suitable shifts (x_s,f_s) and scale factors         |
!  | (X_s,F_s,C_s) for the [possibly parametric] quadratic program (QP)    !
!  |                                                                       |
!  |   minimize   1/2 x^T H x + x^T g + f  [+ theta x^T dg + f + theta df] |
!  |   subjec to  c_l [+ theta dc_l] <= A x <= c_u [+ theta dc_u],         |
!  |   and        x_l [+ theta dx_l] <=  x  <= x_u [+ theta dx_u]          |
!  |                                                                       |
!  |   so that  x_t = X_s^-1 ( x - x_s )                                   |
!  |            f_t( x_t ) = F_s^-1 ( q( x ) - f_s )                       |
!  |   and      A_t x_t = C_s^-1 ( A x - c_s )                             |
!  -------------------------------------------------------------------------

      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SPECFILE_double 
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_TRANS_double, only :                                         &
        SCALE_trans_type => TRANS_trans_type,                                  &
        TRANS_terminate, TRANS_default,                                        &
        TRANS_v_trans_inplace, TRANS_v_untrans_inplace

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SCALE_initialize, SCALE_read_specfile, SCALE_terminate,        &
                SCALE_get, SCALE_apply, SCALE_recover, SCALE_trans_type,       &
                QPT_problem_type

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
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

      TYPE, PUBLIC :: SCALE_control_type

!   error and warning diagnostics occur on stream error 
   
        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   at most maxit inner iterations are allowed 

        INTEGER :: maxit = 100

!   try to shift x if shift_x > 0

        INTEGER :: shift_x = 1

!   try to scale x if scale_x > 0

        INTEGER :: scale_x = 1

!   try to shift c if shift_c > 0

        INTEGER :: shift_c = 1

!   try to scale c if scale_c > 0. If scale_c is 1, try to scale to ensure
!    O(1) changes to x make O(1) changes to c, using the (scaled) infinity 
!    norms of the gradients of the constraints. If scale_c > 1 try to scale
!    to make c = O(1)

        INTEGER :: scale_c = 1

!   try to shift f if shift_f > 0

        INTEGER :: shift_f = 1

!   try to scale f if scale_f > 0. If scale_f is 1, try to scale to ensure
!    O(1) changes to x make O(1) changes to f, using the (scaled) infinity 
!    norms of the gradients of the objective. If scale_f > 1 try to scale 
!    to make f = O(1)

        INTEGER :: scale_f = 1

!   any bound larger than infinity in modulus will be regarded as infinite 

        REAL ( KIND = wp ) :: infinity = ten ** 19

!  the scaling iteration is stopped as soon as a scaled residual is smaller
!    than n * stop_tol 

        REAL ( KIND = wp ) :: stop_tol = 0.1_wp

!  the minimum permitted x scale factor

        REAL ( KIND = wp ) :: scale_x_min = one

!  the minimum permitted c scale factor

        REAL ( KIND = wp ) :: scale_c_min = one

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

      TYPE, PUBLIC :: SCALE_inform_type

!  return status. See QP_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the number of iterations (matrix-vector products) required

        INTEGER :: iter = - 1

!  the deviation from double-stocasticity when appropriate

        REAL ( KIND = wp ) :: deviation = - one

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

      END TYPE

!  ...................
!   data derived type 
!  ...................

     TYPE, PUBLIC :: SCALE_data_type
       PRIVATE 
       INTEGER :: scale = 0
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ROW_val
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: COL_val
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PROD
     END TYPE

   CONTAINS

!-*-*-*-*-*-*-*-*-*-*  S C A L E _ i n i t i a l i z e   *-*-*-*-*-*-*-*-*-

     SUBROUTINE SCALE_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for SCALE. This routine should be called before 
!  other SCALE subprograms
! 
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

     TYPE ( SCALE_data_type ), INTENT( INOUT ) :: data
     TYPE ( SCALE_control_type ), INTENT( OUT ) :: control        
     TYPE ( SCALE_inform_type ), INTENT( OUT ) :: inform

     data%scale = 0
     inform%status = GALAHAD_ok

     RETURN

!  End of SCALE_initialize

     END SUBROUTINE SCALE_initialize

!-*-*-*-   S C A L E _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

     SUBROUTINE SCALE_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by SCALE_initialize could (roughly) 
!  have been set as:

! BEGIN SCALE SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  maximum-number-of-iterations                      100
!  shift-x                                           1
!  scale-x                                           1
!  shift-c                                           1
!  scale-c                                           1
!  shift-f                                           1
!  scale-f                                           1
!  infinity-value                                    1.0D+19
!  stop-tolerance                                    0.1
!  smallest-x-scaling                                1.0
!  smallest-c-scaling                                1.0
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END SCALE SPECIFICATIONS (DEFAULT)

!  Dummy arguments

     TYPE ( SCALE_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: maxit = print_level + 1
     INTEGER, PARAMETER :: shift_x = maxit + 1
     INTEGER, PARAMETER :: scale_x = shift_x + 1
     INTEGER, PARAMETER :: shift_c = scale_x + 1
     INTEGER, PARAMETER :: scale_c = shift_c + 1
     INTEGER, PARAMETER :: shift_f = scale_c + 1
     INTEGER, PARAMETER :: scale_f = shift_f + 1
     INTEGER, PARAMETER :: infinity = scale_f + 1
     INTEGER, PARAMETER :: stop_tol = infinity + 1
     INTEGER, PARAMETER :: scale_x_min = stop_tol + 1
     INTEGER, PARAMETER :: scale_c_min = scale_x_min + 1
     INTEGER, PARAMETER :: space_critical = scale_c_min + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 5 ), PARAMETER :: specname = 'SCALE'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level' 
     spec( maxit )%keyword = 'maximum-number-of-iterations' 
     spec( shift_x )%keyword = 'shift-x'
     spec( scale_x )%keyword = 'scale-x'
     spec( shift_c )%keyword = 'shift-c'
     spec( scale_c )%keyword = 'scale-c'
     spec( shift_f )%keyword = 'shift-f'
     spec( scale_f )%keyword = 'scale-f'

!  Real key-words

     spec( infinity )%keyword = 'infinity-value'
     spec( stop_tol )%keyword = 'stop-tolerance'
     spec( scale_x_min )%keyword = 'smallest-x-scaling'
     spec( scale_c_min )%keyword = 'smallest-c-scaling'

!  Logical key-words

     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

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
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( shift_x ),                              &
                                 control%shift_x,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_x ),                              &
                                 control%scale_x,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( shift_c ),                              &
                                 control%shift_c,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_c ),                              &
                                 control%scale_c,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( shift_f ),                              &
                                 control%shift_f,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_f ),                              &
                                 control%scale_f,                              &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_tol ),                             &
                                 control%stop_tol,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_x_min ),                          &
                                 control%scale_x_min,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_c_min ),                          &
                                 control%scale_c_min,                          &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
!  Set character values

     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

     RETURN

!  End of SCALE_read_specfile

     END SUBROUTINE SCALE_read_specfile

!-*-*-*-*-*-*-*-*-   S C A L E _ g e t  S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE SCALE_get( prob, scale, trans, data, control, inform )

!  ---------------------------------------------------------------------------
!
!  Suppose that x_t = X_s^-1 ( x - x_s )
!               f_t( x_t ) = F_s^-1 ( q( x ) - f_s )
!          and  A_t x_t = C_s^-1 ( A x - c_s )
!
!  Compute suitable shifts (x_s,f_s) and scale factors (X_s,F_s,C_s)
!  for the quadratic programming (QP) problem
!
!      min f(x) = 1/2 x^T H x + x^T g + f
!
!      s.t.        c_l <= A x <= c_u, 
!      and         x_l <=  x  <= x_u
!
!  (or optionally to the parametric problem
!
!      min  1/2 x^T H x + x^T g + theta x^T dg + f + theta df
!
!      s.t. c_l + theta dc_l <= A x <= c_u + theta dc_u,
!      and  x_l + theta dx_l <=  x  <= x_u + theta dx_u )
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold 
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %n is an INTEGER variable, that must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!                 
!   %m is an INTEGER variable, that must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!                 
!   %gradient_kind is an INTEGER variable that defines the type of linear
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
!   %Hessian_kind is an INTEGER variable that defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero, and the analytic centre of the 
!        feasible region will be found. %WEIGHT (see below) need not be set
!
!     1  all the weights will be one. %WEIGHT (see below) need not be set
!
!     2  the weights will be those given by %WEIGHT (see below)
!
!    <0  the Hessian H will be used 
!
!   %H is a structure of type SMT_type used to hold the LOWER TRIANGULAR part 
!    of H. Four storage formats are permitted:
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
!   %WEIGHT is a REAL array, that need only be set if %Hessian_kind is larger
!    than 1. If this is so, it must be of length at least %n, and contain the
!    weights W for the objective function. 
!  
!   %X0 is a REAL array, that need only be set if %Hessian_kind is not 1 or 2.
!    If this is so, it must be of length at least %n, and contain the
!    weights X^0 for the objective function. 
!  
!   %G is a REAL array, that need only be set if %gradient_kind is not 0 
!    or 1. If this is so, it must be of length at least %n, and contain the
!    linear terms g for the objective function. 
!  
!   %f is a REAL variable, that must be set by the user to the value of
!    the constant term f in the objective function. 
!
!   %A is a structure of type SMT_type used to hold the matrix A. 
!    Three storage formats are permitted:
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
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 5 ) = TRANSFER( 'DENSE', A%type )
!       A%val( : )   the values of the components of A, stored row by row,
!                    with each the entries in each row in order of 
!                    increasing column indicies.
!
!   %C is a REAL array of length %m, that must have been set by the user
!    to the values Ax
!
!   %X is a REAL array of length %n, that must be set by the user
!    to estimaes of the solution, x
!
!   %C_l, %C_u are REAL arrays of length %n, that must be set by the user
!    to the values of the arrays c_l and c_u of lower and upper bounds on A x.
!    Any bound c_l_i or c_u_i larger than or equal to control%infinity in 
!    absolute value will be regarded as being infinite (see the entry 
!    control%infinity). Thus, an infinite lower bound may be specified by 
!    setting the appropriate component of %C_l to a value smaller than 
!    -control%infinity, while an infinite upper bound can be specified by 
!    setting the appropriate element of %C_u to a value larger than 
!    control%infinity. 
!   
!   %X_l, %X_u are REAL arrays of length %n, that must be set by the user
!    to the values of the arrays x_l and x_u of lower and upper bounds on x.
!    Any bound x_l_i or x_u_i larger than or equal to control%infinity in 
!    absolute value will be regarded as being infinite (see the entry 
!    control%infinity). Thus, an infinite lower bound may be specified by 
!    setting the appropriate component of %X_l to a value smaller than 
!    -control%infinity, while an infinite upper bound can be specified by 
!    setting the appropriate element of %X_u to a value larger than 
!    control%infinity. 
!   
!   %Y is a REAL array of length %m, that must be set by the user to
!    appropriate estimates of the values of the Lagrange multipliers 
!    corresponding to the general constraints c_l <= A x <= c_u. 
!
!   %Z is a REAL array of length %n, that must be set by the user to
!    appropriate estimates of the values of the dual variables 
!    (Lagrange multipliers corresponding to the simple bound constraints 
!    x_l <= x <= x_u). 
!
!  scale is an INTEGER variable that defines the kind of scaling used.
!   Possible vales are:
!
!    <0  no scaling
!     1  scale to try to map all variables and constraints to [0,1]
!     2  normalize rows of K = ( H A(transpose) ) using Curtis and Reid'
!                              ( A      0       )  symmetric method
!     3  normalize rows & columns of A uing Curtis and Reid' unsymmetric method
!     4  normalize rows of A so that each has one-norm close to 1
!     5  normalize rows of K (cf 2) then normalize rows of A (cf 4)
!     6  normalize rows & columns of A (cf 3) then normalize rows of A (cf 4)
!     7  normalize rows & columns using Sinkhorn-Knopp equilibration
!    >7 (currently) no scaling
!
!  trans is a structure of type SCALE_trans_type that holds the shift and 
!   scale factors for the variables, and constraint and objective values
!   The following components may be set on exit:
!
!   %X_scale is a REAL array of length prob%n, that holds the variable
!     scale factors.
!
!   %X_shift is a REAL array of length prob%n, that holds the variable
!     shifts if appropriate.
!
!   %C_scale is a REAL array of length prob%m, that holds the constraint
!     scale factors.
!
!   %C_shift is a REAL array of length prob%m, that holds the constraint
!     shifts if appropriate.
!
!   %f_scale is a REAL variable, that holds the objective scale factor.
!
!   %f_shift is a REAL variable, that holds the objective scale factor
!     shift if appropriate.
!
!  data is a structure of type SCALE_data_type that holds private internal data.
!
!  control is a structure of type SCALE_control_type as defined in the preamble.
!
!  inform is a structure of type SCALE_inform_type as defined in the preamble.
!    The component status has possible values:
!  
!     0 Normal termination with a locally optimal solution.
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!   - 3 one of the restrictions 
!        prob%n     >=  1
!        prob%m     >=  0
!        prob%H%type in { 'DIAGONAL', 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!        prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated.
!
!  ---------------------------------------------------------------------------

!  Dummy arguments

      INTEGER, INTENT( IN ) :: scale
      TYPE ( QPT_problem_type ), INTENT( IN ) :: prob
      TYPE ( SCALE_trans_type ), INTENT( INOUT ) :: trans
      TYPE ( SCALE_data_type ), INTENT( INOUT ) :: data
      TYPE ( SCALE_control_type ), INTENT( IN ) :: control
      TYPE ( SCALE_inform_type ), INTENT( INOUT ) :: inform

!  local variables

      INTEGER :: out
      LOGICAL :: printi
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      out = control%out
      printi = out > 0 .AND. control%print_level > 0

      IF ( out > 0 .AND. control%print_level >= 5 )                            &
        WRITE( out, "( A, ' entering SCALE_get')") prefix

      SELECT CASE( scale )
      CASE ( 1 )
        data%scale = scale

!  use shift and scale

        IF ( printi ) WRITE( out, 2010 ) prefix
        CALL SCALE_get_shift_and_scale( prob%n, prob%m, prob%A, prob%G,        &
!                                       prob%f,                                &
                                        prob%X, prob%X_l, prob%X_u,            &
                                        prob%C, prob%C_l, prob%C_u,            &
                                        trans, control, inform )
      CASE ( 2 )
        data%scale = scale

!  scale using K

        IF ( printi ) WRITE( out, 2000 ) prefix, 'K'
        CALL SCALE_get_factors_from_K( prob%n, prob%m, prob%H, prob%A,         &
                                       trans, data, control, inform )
      CASE ( 3 )
        data%scale = scale

!  scale using A

        IF ( printi ) WRITE( out, 2000 ) prefix, 'A'
        CALL SCALE_get_factors_from_A( prob%n, prob%m, prob%A,                 &
                                       trans, data, control, inform )
      CASE ( 4 )
        data%scale = scale

!  allocate space for scale factors

        array_name = 'scale: X_scale'
        CALL SPACE_resize_array( prob%n, trans%X_scale, inform%status,         &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'scale: C_scale'
        CALL SPACE_resize_array( prob%m, trans%C_scale, inform%status,         &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  scale to equilibrate A

        IF ( printi ) WRITE( out, 2010 ) prefix
        trans%X_scale( : prob%n ) = one ; trans%C_scale( : prob%m ) = one
        CALL SCALE_normalize_rows_of_A( prob%n, prob%m, prob%A,                &
                                        trans, data, control, inform )
      CASE ( 5 )
        data%scale = scale

!  scale using K

        IF ( printi ) WRITE( out, 2000 ) prefix, 'K'
        CALL SCALE_get_factors_from_K( prob%n, prob%m, prob%H, prob%A,         &
                                       trans, data, control, inform )
        IF ( inform%status /= GALAHAD_ok ) RETURN

!  rescale to equilibrate A

        IF ( printi ) WRITE( out, 2010 ) prefix
        CALL SCALE_normalize_rows_of_A( prob%n, prob%m, prob%A,                &
                                        trans, data, control, inform )
      CASE ( 6 )
        data%scale = scale

!  scale using A

        IF ( printi ) WRITE( out, 2000 ) prefix, 'A'
        CALL SCALE_get_factors_from_A( prob%n, prob%m, prob%A,                 &
                                       trans, data, control, inform )
        IF ( inform%status /= GALAHAD_ok ) RETURN

!  rescale to equilibrate A

        IF ( printi ) WRITE( out, 2010 ) prefix
        CALL SCALE_normalize_rows_of_A( prob%n, prob%m, prob%A,                &
                                        trans, data, control, inform )
      CASE ( 7 )
        data%scale = scale

!  scale using A

        IF ( printi ) WRITE( out, "( /, A,  ' problem will be scaled',         &
       &     ' using Sinkhorn-Knopp equilibration' )" ) prefix
        CALL SCALE_get_sinkhorn_knopp( prob%n, prob%m, prob%H, prob%A,         &
                                       trans, data, control, inform )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      CASE DEFAULT

!  ignore anything else

      END SELECT

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving SCALE_get' )") prefix
      RETURN

!  allocation error

  900 CONTINUE 
      inform%status = GALAHAD_error_allocate
      IF ( control%out > 0 .AND. control%print_level > 0 ) WRITE( control%out, &
        "( A, ' ** Message from -SCALE_get-', /,  A,                           &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status
      RETURN  

!  non-executable statements

 2000 FORMAT( /, A, ' problem will be scaled based on ', A )
 2010 FORMAT( /, A, ' (further) equibration scaling will be applied to A')

!  End of SCALE_get

      END SUBROUTINE SCALE_get

!-*-*-*-*-*-*-*-   S C A L E _ a p p l y  S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE SCALE_apply( prob, trans, data, control, inform )

!  ----------------------------------------------------------------------------
!  Apply the shifts (x_s,f_s) and scale factors (X_s,F_s,C_s) computed
!  by SCALE_get to the data for the quadratic programming (QP) problem
!
!      min f(x) = 1/2 x^T H x + x^T g + f
!
!      s.t.        c_l <= A x <= c_u, 
!      and         x_l <=  x  <= x_u
!
!  (or optionally to the parametric problem
!
!      min  1/2 x^T H x + x^T g + theta x^T dg + f + theta df
!
!      s.t. c_l + theta dc_l <= A x <= c_u + theta dc_u,
!      and  x_l + theta dx_l <=  x  <= x_u + theta dx_u )
!
!  to derive the transformed problem
!
!      min f_t(x_t) = 1/2 x_t^T H_t x_t + x_t^T g_t + f_t
!
!      s.t.           c_t_l <= A_t x_t <= c_t_u, 
!                     x_t_l <=   x_t   <= x_t_u
!
!  (or optionally for the parametric problem
!
!      min  1/2 x_t^T H_t x_t + x_t^T g_t + theta x_t^T dg_t + f_t + theta df_t
!
!      s.t. c_t_l + theta dc_t_l <= A_t x_t <= c_t_u + theta dc_t_u,
!      and  x_t_l + theta dx_t_l <=    x_t  <= x_t_u + theta dx_t_u )
!
!  where H_t = X_s^T H X_s / F_s
!        g_t = X_s ( H x_s + g ) / F_s
!        dg_t = X_s dg / F_s
!        f_t = 1/2 x_s^T H x_s + x_s^T g + f - f_s ) / F_s
!        df_t = x_s^T dg / F_s
!        A_t = C_s^-1 A X_s
!        c_s = A x_s
!        c_t_l = C_s^-1 ( c_l - c_s )
!        dc_t_l = C_s^-1 dc_l
!        c_t_u = C_s^-1 ( c_u - c_s )
!        dc_t_u = C_s^-1 d_u
!        x_t_l = X_s^-1 ( c_l - x_s )
!        dx_t_l = X_s^-1 d_l
!        x_t_u = X_s^-1 ( c_u - x_s )
!  and   dx_t_u = X_s^-1 dc_u
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose values should
!   have been set as described in the preamble to SCALE_get. On exit,
!   the real data will have been transformed by the shift and scaling
!   factors calculated by SCALE_get.
!
!   The following additional components should be provided
!
!   %Y is a REAL array of length %m, that must be set by the user to
!    appropriate estimates of the values of the Lagrange multipliers 
!    corresponding to the general constraints c_l <= A x <= c_u. 
!
!   %Z is a REAL array of length %n, that must be set by the user to
!    appropriate estimates of the values of the dual variables 
!    (Lagrange multipliers corresponding to the simple bound constraints 
!    x_l <= x <= x_u). 
!
!   In addition, the following optional components must be provided and 
!   will be scaled/shifted whenever the array %DG has been allocated:
!
!   %DG is a REAL array of length %n that contains the linear terms dg for 
!    the parametric objective function. 
!
!   %DC_l, %DC_u are REAL arrays of length %m that contain the values of the 
!    arrays dc_l and dc_u for the parametric constraints
!
!   %DX_l, %DX_u are REAL arrays of length %n that contain the values of the 
!    arrays dc_l and dc_u for the parametric constraints

!  trans is a structure of type SCALE_trans_type that holds the shift and 
!   scale factors as calculated in SCALE_get.
!
!  data is a structure of type SCALE_data_type that holds private internal data.
!
!  control is a structure of type SCALE_control_type as defined in the preamble.
!
!  inform is a structure of type SCALE_inform_type as defined in the preamble.
!
!  ---------------------------------------------------------------------------

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( SCALE_trans_type ), INTENT( IN ) :: trans
      TYPE ( SCALE_control_type ), INTENT( IN ) :: control
      TYPE ( SCALE_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SCALE_data_type ), INTENT( INOUT ) :: data

!  no scaling to apply

      IF ( data%scale == 0 ) THEN

!  apply the scaling factors

      ELSE IF ( data%scale > 1 ) THEN
        IF ( ALLOCATED( prob%DG ) ) THEN
          CALL SCALE_apply_factors( prob%n, prob%m, prob%H, prob%A,            &
                                    prob%G, prob%X, prob%X_l, prob%X_u,        &
                                    prob%C, prob%C_l, prob%C_u,                &
                                    prob%Y, prob%Z, .TRUE., trans, control,    &
                                    DG = prob%DG, DX_l = prob%DX_l,            &
                                    DX_u = prob%DX_u, DC_l = prob%DC_l,        &
                                    DC_u = prob%DC_u )
        ELSE
          CALL SCALE_apply_factors( prob%n, prob%m, prob%H, prob%A,            &
                                    prob%G, prob%X, prob%X_l, prob%X_u,        &
                                    prob%C, prob%C_l, prob%C_u,                &
                                    prob%Y, prob%Z, .TRUE., trans, control )
        END IF

!  apply the shifts and scale factors

      ELSE
        IF ( ALLOCATED( prob%DG ) ) THEN
          CALL SCALE_apply_shift_and_scale( prob%n, prob%m, prob%H,            &
                                            prob%A, prob%f, prob%G,            &
                                            prob%X, prob%X_l, prob%X_u,        &
                                            prob%C, prob%C_l, prob%C_u,        &
                                            prob%Y, prob%Z, .TRUE.,            &
                                            trans, data, control, inform,      &
                                            df = prob%df, DG = prob%DG,        &
                                            DX_l = prob%DX_l,                  &
                                            DX_u = prob%DX_u,                  &
                                            DC_l = prob%DC_l,                  &
                                            DC_u = prob%DC_u )
        ELSE
          CALL SCALE_apply_shift_and_scale( prob%n, prob%m, prob%H,            &
                                            prob%A, prob%f, prob%G,            &
                                            prob%X, prob%X_l, prob%X_u,        &
                                            prob%C, prob%C_l, prob%C_u,        &
                                            prob%Y, prob%Z, .TRUE.,            &
                                            trans, data, control, inform )
        END IF
      END IF

      RETURN

!  End of SCALE_apply

      END SUBROUTINE SCALE_apply

!-*-*-*-*-*-*-   S C A L E _ r e c o v e r  S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE SCALE_recover( prob, trans, data, control, inform )

!  ----------------------------------------------------------------------------
!  Undo the effects of the shifts (x_s,f_s) and scale factors (X_s,F_s,C_s) 
!  computed by SCALE_get that have been applied to the data for the quadratic 
!  programming (QP) problem
!
!      min f(x) = 1/2 x^T H x + x^T g + f
!
!      s.t.        c_l <= A x <= c_u, 
!      and         x_l <=  x  <= x_u
!
!  (or optionally to the parametric problem
!
!      min  1/2 x^T H x + x^T g + theta x^T dg + f + theta df
!
!      s.t. c_l + theta dc_l <= A x <= c_u + theta dc_u,
!      and  x_l + theta dx_l <=  x  <= x_u + theta dx_u )
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose values should
!   have been set as described in the preamble to SCALE_apply. On exit,
!   the real data will have been transformed to reverse the effects of the
!   shift and scaling factors calculated by SCALE_get.
!
!  trans is a structure of type SCALE_trans_type that holds the shift and 
!   scale factors as calculated in SCALE_get.
!
!  data is a structure of type SCALE_data_type that holds private internal data.
!
!  control is a structure of type SCALE_control_type as defined in the preamble.
!
!  inform is a structure of type SCALE_inform_type as defined in the preamble.
!
!  ---------------------------------------------------------------------------

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( SCALE_trans_type ), INTENT( IN ) :: trans
      TYPE ( SCALE_control_type ), INTENT( IN ) :: control
      TYPE ( SCALE_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SCALE_data_type ), INTENT( INOUT ) :: data

!  no scaling was applied

      IF ( data%scale == 0 ) THEN

!  undo the effects of the scaling factors

      ELSE IF ( data%scale > 1 ) THEN
        IF ( ALLOCATED( prob%DG ) ) THEN
          CALL SCALE_apply_factors( prob%n, prob%m, prob%H, prob%A,            &
                                    prob%G, prob%X, prob%X_l, prob%X_u,        &
                                    prob%C, prob%C_l, prob%C_u,                &
                                    prob%Y, prob%Z, .FALSE., trans, control,   &
                                    DG = prob%DG, DX_l = prob%DX_l,            &
                                    DX_u = prob%DX_u, DC_l = prob%DC_l,        &
                                    DC_u = prob%DC_u )
        ELSE
          CALL SCALE_apply_factors( prob%n, prob%m, prob%H, prob%A,            &
                                    prob%G, prob%X, prob%X_l, prob%X_u,        &
                                    prob%C, prob%C_l, prob%C_u,                &
                                    prob%Y, prob%Z, .FALSE., trans, control )
        END IF

!  undo the effects of the shifts and scale factors

      ELSE
        IF ( ALLOCATED( prob%DG ) ) THEN
          CALL SCALE_apply_shift_and_scale( prob%n, prob%m, prob%H,            &
                                            prob%A, prob%f, prob%G,            &
                                            prob%X, prob%X_l, prob%X_u,        &
                                            prob%C, prob%C_l, prob%C_u,        &
                                            prob%Y, prob%Z, .FALSE.,           &
                                            trans, data, control, inform,      &
                                            df = prob%df, DG = prob%DG,        &
                                            DX_l = prob%DX_l,                  &
                                            DX_u = prob%DX_u,                  &
                                            DC_l = prob%DC_l,                  &
                                            DC_u = prob%DC_u )
        ELSE
          CALL SCALE_apply_shift_and_scale( prob%n, prob%m, prob%H,            &
                                            prob%A, prob%f, prob%G,            &
                                            prob%X, prob%X_l, prob%X_u,        &
                                            prob%C, prob%C_l, prob%C_u,        &
                                            prob%Y, prob%Z, .FALSE.,           &
                                            trans, data, control, inform )
        END IF
      END IF

      RETURN

!  End of SCALE_recover

      END SUBROUTINE SCALE_recover

!-*-*-*-*-*-*   S C A L E _ g e t _ f a c t o r s _ f r o m _ K  *-*-*-*-*-

      SUBROUTINE SCALE_get_factors_from_K( n, m, H, A, trans, data,            &
                                           control, inform )

!  ---------------------------------------------------------------------------
!
!  Compute row scaling factors for the symmetric matrix
!
!        K = ( H   A(transpose) )
!            ( A        0       )
!
!  using the symmetric version of the algorithm of Curtis and Reid 
!  (J.I.M.A. 10 (1972) 118-124) by approximately minimizing the function
!
!        sum (nonzero K) ( log_2(|k_ij|) + r_j)^2
!
!   The required scalings are then 2^int(r)
!
!   Use Reid's special purpose method for matrices with property "A". 
!   Comments refer to notation in Curtis and Reid's paper

!   The resulting method is to find a solution r to the linear system
!
!       ( M + E ) r = sigma
!
!    using a few iterations of the CG method - M is a diagonal matrix whose
!    entries are the numbers of nonzeros in the rows of K, E replaces the 
!    nonzeros in K by ones and sigma is the vector of column sums of logarithms
!    base 2 of entries of K; the resulting row scale factors are 2**int(r)

!  arguments:
!  ---------

!  H and A See SMT
!  trans%X_scale is an array that need not be be on entry. 
!          On return, it holds the scaling factor for the H rows
!  trans%C_scale is an array that need not be be on entry. 
!          On return, it holds the scaling factor for the A rows
!  inform%status >= 0 for successful entry, -3  if n + m < 1 or ne < 1
!
!  ---------------------------------------------------------------------------

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: H, A
      TYPE ( SCALE_trans_type ), INTENT( INOUT ) :: trans
      TYPE ( SCALE_data_type ), INTENT( INOUT ) :: data
      TYPE ( SCALE_control_type ), INTENT( IN ) :: control
      TYPE ( SCALE_inform_type ), INTENT( INOUT ) :: inform

!  local variables
 
      INTEGER :: i, ii, j, l, npm, ne, iter, a_ne, h_ne
      REAL ( KIND = wp ) :: alpha, beta, ptkp, rtr, rtr_old, log2
      REAL ( KIND = wp ) :: stop_tol, val, s_max, s_min
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering SCALE_get_factors_from_K')") prefix

!  Ensure that input parameters are within allowed ranges

      IF ( n <= 0 .OR. m < 0 .OR. .NOT. QPT_keyword_H( H%type ) .OR.           &
           .NOT. QPT_keyword_A( A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 800
      END IF

!  compute matrix sizes

      IF ( SMT_get( A%type ) == 'DENSE' ) THEN
        a_ne = m * n
      ELSE IF ( SMT_get( A%type ) == 'SPARSE_BY_ROWS' ) THEN
        a_ne = A%ptr( m + 1 ) - 1
      ELSE
        a_ne = A%ne 
      END IF
      IF ( SMT_get( H%type ) == 'DIAGONAL' ) THEN
        h_ne = n
      ELSE IF ( SMT_get( H%type ) == 'DENSE' ) THEN
        h_ne = ( n * ( n + 1 ) ) / 2
      ELSE IF ( SMT_get( H%type ) == 'SPARSE_BY_ROWS' ) THEN
        h_ne = H%ptr( n + 1 ) - 1
      ELSE
        h_ne = H%ne 
      END IF
      npm  = n + m ; ne = a_ne + h_ne

!  check ne

      IF (  ne <= 0 ) THEN
        inform%status = GALAHAD_error_restrictions ; GO TO 800
      END IF

!  allocate space for scale factors

      array_name = 'scale: trans%X_scale'
      CALL SPACE_resize_array( n, trans%X_scale, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: trans%C_scale'
      CALL SPACE_resize_array( m, trans%C_scale, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialize (logs of) scale factors

      trans%X_scale( : n ) = zero ; trans%C_scale( : m ) = zero

!  allocate workspace

      array_name = 'scale: data%ROW_val'
      CALL SPACE_resize_array( npm, data%ROW_val, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: data%RES'
      CALL SPACE_resize_array( npm, data%RES, inform%status,                   &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: data%P'
      CALL SPACE_resize_array( npm, data%P, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: data%PROD'
      CALL SPACE_resize_array( npm, data%PROD, inform%status,                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialise for accumulation of sums and products

      data%ROW_val( : npm ) = zero ; data%RES( : npm ) = zero
      log2 = LOG( two )

!  count non-zeros in the rows - these define the diagonal matrix M - 
!  and compute rhs vectors - contributions from H

      SELECT CASE ( SMT_get( H%type ) )
      CASE ( 'DIAGONAL' ) 
        DO i = 1, n
          val = ABS( H%val( i ) )
          IF ( val /= zero ) THEN
            val = LOG( val ) / log2
            data%ROW_val( i ) = data%ROW_val( i ) + one
            data%RES( i ) = data%RES( i ) - val
          END IF
        END DO
      CASE ( 'DENSE' ) 
        l = 0
        DO i = 1, n
          DO j = 1, i
            l = l + 1
            val = ABS( H%val( l ) )
            IF ( val /= zero ) THEN
              val = LOG( val ) / log2
              data%ROW_val( i ) = data%ROW_val( i ) + one
              data%ROW_val( j ) = data%ROW_val( j ) + one
              data%RES( i ) = data%RES( i ) - val
              IF ( i /= j ) data%RES( j ) = data%RES( j ) - val
            END IF
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, n
          DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
            val = ABS( H%val( l ) )
            IF ( val /= zero ) THEN
              j = H%col( l )
              IF ( j >= 1 .AND. j <= n ) THEN
                val = LOG( val ) / log2
                data%ROW_val( i ) = data%ROW_val( i ) + one
                data%ROW_val( j ) = data%ROW_val( j ) + one
                data%RES( i ) = data%RES( i ) - val
                IF ( i /= j ) data%RES( j ) = data%RES( j ) - val
              END IF
            END IF
          END DO
        END DO
      CASE ( 'COORDINATE' )
        DO l = 1, H%ne
          val = ABS( H%val( l ) )
          IF ( val /= zero ) THEN
            i = H%row( l ) ; j = H%col( l )
            IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
              val = LOG( val ) / log2
              data%ROW_val( i ) = data%ROW_val( i ) + one
              data%ROW_val( j ) = data%ROW_val( j ) + one
              data%RES( i ) = data%RES( i ) - val
              IF ( i /= j ) data%RES( j ) = data%RES( j ) - val
            END IF
          END IF
        END DO
      END SELECT

!  contributions from A

      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' ) 
        l = 0
        DO i = 1, m
          DO j = 1, n
            l = l + 1
            val = ABS( A%val( l ) )
            IF ( val /= zero ) THEN
              val = LOG( val ) / log2
              ii = n + i
              data%ROW_val( ii ) = data%ROW_val( ii ) + one
              data%RES( ii ) = data%RES( ii ) - val
              data%ROW_val( j ) = data%ROW_val( j ) + one 
              data%RES( j ) = data%RES( j ) - val
            END IF
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          ii = n + i
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            val = ABS( A%val( l ) )
            IF ( val /= zero ) THEN
              j = A%col( l )
              IF ( j >= 1 .AND. j <= n ) THEN
                val = LOG( val ) / log2
                data%ROW_val( ii ) = data%ROW_val( ii ) + one
                data%RES( ii ) = data%RES( ii ) - val
                data%ROW_val( j ) = data%ROW_val( j ) + one 
                data%RES( j ) = data%RES( j ) - val
              END IF
            END IF
          END DO
        END DO
      CASE ( 'COORDINATE' )
        DO l = 1, A%ne
          val = ABS( A%val( l ) )
          IF ( val /= zero ) THEN
            i = A%row( l ) ; j = A%col( l )
            IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n ) THEN
              val = LOG( val ) / log2
              ii = n + i
              data%ROW_val( ii ) = data%ROW_val( ii ) + one
              data%RES( ii ) = data%RES( ii ) - val
              data%ROW_val( j ) = data%ROW_val( j ) + one 
              data%RES( j ) = data%RES( j ) - val
            END IF
          END IF
        END DO
      END SELECT

!  find the initial vectors

      WHERE ( data%ROW_val( : npm ) == zero ) data%ROW_val( : npm ) = one
      data%P( : npm ) = data%RES( : npm ) / data%ROW_val( : npm ) 
      data%PROD( : npm ) = data%RES( : npm )
      rtr = SUM( data%RES ** 2 / data%ROW_val )

!  compute the stopping tolerance

      stop_tol = control%stop_tol * ne

!  --------------
!  iteration loop
!  --------------

      inform%status = GALAHAD_ok
      IF ( rtr > stop_tol ) THEN
        DO iter = 1, control%maxit

!  sweep through matrix to add Ep to Mp - contributions from H

          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' ) 
          CASE ( 'DENSE' ) 
            l = 0
            DO i = 1, n
              DO j = 1, i - 1
                l = l + 1
                IF ( H%val( l ) /= zero ) THEN
                  data%PROD( i ) = data%PROD( i ) + data%P( j )
                  data%PROD( j ) = data%PROD( j ) + data%P( i )
                END IF
              END DO
              l = l + 1
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                IF ( H%val( l ) /= zero ) THEN
                  j = H%col( l )
                  IF ( i /= j ) THEN
                    IF ( j >= 1 .AND.  j <= n ) THEN
                      data%PROD( i ) = data%PROD( i ) + data%P( j )
                      data%PROD( j ) = data%PROD( j ) + data%P( i )
                    END IF
                  END IF
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              IF ( H%val( l ) /= zero ) THEN
                i = H%row( l ) ; j = H%col( l )
                IF ( i /= j ) THEN
                  IF ( MIN( i, j ) >= 1 .AND.  MAX( i, j ) <= n ) THEN
                    data%PROD( i ) = data%PROD( i ) + data%P( j )
                    data%PROD( j ) = data%PROD( j ) + data%P( i )
                  END IF
                END IF
              END IF
            END DO
          END SELECT

!  contributions from A

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' ) 
            l = 0
            DO i = 1, m
              ii = n + i
              DO j = 1, n
                l = l + 1
                IF ( A%val( l ) /= zero ) THEN
                  data%PROD( ii ) = data%PROD( ii ) + data%P( j ) 
                  data%PROD( j ) = data%PROD( j ) + data%P( ii )
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, m
              ii = n + i
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                IF ( A%val( l ) /= zero ) THEN
                  j = A%col( l )
                  IF ( j >= 1 .AND. j <= n ) THEN
                    data%PROD( ii ) = data%PROD( ii ) + data%P( j ) 
                    data%PROD( j ) = data%PROD( j ) + data%P( ii )
                  END IF
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              IF ( A%val( l ) /= zero ) THEN
                i = A%row( l ) ; j = A%col( l )
                IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n ) THEN
                  ii = n + i
                  data%PROD( ii ) = data%PROD( ii ) + data%P( j ) 
                  data%PROD( j ) = data%PROD( j ) + data%P( ii )
                END IF
              END IF
            END DO
          END SELECT

          ptkp = DOT_PRODUCT( data%P( : npm ), data%PROD( : npm ) ) 

!  compute the CG stepsize 

          alpha = rtr / ptkp

!  update the solution and residual

          trans%X_scale( : n ) = trans%X_scale( : n ) + alpha * data%P( : n )
          trans%C_scale( : m ) =                                               &
            trans%C_scale( : m ) + alpha * data%P( n + 1 : npm )
          data%RES( : npm ) = data%RES( : npm ) - alpha * data%PROD( : npm )
          rtr_old = rtr
          rtr  = SUM( data%RES( : npm ) ** 2 / data%ROW_val( : npm ) )
          IF ( rtr <= stop_tol ) EXIT

!  compute the CG conjugation parameter

          beta = rtr / rtr_old

!  update the CG serach vector p

          data%P( : npm ) = data%RES( : npm ) / data%ROW_val( : npm )          &
             + beta * data%P( : npm )

!  compute Mp

          data%PROD( : npm ) = data%P( : npm ) * data%ROW_val( : npm )
        END DO
        IF ( iter > control%maxit )                                            &
         inform%status = - GALAHAD_error_max_iterations
        inform%iter = iter
      ELSE
        inform%iter = 0
      END IF

!  ---------------------
!  end of iteration loop
!  ---------------------

!  obtain the scaling factors - factors for the H rows

      trans%X_scale( : n ) = two ** ANINT( trans%X_scale( : n ) )
      s_max = MAXVAL( trans%X_scale( : n ) ) 
      s_min = MINVAL( trans%X_scale( : n ) )
      IF ( control%print_level > 0 .AND. control%out > 0 )                     &
        WRITE( control%out, "( A, '  min, max column scaling = ', 2ES12.4 )" ) &
          prefix, s_min, s_max

!  factors for the A rows

      IF ( m > 0 ) THEN
        trans%C_scale( : m ) = two ** ANINT( trans%C_scale( : m ) )
        s_max = MAXVAL( trans%C_scale( : m ) ) 
        s_min = MINVAL( trans%C_scale( : m ) )
        IF ( control%print_level > 0 .AND. control%out > 0 )                   &
          WRITE( control%out, "( A, ' min, max   row  scaling = ', 2ES12.4 )") &
            prefix, s_min, s_max
      END IF

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving SCALE_get_factors_from_K' )") prefix
      RETURN

!  error returns

  800 CONTINUE
      IF ( control%error > 0 .AND. control%print_level >= 0 )                  &
        WRITE( control%error,                                                  &
        "( ' * Error return from SCALE_get_factors * status = ', I0 )" )       &
          inform%status
      RETURN

!  allocation error

  900 CONTINUE 
      inform%status = GALAHAD_error_allocate
      IF ( control%out > 0 .AND. control%print_level > 0 ) WRITE( control%out, &
        "( A, ' ** Message from -SCALE_get_factors-', /,  A,                   &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status
      RETURN  

!  End of SCALE_get_factors

      END SUBROUTINE SCALE_get_factors_from_K

!-*-*-*-*-*   S C A L E _ g e t _ f a c t o r s _ f r o m _ A  *-*-*-*-*

      SUBROUTINE SCALE_get_factors_from_A( n, m, A, trans, data,               &
                                           control, inform )

!  ---------------------------------------------------------------------------
!
!   Compute row and column scalings using the algorithm of Curtis and Reid
!   (J.I.M.A. 10 (1972) 118-124) by approximately minimizing the function
!
!        sum (nonzero A) ( log_2(|a_ij|) + r_i + c_j)^2
!
!   The required scalings are then 2^int(r) and 2^int(c) respectively
!
!   Use Reid's special purpose method for matrices with property "A". 
!   Comments refer to equation numbers in Curtis and Reid's paper
!
!   The resulting method is to find a solution (r,c) to the linear system
!
!       ( M   E ) ( r ) = ( sigma )
!       ( E^T N ) ( c )   (  tau  )
!
!    using a few iterations of the CG method - M and N are diagonal
!    matrices whose entries are the numbers of nonzeros in the rows
!    and columns of A, E replaces the nonzeros in A by ones and sigma and
!    tau are the vectors of row and column sums of logarithms base 2 of
!    entries of A; the resulting row and column scale factors are 2**int(r,c)
!
!  arguments:
!  ---------
!
!  A See SMT
!  trans%X_scale is an array that need not be be on entry. 
!                On return, it holds the scaling factor for the columns of A
!  trans%C_scale is an array that need not be be on entry. 
!                On return, it holds the scaling factor for the rows of A
!  inform%status >= 0 for successful entry, -3  if n < 1, m < 1 or ne < 1
!
!  ---------------------------------------------------------------------------

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m , n
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SCALE_trans_type ), INTENT( INOUT ) :: trans
      TYPE ( SCALE_data_type ), INTENT( INOUT ) :: data
      TYPE ( SCALE_control_type ), INTENT( IN ) :: control
      TYPE ( SCALE_inform_type ), INTENT( INOUT ) :: inform

!  local variables

      INTEGER :: i, iter, j, l, a_ne
      REAL ( KIND = wp ) :: e, e_old, e_prod, q, q_old, q_prod, log2
      REAL ( KIND = wp ) :: s, s_old, s_max, s_min, stop_tol, val
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering SCALE_get_factors_from_A')") prefix

!  Ensure that input parameters are within allowed ranges

      IF ( n <= 0 .OR. m < 0 .OR. .NOT. QPT_keyword_A( A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 800
      END IF

!  compute matrix sizes

      IF ( SMT_get( A%type ) == 'DENSE' ) THEN
        a_ne = m * n
      ELSE IF ( SMT_get( A%type ) == 'SPARSE_BY_ROWS' ) THEN
        a_ne = A%ptr( m + 1 ) - 1
      ELSE
        a_ne = A%ne 
      END IF

!  set the stopping tolerance

      stop_tol = a_ne * control%stop_tol

!  allocate space for scale factors

      array_name = 'scale: trans%X_scale'
      CALL SPACE_resize_array( n, trans%X_scale, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: trans%C_scale'
      CALL SPACE_resize_array( m, trans%C_scale, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialize (logs of) scale factors

      trans%X_scale( : n ) = zero ; trans%C_scale( : m ) = zero

!  allocate workspace

      array_name = 'scale: data%ROW_val'
      CALL SPACE_resize_array( m, data%ROW_val, inform%status,                 &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: data%COL_val'
      CALL SPACE_resize_array( n, data%COL_val, inform%status,                 &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: data%RES'
      CALL SPACE_resize_array( n, data%RES, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: data%P'
      CALL SPACE_resize_array( n, data%P, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: data%PROD'
      CALL SPACE_resize_array( m, data%PROD, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialise for accumulation of sums and products

      data%ROW_val( : m ) = zero ; data%COL_val( : n ) = zero 
      data%RES( : n ) = zero 
      log2 = LOG( two )

!  count non-zeros in the rows, and compute r.h.s. vectors; use C_scale to store
!  the row r.h.s. (sigma in Curtis+Reid)

      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' ) 
        l = 0
        DO i = 1, m
          DO j = 1, n
            l = l + 1
            val = ABS( A%val( l ) )
            IF ( val /= zero ) THEN
              val = LOG( val ) / log2
              data%ROW_val( i ) = data%ROW_val( i ) + one 
              data%COL_val( j ) = data%COL_val( j ) + one 
              trans%C_scale( i ) = trans%C_scale( i ) + val 
              data%RES( j ) = data%RES( j ) + val 
            END IF
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            val = ABS( A%val( l ) )
            IF ( val /= zero ) THEN
              j = A%col( l )
              IF ( j >= 1 .AND. j <= n ) THEN
                val = LOG( val ) / log2
                data%ROW_val( i ) = data%ROW_val( i ) + one 
                data%COL_val( j ) = data%COL_val( j ) + one 
                trans%C_scale( i ) = trans%C_scale( i ) + val 
                data%RES( j ) = data%RES( j ) + val 
               END IF 
             END IF 
           END DO 
         END DO 
      CASE ( 'COORDINATE' )
        DO l = 1, A%ne
          val = ABS( A%val( l ) )
          IF ( val /= zero ) THEN
            i = A%row( l ) ; j = A%col( l )
            IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n ) THEN
              val = LOG( val ) / log2
              data%ROW_val( i ) = data%ROW_val( i ) + one 
              data%COL_val( j ) = data%COL_val( j ) + one 
              trans%C_scale( i ) = trans%C_scale( i ) + val 
              data%RES( j ) = data%RES( j ) + val 
            END IF
          END IF
        END DO
      END SELECT

!  account for structural singularity

      WHERE ( data%ROW_val( : m ) == zero ) data%ROW_val( : m ) = one 
      WHERE ( data%COL_val( : n ) == zero ) data%COL_val( : n ) = one 

!  form M^-1 sigma and N^-1 tau (in C+R's notation)

      data%PROD( : m ) = trans%C_scale( : m ) / data%ROW_val( : m )
      data%RES( : n ) = data%RES( : n ) / data%COL_val( : n )

!  compute initial residual vector

      trans%C_scale( : m ) = data%PROD( : m )

      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' ) 
        l = 0
        DO i = 1, m
          DO j = 1, n
            l = l + 1
            IF ( A%val( l ) /= zero ) trans%C_scale( i ) =                     &
              trans%C_scale( i ) - data%RES( j ) / data%ROW_val( i )    ! (4.3)
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            val = ABS( A%val( l ) )
            IF ( val /= zero ) THEN
              j = A%col( l )
             IF ( j >= 1 .AND. j <= n ) trans%C_scale( i ) =                  &
                trans%C_scale( i ) - data%RES( j ) / data%ROW_val( i )  ! (4.3)
             END IF 
           END DO 
         END DO 
      CASE ( 'COORDINATE' )
        DO l = 1, A%ne
          val = ABS( A%val( l ) )
          IF ( val /= zero ) THEN
            i = A%row( l ) ; j = A%col( l )
            IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n )                  &
              trans%C_scale( i ) =                                             &
                trans%C_scale( i ) - data%RES( j ) / data%ROW_val( i )  ! (4.3)
          END IF
        END DO
      END SELECT

!  set initial values

      e = zero ; q = one 
      s = DOT_PRODUCT( data%ROW_val( : m ), trans%C_scale( : m ) ** 2 ) 

      IF ( control%out > 0 .AND. control%print_level >= 2 )                    &
         WRITE( control%out, "( ' iter     error   stop_tol = ', ES12.4 )" )   &
            control%stop_tol
      IF ( control%out > 0 .AND. control%print_level >= 2 )                    &
          WRITE( control%out, "( I5, ES12.4 )" ) 0, s

      inform%status = GALAHAD_ok
      IF ( s > stop_tol ) THEN
        data%P( : n ) = zero 

!  --------------
!  iteration loop
!  --------------

          DO iter = 1, control%maxit

!  update column residual vector

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' ) 
            l = 0
            DO i = 1, m
              DO j = 1, n
                l = l + 1
                IF ( A%val( l ) /= zero )                                      &
                  trans%X_scale( j ) = trans%X_scale( j ) + trans%C_scale( i )
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, m
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                IF ( A%val( l ) /= zero ) THEN
                  j = A%col( l )
                  IF ( j >= 1 .AND. j <= n )                                   &
                    trans%X_scale( j ) = trans%X_scale( j ) + trans%C_scale( i )
                END IF
              END DO 
            END DO 
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              val = ABS( A%val( l ) )
              IF ( val /= zero ) THEN
                i = A%row( l ) ; j = A%col( l )
                IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n )              &
                  trans%X_scale( j ) = trans%X_scale( j ) + trans%C_scale( i )
              END IF
            END DO
          END SELECT

!  rescale column residual

          s_old = s ; s = zero 
          DO j = 1, n 
            val = - trans%X_scale( j ) / q 
            trans%X_scale( j ) = val / data%COL_val( j )               ! (4.4a)
            s = s + val * trans%X_scale( j )                           ! (4.5a)
          END DO 

          IF ( control%out > 0 .AND. control%print_level >= 2 )                &
            WRITE( control%out, "( I5, ES12.4 )" ) iter, s

!  rescale row residual vector

          e_old = e 
          e = q * s / s_old                                            ! (4.6)
          q = one - e                                                  ! (4.7)
          IF ( s <= stop_tol ) e = zero 
          trans%C_scale( : m ) =                                               &
            trans%C_scale( : m ) * e * data%ROW_val( : m )

!  test for termination

          IF ( s <= stop_tol ) GO TO 100 
          e_prod = e * e_old 

!  update row residual

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' ) 
            l = 0
            DO i = 1, m
              DO j = 1, n
                l = l + 1
                IF ( A%val( l ) /= zero )                                      &
                  trans%C_scale( i ) = trans%C_scale( i ) + trans%X_scale( j )
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, m
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                IF ( A%val( l ) /= zero ) THEN
                  j = A%col( l )
                  IF ( j >= 1 .AND. j <= n )                                   &
                    trans%C_scale( i ) = trans%C_scale( i ) + trans%X_scale( j )
                END IF
              END DO 
            END DO 
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              val = ABS( A%val( l ) )
              IF ( val /= zero ) THEN
                i = A%row( l ) ; j = A%col( l )
                IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n )              &
                  trans%C_scale( i ) = trans%C_scale( i ) + trans%X_scale( j )
              END IF
            END DO
          END SELECT

!  rescale row residual

          s_old = s ; s = zero 
          DO i = 1, m 
             val = - trans%C_scale( i ) / q 
             trans%C_scale( i ) = val / data%ROW_val( i )              ! (4.4b)
             s = s + val * trans%C_scale( i )                          ! (4.5b)
          END DO 
          e_old = e ; e = q * s / s_old                                ! (4.6)
          q_old = q ; q = one - e                                      ! (4.7)

          IF ( control%out > 0 .AND. control%print_level >= 2 )                &
            WRITE( control%out, "( I5, ES12.4 )" ) iter, s

!  special fixup for last iteration

          IF ( s <= stop_tol ) q = one 

!  rescale column residual vector

          q_prod = q * q_old 
          data%P( : n ) =                                                      &
            ( e_prod * data%P( : n ) + trans%X_scale( : n ) ) / q_prod 
          data%RES( : n ) = data%RES( : n ) + data%P( : n ) 

!  test for termination

          IF ( s <= stop_tol ) EXIT  

!  update column scaling factors

          trans%X_scale( : n ) =                                               &
            e * trans%X_scale( : n ) * data%COL_val( : n )
        END DO 
        IF ( iter > control%maxit )                                            &
          inform%status = - GALAHAD_error_max_iterations
        inform%iter = iter
      ELSE
        inform%iter = 0
      END IF

!  ---------------------
!  end of iteration loop
!  ---------------------

      trans%C_scale( : m ) = trans%C_scale( : m ) * data%ROW_val( : m )

!  sweep through matrix to prepare to get row scaling powers

  100 CONTINUE 

      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' ) 
        l = 0
        DO i = 1, m
          DO j = 1, n
            l = l + 1
            IF ( A%val( l ) /= zero )                                          &
              trans%C_scale( i ) = trans%C_scale( i ) + data%RES( j )
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            IF ( A%val( l ) /= zero ) THEN
              j = A%col( l )
              IF ( j >= 1 .AND. j <= n )                                       &
                trans%C_scale( i ) = trans%C_scale( i ) + data%RES( j )
            END IF
          END DO 
        END DO 
      CASE ( 'COORDINATE' )
        DO l = 1, A%ne
          val = ABS( A%val( l ) )
          IF ( val /= zero ) THEN
            i = A%row( l ) ; j = A%col( l )
            IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n )                  &
              trans%C_scale( i ) = trans%C_scale( i ) + data%RES( j )
          END IF
        END DO
      END SELECT

!  final conversion to output values

      trans%C_scale( : m )                                                     &
        = trans%C_scale( : m ) / data%ROW_val( : m ) - data%PROD( : m )
      trans%X_scale( : n ) = - data%RES( : n )

!  obtain the scaling factors - factors for the H rows

      trans%X_scale( : n ) = two ** ANINT( trans%X_scale( : n ) )
      s_max = MAXVAL( trans%X_scale( : n ) ) 
      s_min = MINVAL( trans%X_scale( : n ) )
      IF ( control%print_level > 0 .AND. control%out > 0 )                     &
        WRITE( control%out, "( A, ' min, max column scaling = ', 2ES12.4 )" )  &
          prefix, s_min, s_max

!  factors for the A rows

      IF ( m > 0 ) THEN
        trans%C_scale( : m ) = two ** ANINT( trans%C_scale( : m ) )
        s_max = MAXVAL( trans%C_scale( : m ) ) 
        s_min = MINVAL( trans%C_scale( : m ) )
        IF ( control%print_level > 0 .AND. control%out > 0 )                   &
          WRITE( control%out, "( A, ' min, max   row  scaling = ', 2ES12.4 )") &
            prefix, s_min, s_max
      END IF

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving SCALE_get_factors_from_A' )") prefix

      RETURN  

!  error returns

  800 CONTINUE 
      IF ( control%error > 0 .AND. control%print_level >= 0 )                  &
        WRITE( control%error,                                                  &
        "( ' * Error return from SCALE_get_factors * status = ', I0 )" )       &
          inform%status
      RETURN

!  allocation error

  900 CONTINUE 
      inform%status = GALAHAD_error_allocate
      IF ( control%out > 0 .AND. control%print_level > 0 ) WRITE( control%out, &
        "( A, ' ** Message from -SCALE_get_factors-', /,  A,                   &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status
      RETURN  

!  End of subroutine SCALE_get_factors_from_A

      END SUBROUTINE SCALE_get_factors_from_A

!-*-*-*-*-*-   S C A L E _ g e t _ f a c t o r s _ f r o m _ A  -*-*-*-*-*-

      SUBROUTINE SCALE_normalize_rows_of_A( n, m, A, trans, data,              &
                                            control, inform )

!  ---------------------------------------------------------------------------
!
!   Renormalize the rows of C_scale * A * X_scale so that each has a 
!   one-norm close to one
!
!  arguments:
!  ---------
!
!  A See SMT
!  trans%C_scale is an array that must be set on entry to the current row 
!                scaling factors C_scale. On exit, trans%C_scale may have been 
!                altered to reflect  the rescaling
!  trans%X_scale is an array that must be set on entry to the current column 
!                scaling factors X_scale. It is unaltered on exit
!
!
!  ---------------------------------------------------------------------------

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m , n
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SCALE_trans_type ), INTENT( INOUT ) :: trans
      TYPE ( SCALE_data_type ), INTENT( INOUT ) :: data
      TYPE ( SCALE_control_type ), INTENT( IN ) :: control
      TYPE ( SCALE_inform_type ), INTENT( INOUT ) :: inform

!  local variables

      INTEGER :: i, j, l
      REAL ( KIND = wp ) :: ci, val, log2, s_min, s_max
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( m <= 0 ) RETURN

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A,' entering SCALE_normalize_rows_of_A')") prefix

      log2 = LOG( two )

!  allocate workspace

      array_name = 'scale: data%ROW_val'
      CALL SPACE_resize_array( m, data%ROW_val, inform%status,                 &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialise the row norms

      data%ROW_val( : m ) = zero

!  compute the row norms

      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' ) 
        l = 0
        DO i = 1, m
          ci = trans%C_scale( i )
          DO j = 1, n
            l = l + 1
            val = A%val( l )
            IF ( val /= zero ) data%ROW_val( i ) =                             &
                data%ROW_val( i ) + ABS( ci * trans%X_scale( j ) * val )
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          ci = trans%C_scale( i )
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            val = A%val( l )
            IF ( val /= zero ) THEN
              j = A%col( l )
              IF ( j >= 1 .AND. j <= n ) data%ROW_val( i ) =                   &
                data%ROW_val( i ) + ABS( ci * trans%X_scale( j ) * val )
            END IF
          END DO
        END DO
      CASE ( 'COORDINATE' )
        DO l = 1, A%ne
          val = A%val( l )
          IF ( val /= zero ) THEN
            i = A%row( l ) ; j = A%col( l )
            IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n )                  &
              data%ROW_val( i ) = data%ROW_val( i ) +                          &
                ABS( trans%C_scale( i ) * trans%X_scale( j ) * val )
          END IF
        END DO
      END SELECT

!  factors for the A rows

      DO i = 1, m
        IF ( data%ROW_val( i ) /= zero ) trans%C_scale( i ) =                  &
          trans%C_scale( i ) / two ** ANINT( LOG( data%ROW_val( i ) ) / log2 )
      END DO
      trans%C_scale( : m ) = two ** ANINT( trans%C_scale( : m ) )
      s_max = MAXVAL( trans%C_scale( : m ) ) 
      s_min = MINVAL( trans%C_scale( : m ) )
      IF ( control%print_level > 0 .AND. control%out > 0 )                     &
        WRITE( control%out, "( A, ' MIN, MAX   row  scaling = ', 2ES12.4 )" )  &
          prefix, s_min, s_max

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A,' leaving SCALE_normalize_rows_of_A' )") prefix
 
      RETURN

!  allocation error

  900 CONTINUE 
      inform%status = GALAHAD_error_allocate
      IF ( control%out > 0 .AND. control%print_level > 0 ) WRITE( control%out, &
        "( A, ' ** Message from -SCALE_normalize_rows_of_A-', /,  A,           &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status
      RETURN  

!  End of SCALE_normalize_rows_of_A

      END SUBROUTINE SCALE_normalize_rows_of_A

!-*-  S C A L E _ g e t _ s i n k h o rn _ k n o p p   S U B R O U T I N E   -*-

      SUBROUTINE SCALE_get_sinkhorn_knopp( n, m, H, A,                         &
                                           trans, data, control, inform )

!  ---------------------------------------------------------------------------
!
!  Compute row scaling factors for the symmetric matrix
!
!        K = ( H   A(transpose) )
!            ( A        0       )
!
!  using the Sinkhorn-Knopp algorithm. 
!
!  In particular, let K_0 = K, S_0 = 1 and k = 0, and iterate 
!  until | || (K_k)i_th column ||_infty - 1 | < tol as follows:
!
!    D_k+1 = Diag_i( || (K_k)i_th column ||_infty )
!    S_k+1 = D_k+1^-1/2 S_k
!
!  See 
!    R. Sinkhorn and P. Knopp (1967).
!   "Concerning nonnegative matrices and doubly stochastic matrices".
!    Pacific J. Math. 21(2) 343-348. 
!
!  ---------------------------------------------------------------------------

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: H, A
      TYPE ( SCALE_trans_type ), INTENT( INOUT ) :: trans
      TYPE ( SCALE_data_type ), INTENT( INOUT ) :: data
      TYPE ( SCALE_control_type ), INTENT( IN ) :: control
      TYPE ( SCALE_inform_type ), INTENT( INOUT ) :: inform

!  local variables

      INTEGER :: i, j, l
      REAL ( KIND = wp ) :: error, val
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering SCALE_get_sinkhorn_knopp')") prefix

!  Ensure that input parameters are within allowed ranges

      IF ( n <= 0 .OR. m < 0 .OR. .NOT. QPT_keyword_H( H%type ) .OR.           &
           .NOT. QPT_keyword_A( A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 800
      END IF

!  allocate space for scale factors

      array_name = 'scale: trans%X_scale'
      CALL SPACE_resize_array( n, trans%X_scale, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: trans%C_scale'
      CALL SPACE_resize_array( m, trans%C_scale, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialize scale factors

      trans%X_scale( : n ) = one ; trans%C_scale( : m ) = one

!  allocate workspace

      array_name = 'scale: data%ROW_val'
      CALL SPACE_resize_array( m, data%ROW_val, inform%status,                 &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: data%COL_val'
      CALL SPACE_resize_array( n, data%COL_val, inform%status,                 &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialise infinity norms of the columns of K

      data%ROW_val( : m ) = zero ; data%COL_val( : n ) = zero

!  compute the infinity norms of the columns of H

      SELECT CASE ( SMT_get( H%type ) )
      CASE ( 'DIAGONAL' ) 
        DO i = 1, n
          data%COL_val( i ) = ABS( H%val( i ) )
        END DO
      CASE ( 'DENSE' ) 
        l = 0
        DO i = 1, n
          DO j = 1, i
            l = l + 1
            val = ABS( H%val( l ) )
            IF ( val /= zero ) THEN
              data%COL_val( i ) = MAX( data%COL_val( i ), val )
              IF ( i /= j ) data%COL_val( j ) = MAX( data%COL_val( j ), val )
            END IF
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, n
          DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
            val = ABS( H%val( l ) )
            IF ( val /= zero ) THEN
              j = H%col( l )
              IF ( j >= 1 .AND. j <= n ) THEN
                data%COL_val( i ) = MAX( data%COL_val( i ), val )
                IF ( i /= j ) data%COL_val( j ) = MAX( data%COL_val( j ), val )
              END IF
            END IF
          END DO
        END DO
      CASE ( 'COORDINATE' )
        DO l = 1, H%ne
          val = ABS( H%val( l ) )
          IF ( val /= zero ) THEN
            i = H%row( l ) ; j = H%col( l )
            IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
              data%COL_val( i ) = MAX( data%COL_val( i ), val )
              IF ( i /= j ) data%COL_val( j ) = MAX( data%COL_val( j ), val )
            END IF
          END IF
        END DO
      END SELECT

!  compute the infinity norms of the columns of K by including contributions 
!  from A

      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' ) 
        l = 0
        DO i = 1, m
          DO j = 1, n
            l = l + 1
            val = ABS( A%val( l ) )
            IF ( val /= zero ) THEN
              data%ROW_val( i ) = MAX( data%ROW_val( i ), val )
              data%COL_val( j ) = MAX( data%COL_val( j ), val )
            END IF
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            val = ABS( A%val( l ) )
            IF ( val /= zero ) THEN
              j = A%col( l )
              IF ( j >= 1 .AND. j <= n ) THEN
                data%ROW_val( i ) = MAX( data%ROW_val( i ), val )
                data%COL_val( j ) = MAX( data%COL_val( j ), val )
              END IF
            END IF
          END DO
        END DO
      CASE ( 'COORDINATE' )
        DO l = 1, A%ne
          val = ABS( A%val( l ) )
          IF ( val /= zero ) THEN
            i = A%row( l ) ; j = A%col( l )
            IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n ) THEN
              data%ROW_val( i ) = MAX( data%ROW_val( i ), val )
              data%COL_val( j ) = MAX( data%COL_val( j ), val )
            END IF
          END IF
        END DO
      END SELECT

!  --------------
!  iteration loop
!  --------------

      inform%status = GALAHAD_ok
      IF ( control%out > 0 .AND. control%print_level >= 2 )                    &
        WRITE( control%out, "( ' iter     error   stop_tol = ', ES12.4 )" )    &
          control%stop_tol
 
      inform%iter = 0
      DO
        inform%iter = inform%iter + 1

!  test for convergence

        IF ( m > 0 ) THEN
          error = MAX( MAXVAL( ABS( data%COL_val( : n ) - one ),               &
                               MASK = data%COL_val( : n ) /= zero ),           &
                       MAXVAL( ABS( data%ROW_val( : m ) - one ),               &
                               MASK = data%ROW_val( : m ) /= zero ) )
        ELSE
          error = MAXVAL( ABS( data%COL_val( : n ) - one ) ,                   &
                          MASK = data%COL_val( : n ) /= zero )
        END IF

        IF ( control%out > 0 .AND. control%print_level >= 2 )                  &
          WRITE( control%out, "( I5, ES12.4 )" ) inform%iter, error

        IF ( error <= control%stop_tol ) EXIT
        IF ( inform%iter >  control%maxit )  THEN
          inform%status = - GALAHAD_error_max_iterations ; EXIT
        END IF

!  update scalings

        WHERE( data%COL_val( : n ) /= zero ) trans%X_scale( : n ) =            &
          trans%X_scale( : n ) / SQRT( data%COL_val( : n ) ) 
        WHERE( data%ROW_val( : m ) /= zero ) trans%C_scale( : m ) =            &
          trans%C_scale( : m ) / SQRT( data%ROW_val( : m ) )
         
!  initialise infinity norms of the columns of the scaled K

        data%ROW_val( : m ) = zero ; data%COL_val( : n ) = zero

!  compute the infinity norms of the columns of the scaled H

        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' ) 
          DO i = 1, n
            data%COL_val( i ) = ABS( H%val( i ) ) * trans%X_scale( i ) ** 2
          END DO
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, n
            DO j = 1, i
              l = l + 1
              val = ABS( H%val( l ) )
              IF ( val /= zero ) THEN
                val = val * trans%X_scale( i ) * trans%X_scale( j )
                data%COL_val( i ) = MAX( data%COL_val( i ), val )
                IF ( i /= j ) data%COL_val( j ) = MAX( data%COL_val( j ), val )
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, n
            DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
              val = ABS( H%val( l ) )
              IF ( val /= zero ) THEN
                j = H%col( l )
                IF ( j >= 1 .AND. j <= n ) THEN
                  val = val * trans%X_scale( i ) * trans%X_scale( j )
                  data%COL_val( i ) = MAX( data%COL_val( i ), val )
                  IF ( i /= j ) data%COL_val( j ) = MAX( data%COL_val( j ), val)
                END IF
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, H%ne
            val = ABS( H%val( l ) )
            IF ( val /= zero ) THEN
              i = H%row( l ) ; j = H%col( l )
              IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
                val = val * trans%X_scale( i ) * trans%X_scale( j )
                data%COL_val( i ) = MAX( data%COL_val( i ), val )
                IF ( i /= j ) data%COL_val( j ) = MAX( data%COL_val( j ), val )
              END IF
            END IF
          END DO
        END SELECT

!  compute the infinity norms of the columns of the scaled K by including 
!  contributions from the scaled A

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1
              val = ABS( A%val( l ) )
              IF ( val /= zero ) THEN
                val = val * trans%C_scale( i ) * trans%X_scale( j )
                data%ROW_val( i ) = MAX( data%ROW_val( i ), val )
                data%COL_val( j ) = MAX( data%COL_val( j ), val )
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              val = ABS( A%val( l ) )
              IF ( val /= zero ) THEN
                j = A%col( l )
                IF ( j >= 1 .AND. j <= n ) THEN
                  val = val * trans%C_scale( i ) * trans%X_scale( j )
                  data%ROW_val( i ) = MAX( data%ROW_val( i ), val )
                  data%COL_val( j ) = MAX( data%COL_val( j ), val )
                END IF
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            val = ABS( A%val( l ) )
            IF ( val /= zero ) THEN
              i = A%row( l ) ; j = A%col( l )
              IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n ) THEN
                val = val * trans%C_scale( i ) * trans%X_scale( j )
                data%ROW_val( i ) = MAX( data%ROW_val( i ), val )
                data%COL_val( j ) = MAX( data%COL_val( j ), val )
              END IF
            END IF
          END DO
        END SELECT

!  ---------------------
!  end of iteration loop
!  ---------------------

      END DO

      IF ( control%print_level > 0 .AND. control%out > 0 )                     &
        WRITE( control%out, "( A, ' min, max column scaling = ', 2ES12.4 )" )  &
          prefix, MINVAL( trans%X_scale( : n ) ), MAXVAL( trans%X_scale( : n ) )

      IF ( m > 0 ) THEN
        IF ( control%print_level > 0 .AND. control%out > 0 )                   &
          WRITE( control%out, "( A, ' min, max   row  scaling = ', 2ES12.4 )") &
            prefix, MINVAL( trans%C_scale( : m ) ), MAXVAL( trans%C_scale( : m))
      END IF

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving SCALE_get_sinkhorn_knopp' )") prefix
      RETURN

!  error returns

  800 CONTINUE 
      IF ( control%error > 0 .AND. control%print_level >= 0 )                  &
        WRITE( control%error,                                                  &
        "( ' * Error return from SCALE_get_sinkhorn_knopp * status = ', I0 )" )&
          inform%status
      RETURN

!  allocation error

  900 CONTINUE 
      inform%status = GALAHAD_error_allocate
      IF ( control%out > 0 .AND. control%print_level > 0 ) WRITE( control%out, &
        "( A, ' ** Message from -SCALE_get_sinkhorn_knopp-', /,  A,            &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status
      RETURN  

!  End of SCALE_get_sinkhorn_knopp

      END SUBROUTINE SCALE_get_sinkhorn_knopp

!-*-*-*-*-*-*-*-*-*-   S C A L E _ a p p l y _ f a c t o r s  -*-*-*-*-*-*-*-

      SUBROUTINE SCALE_apply_factors( n, m, H, A, G, X, X_l, X_u, C, C_l,      &
                                      C_u, Y, Z, apply, trans, control,        &
                                      DG, DX_l, DX_u, DC_l, DC_u )

!  -------------------------------------------------------------------
!
!  Scale or unscale the problem
!
!      min  1/2 x^T H x + x^T g
!      s.t. c_l <= A x <= c_u, x_l <= x < x_u
!
!   (optionally the parametric problem
!
!      min   1/2 x(T) H x + g(T) x + theta dg(T) x
!      s.t.  c_l + theta dc_l <= A x <= c_u + theta dc_u
!      and   x_l + theta dx_l <=  x  <= x_u + theta dx_u )
!
!  and its Lagrange multipliers/dual variables so that the resulting problem is
!
!      min  1/2 y^T ( X_s H X_s ) v + v^T ( X_s g )
!      s.t. ( C_s c_l )  <= ( C_s A X_s ) v <=  ( C_s c_u ),   
!      and  ( X_s^-1 x_l) <=       v       <= ( X_s^-1 x_u )
!
!   (optionally the parametric problem
!
!      min  1/2 y^T ( X_s H X_s ) v + y^T ( X_s g ) + theta y^T ( X_s dg )
!      s.t. ( C_s c_l ) + theta ( C_s dc_l ) 
!                 <= ( C_s A X_s ) v <=
!           ( C_s c_u ) + theta ( C_s dc_u )
!      and  ( X_s^-1 x_l ) + theta ( X_s^-1 dx_l )
!                       <= v <=
!           ( X_s^-1 x_u ) + theta ( X_s^-1 x_u ).)
!
!  If apply is .TRUE., X_s and C_s are as input in trans%X_scale and 
!  trans%C_scale. Otherwise, X_s and C_s are the reciprocals of trans%X_scale 
!  and trans%C_scale (the transformations are un-applied)
!
!  The data H, x, g, A, c_l, c_u, x_l and x_u and the multipliers for
!  the general constraints and dual variables for the bounds is input as 
!           H, X, G, A, C_l, C_u, X_l, X_u, Y and Z 
!  (and optionally C = Ax, DG, DC_l, DC_u, DX_l and DX_u )
!
!  The resulting scaled variants, 
!  ( X_s H X_s ), ( X_s^-1 x ), ( X_s g ), ( C_s A X_s ), ( C_s c bounds ), 
!  ( X_s^-1 x bounds), ( C_s^-1 multipliers) and ( X_s^-1 duals ) are output as 
!           H, X, G, A, ( C_l, C_u ), ( X_l, X_u ), Y and Z
!  (optionally (C_s c ), ( X_s g ), ( C_s c bounds ) and ( X_s^-1 x bounds) are 
!     output as C, DG, DC_l, DC_u, DX_l and DX_u )
!
!  -------------------------------------------------------------------

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      LOGICAL, INTENT( IN ) :: apply
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: G, X, X_l, X_u, Z
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C, C_l, C_u, Y
      REAL ( KIND = wp ), OPTIONAL, DIMENSION( n ) :: DG, DX_l, DX_u
      REAL ( KIND = wp ), OPTIONAL, DIMENSION( m ) :: DC_l, DC_u
      TYPE ( SMT_type ), INTENT( INOUT ) :: H, A
      TYPE ( SCALE_trans_type ), INTENT( IN ) :: trans
      TYPE ( SCALE_control_type ), INTENT( IN ) :: control

!  local variables

      INTEGER :: i, j, l
      REAL ( KIND = wp ) :: val

! ================
!  Scale the data
! ================

      IF ( apply ) THEN

!  scale H

        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' ) 
          DO i = 1, n
            val = H%val( i )
            IF ( val /= zero )                                                 &
              H%val( i ) = val * trans%X_scale( i ) * trans%X_scale( i )
          END DO
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, n
            DO j = 1, i
              l = l + 1
              val = H%val( l )
              IF ( val /= zero )                                               &
                H%val( l ) = val * trans%X_scale( i ) * trans%X_scale( j )
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, n
            DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
              val = H%val( l )
              IF ( val /= zero ) THEN
                j = H%col( l )
                IF ( j >= 1 .AND. j <= n )                                     &
                  H%val( l ) = val * trans%X_scale( i ) * trans%X_scale( j )
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, H%ne
            val = H%val( l )
            IF ( val /= zero ) THEN
              i = H%row( l ) ; j = H%col( l )
              IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n )                   &
                H%val( l ) = val * trans%X_scale( i ) * trans%X_scale( j )
            END IF
          END DO
        END SELECT

!  scale A

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1
              val = A%val( l )
              IF ( val /= zero )                                               &
                A%val( l ) = val * trans%C_scale( i ) * trans%X_scale( j )
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                j = A%col( l )
                IF ( j >= 1 .AND. j <= n )                                     &
                  A%val( l ) = val * trans%C_scale( i ) * trans%X_scale( j )
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            val = A%val( l )
            IF ( val /= zero ) THEN
              i = A%row( l ) ; j = A%col( l )
              IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n )                &
                A%val( l ) = val * trans%C_scale( i ) * trans%X_scale( j )
            END IF
          END DO
        END SELECT

!  scale X and G

        X = X / trans%X_scale ; G = G * trans%X_scale
        IF ( PRESENT( DG ) ) DG = DG * trans%X_scale
        C = C * trans%C_scale

!  Scale the bounds on X and the associated dual variables

        DO i = 1, n
          IF ( X_l( i ) == X_u( i ) ) THEN
            X_l( i ) = X_l( i ) / trans%X_scale( i )
            X_u( i ) = X_u( i ) / trans%X_scale( i )
          ELSE 
            IF ( X_l( i ) > - control%infinity ) THEN
              X_l( i )  = X_l( i ) / trans%X_scale( i )
              IF ( PRESENT( DX_l ) ) DX_l( i )  = DX_l( i ) / trans%X_scale( i )
            END IF
            IF  ( X_u( i ) < control%infinity ) THEN 
              X_u( i ) = X_u( i ) / trans%X_scale( i )
              IF ( PRESENT( DX_u ) ) DX_u( i )  = DX_u( i ) / trans%X_scale( i )
            END IF
          END IF
        END DO

        Z = Z * trans%X_scale

!  Scale the bounds on Ax and the associated Lagrange multipliers

        DO i = 1, m
          IF ( C_l( i ) == C_u( i ) ) THEN
            C_l( i ) = C_l( i ) * trans%C_scale( i )
            C_u( i ) = C_u( i ) * trans%C_scale( i )
          ELSE 
            IF ( C_l( i ) > - control%infinity ) THEN
              C_l( i )  = C_l( i ) * trans%C_scale( i )
              IF ( PRESENT( DC_l ) ) DC_l( i ) = DC_l( i ) * trans%C_scale( i )
            END IF
            IF  ( C_u( i ) < control%infinity ) THEN 
              C_u( i ) = C_u( i ) * trans%C_scale( i )
              IF ( PRESENT( DC_u ) ) DC_u( i ) = DC_u( i ) * trans%C_scale( i )
            END IF
          END IF
        END DO

        Y = Y / trans%C_scale
        
! ==================
!  Unscale the data
! ==================

      ELSE

!  Unscale H

        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' ) 
          DO i = 1, n
            val = H%val( i )
            IF ( val /= zero )                                                 &
              H%val( i ) = val / ( trans%X_scale( i ) * trans%X_scale( i ) )
          END DO
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, n
            DO j = 1, i
              l = l + 1
              val = H%val( l )
              IF ( val /= zero )                                               &
                H%val( l ) = val / ( trans%X_scale( i ) * trans%X_scale( j ) )
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, n
            DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
              val = H%val( l )
              IF ( val /= zero ) THEN
                j = H%col( l )
                IF ( j >= 1 .AND. j <= n )                                     &
                  H%val( l ) = val / ( trans%X_scale( i ) * trans%X_scale( j ) )
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, H%ne
            val = H%val( l )
            IF ( val /= zero ) THEN
              i = H%row( l ) ; j = H%col( l )
              IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n )                   &
                H%val( l ) = val / ( trans%X_scale( i ) * trans%X_scale( j ) )
            END IF
          END DO
        END SELECT

!  Unscale A

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1
              val = A%val( l )
              IF ( val /= zero )                                               &
                A%val( l ) = val / ( trans%C_scale( i ) * trans%X_scale( j ) )
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                j = A%col( l )
                IF ( j >= 1 .AND. j <= n )                                     &
                  A%val( l ) = val / ( trans%C_scale( i ) * trans%X_scale( j ) )
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            val = A%val( l )
            IF ( val /= zero ) THEN
              i = A%row( l ) ; j = A%col( l )
              IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n )                &
                A%val( l ) = val / ( trans%C_scale( i ) * trans%X_scale( j ) )
            END IF
          END DO
        END SELECT

!  Unscale X and G

        X = X * trans%X_scale ; G = G / trans%X_scale
        IF ( PRESENT( DG ) ) DG = DG / trans%X_scale
        C = C / trans%C_scale

!  Unscale the bounds on X and the associated dual variables

        DO i = 1, n
          IF ( X_l( i ) == X_u( i ) ) THEN
            X_l( i ) = X_l( i ) * trans%X_scale( i )
            X_u( i ) = X_u( i ) * trans%X_scale( i )
          ELSE 
            IF ( X_l( i ) > - control%infinity ) THEN
              X_l( i )  = X_l( i ) * trans%X_scale( i )
              IF ( PRESENT( DX_l ) ) DX_l( i )  = DX_l( i ) * trans%X_scale( i )
            END IF
            IF  ( X_u( i ) < control%infinity ) THEN 
              X_u( i ) = X_u( i ) * trans%X_scale( i )
              IF ( PRESENT( DX_u ) ) DX_u( i )  = DX_u( i ) * trans%X_scale( i )
            END IF
          END IF
        END DO

        Z = Z / trans%X_scale

!  Unscale the bounds on Ax and the associated Lagrange multipliers

        DO i = 1, m
          IF ( C_l( i ) == C_u( i ) ) THEN
            C_l( i ) = C_l( i ) / trans%C_scale( i )
            C_u( i ) = C_u( i ) / trans%C_scale( i )
          ELSE 
            IF ( C_l( i ) > - control%infinity ) THEN
              C_l( i )  = C_l( i ) / trans%C_scale( i )
              IF ( PRESENT( DC_l ) ) DC_l( i ) = DC_l( i ) / trans%C_scale( i )
            END IF
            IF  ( C_u( i ) < control%infinity ) THEN 
              C_u( i ) = C_u( i ) / trans%C_scale( i )
              IF ( PRESENT( DC_u ) ) DC_u( i ) = DC_u( i ) / trans%C_scale( i )
            END IF
          END IF
        END DO

        Y = Y * trans%C_scale

      END IF
      RETURN

!  End of SCALE_apply_factors

      END SUBROUTINE SCALE_apply_factors

!  -  S C A L E _ g e t _ s h i f t _ a n d _ s c a l e  S U B R O U T I N E  -

!     SUBROUTINE SCALE_get_shift_and_scale( n, m, A, G, f, X, X_l, X_u, C,     &
      SUBROUTINE SCALE_get_shift_and_scale( n, m, A, G, X, X_l, X_u, C,        &
                                            C_l, C_u, trans, control, inform )

!  ---------------------------------------------------------------------------
!
!  Suppose that x_t = X_s^-1 ( x - x_s )
!               f_t( x_t ) = F_s^-1 ( q( x ) - f_s )
!          and  A_t x_t = C_s^-1 ( A x - c_s )
!
!  Compute suitable shifts (x_s,f_s) and scale factors (X_s,F_s,C_s)
!  for the quadratic programming (QP) problem
!
!      min f(x) = 1/2 x^T H x + x^T g + f
!
!      s.t.        c_l <= A x <= c_u, 
!      and         x_l <=  x  <= x_u
!
!  (or optionally to the parametric problem
!
!      min  1/2 x^T H x + x^T g + theta x^T dg + f + theta df
!
!      s.t. c_l + theta dc_l <= A x <= c_u + theta dc_u,
!      and  x_l + theta dx_l <=  x  <= x_u + theta dx_u )
!
!  ---------------------------------------------------------------------------

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m, n
!     REAL ( KIND = wp ), INTENT( IN ) :: f
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u, G
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C, C_l, C_u
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SCALE_trans_type ), INTENT( INOUT ) :: trans
      TYPE ( SCALE_control_type ), INTENT( IN ) :: control
      TYPE ( SCALE_inform_type ), INTENT( INOUT ) :: inform

!  local variables

      INTEGER :: i, j, l, out
      REAL ( KIND = wp ) :: val
      LOGICAL :: printi, printd

      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Ensure that input parameters are within allowed ranges

      IF ( n <= 0 .OR. m < 0 .OR. .NOT. QPT_keyword_A( A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 800
      END IF

      out = control%out
      printi = out > 0 .AND. control%print_level > 0
      printd = out > 0 .AND. control%print_level > 5

!  allocate space for scale factors

      array_name = 'scale: trans%X_scale'
      CALL SPACE_resize_array( n, trans%X_scale, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: trans%X_shift'
      CALL SPACE_resize_array( n, trans%X_shift, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: trans%C_scale'
      CALL SPACE_resize_array( m, trans%C_scale, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'scale: trans%C_shift'
      CALL SPACE_resize_array( m, trans%C_shift, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  set intial values for the scale and shift factors

      trans%f_scale = 1.0_wp ; trans%f_shift = 0.0_wp
      trans%X_scale = 1.0_wp ; trans%X_shift = 0.0_wp
      trans%C_scale = 1.0_wp ; trans%C_shift = 0.0_wp

!  scale and/or shift the variables

      IF ( control%shift_x > 0 .OR. control%scale_x > 0 ) THEN
        DO i = 1, n
          IF ( X_l( i ) < X_u( i ) ) THEN
            IF ( X_u( i ) < control%infinity ) THEN
              IF ( X_l( i ) > - control%infinity ) THEN
                IF ( control%shift_x > 0 )                                     &
                  trans%X_shift( i ) = half * ( X_u( i ) + X_l( i ) )
                IF ( control%scale_x > 0 ) trans%X_scale( i )                  &
                  = MAX( control%scale_x_min, half * ( X_u( i ) - X_l( i ) ) )
              ELSE
                IF ( control%shift_x > 0 )                                     &
                  trans%X_shift( i ) = X_u( i )
                IF ( control%scale_x > 0 ) trans%X_scale( i )                  &
                  = MAX( control%scale_x_min, X_u( i ) - X( i ) )
              END IF
            ELSE IF ( X_l( i ) > - control%infinity ) THEN
              IF ( control%shift_x > 0 )                                       &
                trans%X_shift( i ) = X_l( i )
              IF ( control%scale_x > 0 ) trans%X_scale( i )                    &
                = MAX( control%scale_x_min, X( i ) - X_l( i ) )
            END IF
          END IF
        END DO

        IF ( printd ) THEN
          WRITE( out, "( A, '  shift_x ', /, ( 3ES22.14 ) )" ) prefix,         &
            trans%X_shift( 1 : n )
          WRITE( out, "( A, '  scale_x ', /, ( 3ES22.14 ) )" ) prefix,         &
            trans%X_scale( 1 : n )
        ELSE IF ( printi ) THEN
          WRITE( out, "( A, '  max shift_x ', /, ES22.14 )" ) prefix,          &
            MAXVAL( ABS( trans%X_shift( 1 : n ) ) )
          WRITE( out, "( A, '  max scale_x ', /, ES22.14 )" ) prefix,          &
            MAXVAL( ABS( trans%X_scale( 1 : n ) ) )
        END IF
      END IF

!  if the variables have been shifted, make sure that the shift
!  is reflected in a shift in c

      IF ( control%shift_x > 0 ) THEN
        trans%C_shift( : m ) = zero

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1
              val = A%val( l )
              IF ( val /= zero ) trans%C_shift( i ) = trans%C_shift( i ) +     &
                  val * trans%X_shift( j ) 
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                j = A%col( l )
                IF ( j >= 1 .AND. j <= n ) trans%C_shift( i ) =                &
                   trans%C_shift( i ) + val * trans%X_shift( j ) 
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            val = A%val( l )
            IF ( val /= zero ) THEN
              i = A%row( l ) ; j = A%col( l )
              IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n )                &
                trans%C_shift( i ) = trans%C_shift( i ) +                      &
                  val * trans%X_shift( j ) 
            END IF
          END DO
        END SELECT

        IF ( printd ) THEN
          WRITE( out, "( A, '  shift_c ', /, ( 3ES22.14 ) )" ) prefix,         &
            trans%C_shift( 1 : m )
        ELSE IF ( printi ) THEN
          WRITE( out, "( A, '  max shift_c ', /, ES22.14 )" ) prefix,          &
            MAXVAL( ABS( trans%C_shift( 1 : m ) ) )
        END IF
      END IF

!  scale the constraints

      IF ( control%scale_c > 0 ) THEN

!  scale and shift so that shifts try to make c of O(1)

        IF ( control%scale_c == 2 ) THEN
          DO i = 1, m
            IF ( C_l( i ) < C_u( i ) ) THEN
              IF ( C_u( i ) < control%infinity ) THEN
                IF ( C_l( i ) > - control%infinity ) THEN
                  trans%C_scale( i ) = MAX( control%scale_c_min,               &
                                            half * ( C_u( i ) - C_l( i ) ) )
                ELSE
                  trans%C_scale( i ) = MAX( control%scale_c_min,               &
                                            ABS( C_u( i ) - C( i ) ) )
                END IF
              ELSE IF ( C_l( i ) > - control%infinity ) THEN
                trans%C_scale( i ) = MAX( control%scale_c_min,                 &
                                           ABS( C( i ) - C_l( i ) ) )
              END IF
            END IF
          END DO

!  scale and shift so that shifts try to make O(1) changes to x make O(1)
!  changes to c, using the (scaled) infinity norms of the gradients of 
!  the constraints.

        ELSE

          DO i = 1, m
            trans%C_scale( i ) = one
            IF ( C_u( i ) < control%infinity )                                 &
              trans%C_scale( i ) = MAX( trans%C_scale( i ), ABS( C_u( i ) ) ) 
            IF ( C_l( i ) > - control%infinity )                               &
              trans%C_scale( i ) = MAX( trans%C_scale( i ), ABS( C_l( i ) ) )
          END DO

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' ) 
            l = 0
            DO i = 1, m
              DO j = 1, n
                l = l + 1
                val = ABS( A%val( l ) )
                IF ( val /= zero ) THEN
                  IF ( control%scale_x > 0 ) THEN
                    trans%C_scale( i ) = MAX( trans%C_scale( i ),              &
                      ABS( trans%X_scale( j ) * val ) )
                  ELSE
                    trans%C_scale( i ) = MAX( trans%C_scale( i ), val )
                  END IF
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, m
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                val = ABS( A%val( l ) )
                IF ( val /= zero ) THEN
                  j = A%col( l )
                  IF ( j >= 1 .AND. j <= n ) THEN
                    IF ( control%scale_x > 0 ) THEN
                      trans%C_scale( i ) = MAX( trans%C_scale( i ),            &
                        ABS( trans%X_scale( j ) * val ) )
                    ELSE
                      trans%C_scale( i ) = MAX( trans%C_scale( i ), val )
                    END IF
                  END IF
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              val = ABS( A%val( l ) )
              IF ( val /= zero ) THEN
                i = A%row( l ) ; j = A%col( l )
                IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n ) THEN
                  IF ( control%scale_x > 0 ) THEN
                    trans%C_scale( i ) = MAX( trans%C_scale( i ),              &
                      ABS( trans%X_scale( j ) * val ) )
                  ELSE
                    trans%C_scale( i ) = MAX( trans%C_scale( i ), val )
                  END IF
                END IF
              END IF
            END DO
          END SELECT

        END IF

        IF ( printd ) THEN
          WRITE( out, "( A, '  scale_c ', /, ( 3ES22.14 ) )" ) prefix,         &
            trans%C_scale( 1 : m )
        ELSE IF ( printi ) THEN
          WRITE( out, "( A, '  max scale_c ', /, ES22.14 )" ) prefix,          &
            MAXVAL( ABS( trans%C_scale( 1 : m ) ) )
        END IF
!         WRITE( out, "( A, '  scale_c ', /, ( 3ES22.14 ) )" ) prefix,         &
!           trans%C_scale( 1 : m )
      END IF

!  scale the objective

      IF ( control%scale_f > 1 ) THEN

!  scale and shift so that shifts try to make f of O(1)

!       trans%f_shift = f
        IF ( control%scale_f == 2 ) THEN
          trans%f_scale = one

!  scale and shift so that shifts try to make O(1) changes to x make O(1)
!  changes to f, using the (scaled) infinity norm of the gradients of 
!  the objective

        ELSE
          IF ( control%scale_x > 0 ) THEN
            DO i = 1, n
              trans%f_scale = MAX( trans%f_scale,                              &
                                   ABS( trans%X_scale( i ) * G( i ) ) )
            END DO
          ELSE
            DO i = 1, n
              trans%f_scale = MAX( trans%f_scale, ABS( G( i ) ) )
            END DO
          END IF
        END IF
        IF ( printi ) THEN
          WRITE( out, "( A, '  shift_f ', /, ES22.14 )" ) prefix, trans%f_shift
          WRITE( out, "( A, '  scale_f ', /, ES22.14 )" ) prefix, trans%f_scale
        END IF
      END IF
      RETURN

!  error returns

  800 CONTINUE 
      IF ( control%error > 0 .AND. control%print_level >= 0 )                  &
        WRITE( control%error,                                                  &
        "( ' * Error return from SCALE_get_shift_and_scale * status = ', I0 )")&
          inform%status
      RETURN

!  allocation error

  900 CONTINUE 
      inform%status = GALAHAD_error_allocate
      IF ( control%out > 0 .AND. control%print_level > 0 ) WRITE( control%out, &
        "( A, ' ** Message from -SCALE_get_factors-', /,  A,                   &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status
      RETURN  

!  End of SCALE_get_shift_and_scale

      END SUBROUTINE SCALE_get_shift_and_scale

! - Q T R A N S _ a p p l y _ s h i f t _ a n d _ s c a l e  S U B R O U T I N E

      SUBROUTINE SCALE_apply_shift_and_scale( n, m, H, A, f, G,                &
                                              X, X_l, X_u, C, C_l, C_u, Y, Z,  &
                                              apply, trans, data, control,     &
                                              inform, df, DG, DX_l, DX_u,      &
                                              DC_l, DC_u )

!  ----------------------------------------------------------------------------
!  Apply the shifts (x_s,f_s) and scale factors (X_s,F_s,C_s) computed by
!  SCALE_get_shift_and_scale to the data for the quadratic programming 
!  (QP) problem
!
!      min f(x) = 1/2 x^T H x + x^T g + f
!
!      s.t.        c_l <= A x <= c_u, 
!      and         x_l <=  x  <= x_u
!
!  (or optionally to the parametric problem
!
!      min  1/2 x^T H x + x^T g + theta x^T dg + f + theta df
!
!      s.t. c_l + theta dc_l <= A x <= c_u + theta dc_u,
!      and  x_l + theta dx_l <=  x  <= x_u + theta dx_u )
!
!  to derive the transformed problem
!
!      min f_t(x_t) = 1/2 x_t^T H_t x_t + x_t^T g_t + f_t
!
!      s.t.           c_t_l <= A_t x_t <= c_t_u, 
!                     x_t_l <=   x_t   <= x_t_u
!
!  (or optionally for the parametric problem
!
!      min  1/2 x_t^T H_t x_t + x_t^T g_t + theta x_t^T dg_t + f_t + theta df_t
!
!      s.t. c_t_l + theta dc_t_l <= A_t x_t <= c_t_u + theta dc_t_u,
!      and  x_t_l + theta dx_t_l <=    x_t  <= x_t_u + theta dx_t_u )
!
!  where H_t = X_s^T H X_s / F_s
!        g_t = X_s ( H x_s + g ) / F_s
!        dg_t = X_s dg / F_s
!        f_t = 1/2 x_s^T H x_s + x_s^T g + f - f_s ) / F_s
!        df_t = x_s^T dg / F_s
!        A_t = C_s^-1 A X_s
!        c_s = A x_s
!        c_t_l = C_s^-1 ( c_l - c_s )
!        dc_t_l = C_s^-1 dc_l
!        c_t_u = C_s^-1 ( c_u - c_s )
!        dc_t_u = C_s^-1 d_u
!        x_t_l = X_s^-1 ( c_l - x_s )
!        dx_t_l = X_s^-1 d_l
!        x_t_u = X_s^-1 ( c_u - x_s )
!  and   dx_t_u = X_s^-1 dc_u
!
!  ---------------------------------------------------------------------------

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m, n
      LOGICAL, INTENT( IN ) :: apply
      REAL ( KIND = wp ), INTENT( INOUT ) :: f
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: G, X, X_l, X_u, Z
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C, C_l, C_u, Y
      REAL ( KIND = wp ), OPTIONAL :: df
      REAL ( KIND = wp ), OPTIONAL, DIMENSION( n ) :: DG, DX_l, DX_u
      REAL ( KIND = wp ), OPTIONAL, DIMENSION( m ) :: DC_l, DC_u
      TYPE ( SMT_type ), INTENT( INOUT ) :: H, A
      TYPE ( SCALE_trans_type ), INTENT( IN ) :: trans
      TYPE ( SCALE_data_type ), INTENT( INOUT ) :: data
      TYPE ( SCALE_control_type ), INTENT( IN ) :: control
      TYPE ( SCALE_inform_type ), INTENT( INOUT ) :: inform

!  local variables

      INTEGER :: i, j, l
      REAL ( KIND = wp ) :: val
      CHARACTER ( LEN = 80 ) :: array_name

! ================
!  Scale the data
! ================

      IF ( apply ) THEN

!  Compute H x_s

        array_name = 'scale: data%PROD'
        CALL SPACE_resize_array( n, data%PROD, inform%status,                  &
                      inform%alloc_status, exact_size = .TRUE.,                &
                      deallocate_error_fatal = control%deallocate_error_fatal, &
                      array_name = array_name, bad_alloc = inform%bad_alloc,   &
                      out = control%out )
        IF ( inform%status /= 0 ) RETURN

        data%PROD( : n ) = zero
        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' ) 
          DO i = 1, n
            val = H%val( i )
            IF ( val /= zero )                                                 &
              data%PROD( i ) = data%PROD( i ) + val * trans%X_shift( i )
          END DO
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, n
            DO j = 1, i
              l = l + 1
              val = H%val( l )
              IF ( val /= zero ) THEN
                data%PROD( i ) = data%PROD( i ) + val * trans%X_shift( j )
                IF ( i /= j )                                                  &
                  data%PROD( j ) = data%PROD( j ) + val * trans%X_shift( i )
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, n
            DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
              val = H%val( l )
              IF ( val /= zero ) THEN
                j = H%col( l )
                IF ( j >= 1 .AND. j <= n ) THEN
                  data%PROD( i ) = data%PROD( i ) + val * trans%X_shift( j )
                  IF ( i /= j )                                                &
                    data%PROD( j ) = data%PROD( j ) + val * trans%X_shift( i )
                END IF
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, H%ne
            val = H%val( l )
            IF ( val /= zero ) THEN
              i = H%row( l ) ; j = H%col( l )
              IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
                data%PROD( i ) = data%PROD( i ) + val * trans%X_shift( j )
                IF ( i /= j )                                                  &
                  data%PROD( j ) = data%PROD( j ) + val * trans%X_shift( i )
              END IF
            END IF
          END DO
        END SELECT

!  Compute f <- 1/2 x_s^T H x_s + x_s^T g + f - f_s ) / F_s

        f = ( half * DOT_PRODUCT( data%PROD( : n ), trans%X_shift ) +          &
              DOT_PRODUCT( G, trans%X_shift ) + f                              &
              - trans%f_shift ) / trans%f_scale

!  Compute g <- X_s ( H x_s + G ) / F_s
  
        G = trans%X_scale * ( data%PROD( : n ) + G ) / trans%f_scale

!  Compute df <- x_s^T dg / F_s

        IF ( PRESENT( df ) .AND. PRESENT( DG ) )                               &
          df = DOT_PRODUCT( DG, trans%X_shift ) / trans%f_scale

!  Compute dg <- X_s dg / F_s

        IF ( PRESENT( DG ) ) DG = trans%X_scale * DG / trans%f_scale

!  Compute H <- X_s^T H X_s / F_s

        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' ) 
          DO i = 1, n
            val = H%val( i )
            IF ( val /= zero ) H%val( i ) =                                    &
               val * ( trans%X_scale( i ) ** 2 / trans%f_scale ) 
          END DO
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, n
            DO j = 1, i
              l = l + 1
              val = H%val( l )
              IF ( val /= zero ) H%val( l ) = val *                            &
                ( trans%X_scale( i ) * trans%X_scale( j ) / trans%f_scale ) 
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, n
            DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
              val = H%val( l )
              IF ( val /= zero ) THEN
                j = H%col( l )
                IF ( j >= 1 .AND. j <= n ) H%val( l ) = val *                  &
                  ( trans%X_scale( i ) * trans%X_scale( j ) / trans%f_scale ) 
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, H%ne
            val = H%val( l )
            IF ( val /= zero ) THEN
              i = H%row( l ) ; j = H%col( l )
              IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n )                   &
                  H%val( l ) = val * ( trans%X_scale( i ) *                    &
                    trans%X_scale( j ) / trans%f_scale ) 
            END IF
          END DO
        END SELECT

!  Compute A <- C_s^-1 A X_s

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1
              val = A%val( l )
              IF ( val /= zero )                                               &
                A%val( l ) = val * ( trans%X_scale( j ) / trans%C_scale( i ) )
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                j = A%col( l )
                IF ( j >= 1 .AND. j <= n )                                     &
                  A%val( l ) = val * ( trans%X_scale( j ) / trans%C_scale( i ) )
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            val = A%val( l )
            IF ( val /= zero ) THEN
              i = A%row( l ) ; j = A%col( l )
              IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n )                &
                A%val( l ) = val * ( trans%X_scale( j ) / trans%C_scale( i ) )
            END IF
          END DO
        END SELECT

!  Compute c_l <- C_s^-1 ( c_l - c_s )

        CALL TRANS_v_trans_inplace( m, trans%C_scale, trans%C_shift, C_l,      &
                                 lower = .TRUE., infinity = control%infinity )

!  Compute dc_l <- C_s^-1 dc_l

        IF ( PRESENT( DC_l ) )                                                 &
          WHERE ( DC_l >  - control%infinity ) DC_l = DC_l / trans%C_scale

!  Compute c_u <- C_s^-1 ( c_u - c_s )

        CALL TRANS_v_trans_inplace( m, trans%C_scale, trans%C_shift, C_u,      &
                                 lower = .FALSE., infinity = control%infinity )

!  Compute dc_u <- C_s^-1 d_u

        IF ( PRESENT( DC_u ) )                                                 &
          WHERE ( DC_u <  control%infinity ) DC_u = DC_u / trans%C_scale

!  Compute x <- X_s^-1 ( x - x_s )

        CALL TRANS_v_trans_inplace( n, trans%X_scale, trans%X_shift, X )

!  Compute x_l <- X_s^-1 ( x_l - x_s )

        CALL TRANS_v_trans_inplace( n, trans%X_scale, trans%X_shift, X_l,      &
                                 lower = .TRUE., infinity = control%infinity )

!  Compute dx_l <- X_s^-1 d_l

        IF ( PRESENT( DX_l ) )                                                 &
          WHERE ( DX_l >  - control%infinity ) DX_l = DX_l / trans%X_scale

!  Compute x_u <- X_s^-1 ( x_u - x_s )

        CALL TRANS_v_trans_inplace( n, trans%X_scale, trans%X_shift, X_u,      &
                                 lower = .FALSE., infinity = control%infinity )

!  Compute dx_u <- X_s^-1 dc_u

        IF ( PRESENT( DX_u ) )                                                 &
          WHERE ( DX_u <  control%infinity ) DX_u = DX_u / trans%X_scale

! ==================
!  Unscale the data
! ==================

      ELSE

!  Compute dx_u <- X_s dc_u

        IF ( PRESENT( DX_u ) )                                                 &
          WHERE ( DX_u <  control%infinity ) DX_u = DX_u * trans%X_scale

!  Compute x_u <- X_s x_u + x_s

        CALL TRANS_v_untrans_inplace( n, trans%X_scale, trans%X_shift, X_u,    &
                                 lower = .FALSE., infinity = control%infinity )

!  Compute dx_l <- X_s d_l

        IF ( PRESENT( DX_l ) )                                                 &
          WHERE ( DX_l >  - control%infinity ) DX_l = DX_l * trans%X_scale

!  Compute x_l <- X_s x_l + x_s

        CALL TRANS_v_untrans_inplace( n, trans%X_scale, trans%X_shift, X_l,    &
                                 lower = .TRUE., infinity = control%infinity )

!  Compute x <- X_s x + x_s

        CALL TRANS_v_untrans_inplace( n, trans%X_scale, trans%X_shift, X )

!  Compute dc_u <- C_s d_u

        IF ( PRESENT( DC_u ) )                                                 &
          WHERE ( DC_u <  control%infinity ) DC_u = DC_u * trans%C_scale

!  Compute c_u <- C_s c_u + c_s

        CALL TRANS_v_untrans_inplace( m, trans%C_scale, trans%C_shift, C_u,    &
                                 lower = .FALSE., infinity = control%infinity )

!  Compute dc_l <- C_s dc_l

        IF ( PRESENT( DC_l ) )                                                 &
          WHERE ( DC_l >  - control%infinity ) DC_l = DC_l * trans%C_scale

!  Compute c_l <- C_s c_l + c_s

        CALL TRANS_v_untrans_inplace( m, trans%C_scale, trans%C_shift, C_l,    &
                                 lower = .TRUE., infinity = control%infinity )

!  Compute c <- C_s c + c_s

        CALL TRANS_v_untrans_inplace( m, trans%C_scale, trans%C_shift, C,      &
                                 lower = .TRUE., infinity = control%infinity )

!  Compute A <- C_s A X_s^-1

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1
              val = A%val( l )
              IF ( val /= zero )                                               &
                A%val( l ) = val / ( trans%X_scale( j ) / trans%C_scale( i ) )
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                j = A%col( l )
                IF ( j >= 1 .AND. j <= n )                                     &
                  A%val( l ) = val / ( trans%X_scale( j ) / trans%C_scale( i ) )
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            val = A%val( l )
            IF ( val /= zero ) THEN
              i = A%row( l ) ; j = A%col( l )
              IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n )                &
                A%val( l ) = val / ( trans%X_scale( j ) / trans%C_scale( i ) )
            END IF
          END DO
        END SELECT

!  Compute H <- X_s^-T H X_s^-1 * F_s

        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' ) 
          DO i = 1, n
            val = H%val( i )
            IF ( val /= zero ) H%val( i ) =                                    &
               val / ( trans%X_scale( i ) ** 2 / trans%f_scale ) 
          END DO
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, n
            DO j = 1, i
              l = l + 1
              val = H%val( l )
              IF ( val /= zero ) H%val( l ) = val /                            &
                ( trans%X_scale( i ) * trans%X_scale( j ) / trans%f_scale ) 
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, n
            DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
              val = H%val( l )
              IF ( val /= zero ) THEN
                j = H%col( l )
                IF ( j >= 1 .AND. j <= n ) H%val( l ) = val /                  &
                  ( trans%X_scale( i ) * trans%X_scale( j ) / trans%f_scale ) 

              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, H%ne
            val = H%val( l )
            IF ( val /= zero ) THEN
              i = H%row( l ) ; j = H%col( l )
              IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n )                   &
                  H%val( l ) = val / ( trans%X_scale( i ) *                    &
                    trans%X_scale( j ) / trans%f_scale ) 
            END IF
          END DO
        END SELECT

!  Compute dg <- X_s^-1 dg * F_s

        IF ( PRESENT( DG ) ) DG = ( DG / trans%X_scale ) * trans%f_scale

!  Compute H x_s

        array_name = 'scale: data%PROD'
        CALL SPACE_resize_array( n, data%PROD, inform%status,                  &
                      inform%alloc_status, exact_size = .TRUE.,                &
                      deallocate_error_fatal = control%deallocate_error_fatal, &
                      array_name = array_name, bad_alloc = inform%bad_alloc,   &
                      out = control%out )
        IF ( inform%status /= 0 ) RETURN

        data%PROD( : n ) = zero
        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' ) 
          DO i = 1, n
            val = H%val( i )
            IF ( val /= zero )                                                 &
              data%PROD( i ) = data%PROD( i ) + val * trans%X_shift( i )
          END DO
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, n
            DO j = 1, i
              l = l + 1
              val = H%val( l )
              IF ( val /= zero ) THEN
                data%PROD( i ) = data%PROD( i ) + val * trans%X_shift( j )
                IF ( i /= j )                                                  &
                  data%PROD( j ) = data%PROD( j ) + val * trans%X_shift( i )
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, n
            DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
              val = H%val( l )
              IF ( val /= zero ) THEN
                j = H%col( l )
                IF ( j >= 1 .AND. j <= n ) THEN
                  data%PROD( i ) = data%PROD( i ) + val * trans%X_shift( j )
                  IF ( i /= j )                                                &
                    data%PROD( j ) = data%PROD( j ) + val * trans%X_shift( i )
                END IF
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, H%ne
            val = H%val( l )
            IF ( val /= zero ) THEN
              i = H%row( l ) ; j = H%col( l )
              IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
                data%PROD( i ) = data%PROD( i ) + val * trans%X_shift( j )
                IF ( i /= j )                                                  &
                  data%PROD( j ) = data%PROD( j ) + val * trans%X_shift( i )
              END IF
            END IF
          END DO
        END SELECT

!  Compute g <- X_s^{-1} ( G - H x_s  ) * F_s
  
        G = trans%f_scale * G / trans%X_scale - data%PROD( : n )

!  Compute y <- C_s^{-1} y
  
        Y = trans%f_scale * Y / trans%C_scale

!  Compute z <- X_s^{-1} z
  
        Z = trans%f_scale * Z / trans%X_scale

!  Compute df <- 0

        IF ( PRESENT( df ) .AND. PRESENT( DG ) ) df = zero

!  Compute f <- F_S * ( f + f_s - x_s^T g -  1/2 x_s^T H x_s )

        f = trans%f_scale * ( f + trans%f_shift                                &
             - DOT_PRODUCT( G, trans%X_shift( : n ) )                          &
             - half * DOT_PRODUCT( data%PROD( : n ), trans%X_shift( : n ) ) )

      END IF
      RETURN

!  End of SCALE_apply_shift_and_scale

      END SUBROUTINE SCALE_apply_shift_and_scale

!-*-*-*-*-*-   S C A L E _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE SCALE_terminate( data, control, inform, trans )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine SCALE_initialize
!   control see Subroutine SCALE_initialize
!   inform  see other SCALE subroutines

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( SCALE_data_type ), INTENT( INOUT ) :: data
      TYPE ( SCALE_control_type ), INTENT( IN ) :: control        
      TYPE ( SCALE_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SCALE_trans_type ), OPTIONAL :: trans
 
!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaing allocated arrays

      array_name = 'scale: data%ROW_val'
      CALL SPACE_dealloc_array( data%ROW_val,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'scale: data%COL_val'
      CALL SPACE_dealloc_array( data%COL_val,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'scale: data%RES'
      CALL SPACE_dealloc_array( data%RES,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'scale: data%P'
      CALL SPACE_dealloc_array( data%P,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'scale: data%PROD'
      CALL SPACE_dealloc_array( data%PROD,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  optionally, deallocate all allocated arrays from trans

      IF ( PRESENT( trans ) ) THEN

        array_name = 'scale: trans%X_scale'
        CALL SPACE_dealloc_array( trans%X_scale,                               &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'scale: trans%X_shift'
        CALL SPACE_dealloc_array( trans%X_shift,                               &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'scale: trans%C_scale'
        CALL SPACE_dealloc_array( trans%C_scale,                               &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'scale: trans%C_shift'
        CALL SPACE_dealloc_array( trans%C_shift,                               &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

      END IF
      RETURN

!  End of SCALE_terminate

      END SUBROUTINE SCALE_terminate

!  End of module SCALE

   END MODULE GALAHAD_SCALE_double



