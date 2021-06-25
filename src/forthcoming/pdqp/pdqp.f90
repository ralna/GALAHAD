! THIS VERSION: GALAHAD 3.3 - 14/04/2021 AT 08:30 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ P D Q P   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started August 21st 2009
!   originally released GALAHAD Version 2.4. August 22nd 2009 as QPE
!   renamed as PDQP, GALAHAD Version 3.3, April 14th 2021

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_PDQP_double

!     -------------------------------------------
!     | Solve the convex quadratic program      |
!     |                                         |
!     |    minimize   1/2 x(T) H x + g(T) x + f |
!     |    subject to   c_l <= A x  <= c_u      |
!     |    and          x_l <=  x   <= x_u      |
!     |                                         |
!     | where H is an M-matrix using a          |
!     | primal-dual active-set method           |
!     -------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SORT_double
      USE GALAHAD_SPECFILE_double 
      USE GALAHAD_SBLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: PDQP_initialize, PDQP_read_specfile, PDQP_solve,               &
                PDQP_terminate, QPT_problem_type

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
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: PDQP_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!   the maximum number of iterations permitted

        INTEGER :: maxit = 100

!   the maximum number of variables that are will initially be fixed away
!   from their bounds (<0 says that as many as possible will be picked)

        INTEGER :: temporarily_fixed = - 1

!  variable bounds larger than infinity are infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!  primal violations and dual variable that smaller in absolute value than 
!  var_small will be set to zero

        REAL ( KIND = wp ) :: var_small = ten ** ( - 15 )

!  use the initial status provided in X_stat and C_stat if they are present

        LOGICAL :: initial_status_provided = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  all output lines will be prefixed by 
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes, 
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for SBLS

        TYPE ( SBLS_control_type ) :: SBLS_control
      END TYPE

      TYPE, PUBLIC :: PDQP_inform_type
        INTEGER :: status = 0
        INTEGER :: alloc_status = 0
        INTEGER :: iter = - 1
        REAL :: time = 0.0
        REAL ( KIND = wp ):: obj = HUGE( one ) 
        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )
        TYPE ( SBLS_inform_type ) :: SBLS_inform
      END TYPE

      TYPE, PUBLIC :: PDQP_data_type
        TYPE ( SMT_type ) :: A
        TYPE ( SMT_type ) :: H
        TYPE ( SMT_type ) :: A_free
        TYPE ( SMT_type ) :: H_free
        TYPE ( SMT_type ) :: C_null
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: STATE
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: STATE_old
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_l
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_u
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_free
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B_free
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
        TYPE ( SBLS_data_type ) :: SBLS_data
      END TYPE

   CONTAINS

!-*-*-*-*-*-   P D Q P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE PDQP_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for PDQP. This routine should be called before
!  PDQP_solve
! 
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data is a structure of type PDQP_data_type. On output, 
!   pointer array components will have been nullified.
!
!  control is a structure of type PDQP_control_type that contains
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
!   deallocate_error_fatal. If true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( PDQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( PDQP_control_type ), INTENT( INOUT ) :: control
      TYPE ( PDQP_inform_type ), INTENT( OUT ) :: inform

     inform%status = GALAHAD_ok

!  initialize control parameters for SBLS (see GALAHAD_SBLS for details)

       CALL SBLS_initialize( data%sbls_data, control%sbls_control,             &
                             inform%sbls_inform )

!  End of PDQP_initialize

      END SUBROUTINE PDQP_initialize

!-*-*-*-*-   P D Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE PDQP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by PDQP_initialize could (roughly) 
!  have been set as:

! BEGIN PDQP SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  maximum-number-of-iterations                      100
!  maximum-number-of-initially-free-variables        -1
!  infinity-value                                    1.0D+19
!  small-variable-tolerance                          1.0D-15
!  initial-status-provided                           F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END PDQP SPECIFICATIONS

!  Dummy arguments

      TYPE ( PDQP_control_type ), INTENT( INOUT ) :: control        
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: maxit = print_level + 1
      INTEGER, PARAMETER :: temporarily_fixed = maxit + 1
      INTEGER, PARAMETER :: infinity = temporarily_fixed + 1
      INTEGER, PARAMETER :: var_small = infinity + 1
      INTEGER, PARAMETER :: initial_status_provided = var_small + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = initial_status_provided + 1
      INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 16 ), PARAMETER :: specname = 'PDQP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level' 
      spec( maxit )%keyword = 'maximum-number-of-iterations'
      spec( temporarily_fixed )%keyword =                                      &
        'maximum-number-of-initially-free-variables'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( var_small )%keyword = 'small-variable-tolerance'

!  Logical key-words

     spec( initial_status_provided )%keyword = 'initial-status-provided'
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

      CALL SPECFILE_assign_value( spec( error ),                               &
                                  control%error,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( out ),                                 &
                                  control%out,                                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_level ),                         &
                                  control%print_level,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( maxit ),                               &
                                  control%maxit,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( temporarily_fixed ),                   &
                                  control%temporarily_fixed,                   &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_real( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
      CALL SPECFILE_assign_real( spec( var_small ),                            &
                                 control%var_small,                            &
                                 control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( initial_status_provided ),             &
                                  control%initial_status_provided,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

!  Read the specfile for SBLS

      CALL SBLS_read_specfile( control%SBLS_control, device )

      RETURN

      END SUBROUTINE PDQP_read_specfile

!-*-*-*-*-*-*-*-*-   P D Q P _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE PDQP_solve( prob, data, control, inform, X_stat, C_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     subject to    (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!        and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ), const is a
!  constant, g is an n-vector, H is a symmetric, posistive-definite M-matrix, 
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
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!                 
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!                 
!   %H_* is used to hold the LOWER TRIANGULAR part of H.
!   Three storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       %H%val( : )  the values of the components of H
!       %H%row( : )  the row indices of the components of H
!       %H%col( : )  the column indices of the components of H
!       %H%ne        the number of nonzeros used to store 
!                    the LOWER TRIANGULAR part of H
!
!       In addition, the array
!
!       %H%ptr( : )   must be of length %n + 1 
!
!       but need not be set
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       %H%val( : )  the values of the components of H, stored row by row
!       %H%col( : )  the column indices of the components of H
!       %H%ptr( : )  pointers to the start of each row, and past the end of
!                    the last row
!       %H%ne    = - 1
!
!       In addition, the array
!
!       %H%row( : )   must be of length >= 0
!
!       but need not be set
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       %H%val( : )  the values of the components of H, stored row by row,
!                    with each the entries in each row in order of 
!                    increasing column indicies.
!       %H%ne    = - 2
!
!       In addition, the arrays
!
!       %H%row( : )   must be of length >= 0
!       %H%col( : )   must be of length %n * ( %n + 1 ) / 2
!       %H%ptr( : )   must be of length %n + 1
!
!       but need not be set
!
!   %G is a REAL array of length %n, which must be set by
!    the user to the value of the gradient, g, of the linear term of the
!    quadratic objective function. The i-th component of G, i = 1, ....,
!    n, should contain the value of g_i.  
!   
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function.
!
!   %A_* is used to hold the matrix A. Three storage formats
!    are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       %A%val( : )   the values of the components of A
!       %A%row( : )   the row indices of the components of A
!       %A%col( : )   the column indices of the components of A
!       %A%ne         the number of nonzeros used to store A
!
!       In addition, the array
!
!       %A%ptr( : )   must be of length %m + 1 
!
!       but need not be set
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       %A%val( : )   the values of the components of A, stored row by row
!       %A%col( : )   the column indices of the components of A
!       %A%ptr( : )   pointers to the start of each row, and past the end of
!                     the last row
!       %A%ne    = -1
!
!       In addition, the array
!
!       %A%row( : )   must be of length >= 0
!
!       but need not be set
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       %A%val( : )   the values of the components of A, stored row by row,
!                     with each the entries in each row in order of 
!                     increasing column indicies.
!       %A%ne    = -2
!
!       In addition, the arrays
!
!       %A%row( : )   must be of length >= 0
!       %A%col( : )   must be of length %n * %m
!       %A%ptr( : )   must be of length %m + 1
!
!       but need not be set
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
!    control%infinity. 
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
!    control%infinity. 
!   
!   %Z is a REAL array of length %n, which must be set by the user to
!    appropriate estimates of the values of the dual variables 
!    (Lagrange multipliers corresponding to the simple bound constraints 
!    x_l <= x <= x_u). On successful exit, it will contain
!    the required vector of dual variables. 
!
!  control is a structure of type PDQP_control_type that contains
!   control parameters. See PDQP_initialize for details.
!
!  inform is a structure of type PDQP_inform_type that provides 
!    information on exit from PDQP_solve. The component status 
!    has possible values:
!  
!     0 Normal termination with the solution to the problem.
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
!  On exit from QPB_solve, other components of inform give the 
!  following:
!
!     alloc_status = The status of the last attempted allocation/deallocation 
!     time = the total time spent in the package.
!
!  X_stat and C_stat are optional INTEGER pointer arrays of length n 
!  and m (respectively) that contain the input and output status of the 
!  variables and constraints. The ith variable/constraint is on its lower bound
!  if X/C_stat(i) < 0, is free if X/C_stat(i) =0 and on its upper bound
!  if X/C_stat(i) > 0.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( PDQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( PDQP_control_type ), INTENT( IN ) :: control
      TYPE ( PDQP_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, OPTIONAL, ALLOCATABLE, DIMENSION( : ) :: X_stat, C_stat

!  Local variables

      INTEGER :: m, n, a_ne, h_ne, i, j, l, ll, ii, jj, alloc_status, n_infeas
      INTEGER :: n_orig, a_ne_orig, h_ne_orig, n_free, n_low, n_up
      INTEGER :: n_fixed, temporarily_fixed
      REAL :: time_start, time
      REAL ( KIND = wp ) :: cl, cu, x, xl, xu, z, val, infinity, rho
      LOGICAL :: stats
      CHARACTER ( LEN =  6 ) :: st
      CHARACTER ( LEN = 20 ) :: bad_alloc
      
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' entering PDQP_fomulate ' )" )

!  If required, write out problem 

!     IF ( control%out > 0 .AND. control%print_level >= 1 ) THEN
      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', I0, 1X, I0 )" ) prob%n, prob%m
        WRITE( control%out, "( ' f = ', ES12.4 )" ) prob%f
        WRITE( control%out, "( ' G = ', /, ( 5ES12.4 ) )" ) prob%G( : prob%n )
        IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
          WRITE( control%out, "( ' H (diagonal) = ', /, ( 5ES12.4 ) )" )       &
            prob%H%val( : prob%n )
        ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          WRITE( control%out, "( ' H (dense) = ', /, ( 5ES12.4 ) )" )          &
            prob%H%val( : prob%n * ( prob%n + 1 ) / 2 )
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( control%out, "( ' H (row-wise) = ' )" )
          DO i = 1, prob%m
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, prob%H%col( j ), prob%H%val( j ),                           &
                j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1 )
          END DO
        ELSE
          WRITE( control%out, "( ' H (co-ordinate) = ' )" )
          WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
          ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ), i = 1, prob%H%ne)
        END IF
        WRITE( control%out, "( ' X_l = ', /, ( 5ES12.4 ) )" )                  &
          prob%X_l( : prob%n )
        WRITE( control%out, "( ' X_u = ', /, ( 5ES12.4 ) )" )                  &
          prob%X_u( : prob%n )
        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          WRITE( control%out, "( ' A (dense) = ', /, ( 5ES12.4 ) )" )          &
            prob%A%val( : prob%n * prob%m )
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( control%out, "( ' A (row-wise) = ' )" )
          DO i = 1, prob%m
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, prob%A%col( j ), prob%A%val( j ),                           &
                j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1 )
          END DO
        ELSE
          WRITE( control%out, "( ' A (co-ordinate) = ' )" )
          WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
          ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne)
        END IF
        WRITE( control%out, "( ' C_l = ', /, ( 5ES12.4 ) )" )                  &
          prob%C_l( : prob%m )
        WRITE( control%out, "( ' C_u = ', /, ( 5ES12.4 ) )" )                  &
          prob%C_u( : prob%m )
      END IF

!  initialize time

      inform%time = 0.0
      CALL CPU_TIME( time_start )

!  ensure bounds on x are consistent

      DO i = 1, prob%n
        IF ( prob%x_l( i ) > prob%x_u( i ) ) THEN
          inform%status = GALAHAD_error_bad_bounds
          GO TO 800
        END IF
      END DO

!  compute how many variables and constraints there will be

      IF ( prob%n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 800
      ELSE
        n = prob%n
      END IF
      n_orig = n

      IF ( prob%m < 0 ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 800
      ELSE
        m = prob%m
      END IF

!  record the input array types

      IF ( m > 0 ) THEN
        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          a_ne = n * m
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          a_ne = prob%A%ptr( m + 1 ) - 1
        ELSE IF ( SMT_get( prob%A%type ) == 'COORDINATE' ) THEN
          a_ne = prob%A%ne
        ELSE
          inform%status = GALAHAD_error_restrictions
          GO TO 800
        END IF
      ELSE
        a_ne = 0
      END IF
      a_ne_orig = a_ne

      IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
        h_ne = ( n * ( n + 1 ) ) / 2
      ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
        h_ne = prob%H%ptr( n + 1 ) - 1
      ELSE IF ( SMT_get( prob%H%type ) == 'COORDINATE' ) THEN
        h_ne = prob%H%ne
      ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
        h_ne = prob%H%n
      ELSE
        inform%status = GALAHAD_error_restrictions
        GO TO 800
      END IF     
      h_ne_orig = h_ne

!  ensure bounds on A x are consistent

      DO i = 1, m
        cl = prob%c_l( i ) ; cu = prob%c_u( i )
        IF ( cu == cl ) THEN      !  equality constraint
        ELSE IF ( cl > - control%infinity ) THEN
          IF ( cu < control%infinity ) THEN 
            IF ( cl <= cl ) THEN  !  constraint bounded on both sides
              n = n + 1
              a_ne = a_ne + 1
            ELSE                  !  inconsistent constraint
              inform%status = GALAHAD_error_bad_bounds
              GO TO 800
            END IF
          ELSE                    !  constraint bounded from below
            n = n + 1
            a_ne = a_ne + 1
          END IF
        ELSE
          IF ( cu < control%infinity ) THEN !  constraint bounded from above
            n = n + 1
            a_ne = a_ne + 1
          ELSE                    !  free constraint
          END IF
        END IF
      END DO
!write(6,*) ' n_orig, n, m ', n_orig, n, m

!  record problem dimensions

      data%A%m = m
      data%A%n = n
      data%A%ne = a_ne
      data%H%n = n
      data%H%ne = h_ne
      data%C_null%n = 0
      data%C_null%ne = 0

!  compute how many variables will initially be "free" (i.e., fixed off bounds)

      temporarily_fixed = control%temporarily_fixed
      IF ( temporarily_fixed <= 0 ) THEN
        temporarily_fixed = MAX( 1, n - m )
      ELSE
        temporarily_fixed = MIN( temporarily_fixed, n - m )
      END IF

!  allocate space

      CALL SMT_put( data%A%type, 'COORDINATE', alloc_status )
      CALL SMT_put( data%H%type, 'COORDINATE', alloc_status )
      CALL SMT_put( data%A_free%type, 'COORDINATE', alloc_status )
      CALL SMT_put( data%H_free%type, 'COORDINATE', alloc_status )
      CALL SMT_put( data%C_null%type, 'COORDINATE', alloc_status )

      CALL SPACE_resize_array( a_ne, data%A%ROW,                               &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( a_ne, data%A%COL,                               &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( a_ne, data%A%VAL,                               &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( h_ne, data%H%ROW,                               &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( h_ne, data%H%COL,                               &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( h_ne, data%H%VAL,                               &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( a_ne, data%A_free%ROW,                          &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( a_ne, data%A_free%COL,                          &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( a_ne, data%A_free%VAL,                          &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( h_ne, data%H_free%ROW,                          &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( h_ne, data%H_free%COL,                          &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( h_ne, data%H_free%VAL,                          &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( 0, data%C_null%ROW,                             &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( 0, data%C_null%COL,                             &
                               inform%status,inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( 0, data%C_null%VAL,                             &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( n, data%X,                                      &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( n, data%X_l,                                    &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( n, data%X_u,                                    &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( n, data%Z,                                      &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( n, data%C,                                      &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( n, data%C_free,                                 &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( m, data%B,                                      &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( m, data%B_free,                                 &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( m, data%Y,                                      &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( n + m, data%SOL,                                &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( n, data%STATE,                                  &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      CALL SPACE_resize_array( n, data%STATE_old,                              &
                               inform%status, inform%alloc_status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!   transform H to coordinate form

      SELECT CASE ( SMT_get( prob%H%type ) )
      CASE ( 'COORDINATE' )
        data%H%ROW( : h_ne_orig ) = prob%H%ROW( : h_ne_orig )
        data%H%COL( : h_ne_orig ) = prob%H%COL( : h_ne_orig )
        data%H%VAL( : h_ne_orig ) = prob%H%VAL( : h_ne_orig )
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, n_orig
          DO l = prob%H%PTR( i ), prob%H%PTR( i + 1 ) - 1
            data%H%ROW( l ) = i
            data%H%COL( l ) = prob%H%COL( l )
            data%H%VAL( l ) = prob%H%VAL( l )
          END DO
        END DO
      CASE ( 'DENSE' ) 
        l = 0
        DO i = 1, n_orig
          DO j = 1, i
            l = l + 1
            data%H%ROW( l ) = i
            data%H%COL( l ) = j
            data%H%VAL( l ) = prob%H%VAL( l )
          END DO
        END DO
      CASE ( 'DIAGONAL' ) 
        DO i = 1, n_orig
          data%H%ROW( i ) = i
          data%H%COL( i ) = i
          data%H%VAL( i ) = prob%H%VAL( i )
        END DO
      END SELECT

!   do the same for A

      SELECT CASE ( SMT_get( prob%A%type ) )
      CASE ( 'COORDINATE' )
        data%A%ROW( : a_ne_orig ) = prob%A%ROW( : a_ne_orig )
        data%A%COL( : a_ne_orig ) = prob%A%COL( : a_ne_orig )
        data%A%VAL( : a_ne_orig ) = prob%A%VAL( : a_ne_orig )
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          DO l = prob%A%PTR( i ), prob%A%PTR( i + 1 ) - 1
            data%A%ROW( l ) = i
            data%A%COL( l ) = prob%A%COL( l )
            data%A%VAL( l ) = prob%A%VAL( l )
          END DO
        END DO
      CASE ( 'DENSE' ) 
        l = 0
        DO i = 1, m
          DO j = 1, n_orig
            l = l + 1
            data%A%ROW( l ) = i
            data%A%COL( l ) = j
            data%A%VAL( l ) = prob%A%VAL( l )
          END DO
        END DO
      END SELECT

!   and for the linear term and bounds

      data%X_l( : n_orig ) = prob%X_l( : n_orig )
      data%X_u( : n_orig ) = prob%X_u( : n_orig )
      data%X( : n_orig ) = MIN( prob%X_u( : n_orig ),                          &
                              MAX( prob%X( : n_orig ), prob%X_l( : n_orig ) ) )
      data%C( : n_orig ) = prob%G( : n_orig )
      data%C( n_orig + 1 : n ) = zero

!  continue by assigning slack variables and their bounds

      n = n_orig
      a_ne = a_ne_orig
      DO i = 1, m
        cl = prob%c_l( i ) ; cu = prob%c_u( i )
        IF ( cu == cl ) THEN      !  equality constraint
          data%B( i ) = cl
        ELSE IF ( cl > - control%infinity ) THEN
          IF ( cu < control%infinity ) THEN  !  constraint bounded on both sides
            n = n + 1
            a_ne = a_ne + 1
            data%A%ROW( a_ne ) = i
            data%A%COL( a_ne ) = n
            data%A%VAL( a_ne ) = - one
            data%X( n ) = cl
            data%X_l( n ) = cl
            data%X_u( n ) = cu
            data%B( i ) = zero
          ELSE                    !  constraint bounded from below
            n = n + 1
            a_ne = a_ne + 1
            data%A%ROW( a_ne ) = i
            data%A%COL( a_ne ) = n
            data%A%VAL( a_ne ) = - one
            data%X( n ) = cl
            data%X_l( n ) = cl
            data%X_u( n ) = cu
            data%B( i ) = zero
          END IF
        ELSE
          IF ( cu < control%infinity ) THEN !  constraint bounded from above
            n = n + 1
            a_ne = a_ne + 1
            data%A%ROW( a_ne ) = i
            data%A%COL( a_ne ) = n
            data%A%VAL( a_ne ) = - one
            data%X( n ) = cu
            data%X_l( n ) = cl
            data%X_u( n ) = cu
            data%B( i ) = zero
          ELSE                    !  free constraint
            data%B( i ) = zero
          END IF
        END IF
      END DO

!  initialize the state of the variables to impossible values

      data%STATE_old( : n ) = - 2

!  if desired, use provided initial assigment of variables and constraints

      stats = PRESENT( X_stat ) .AND. PRESENT( C_stat )
      IF ( stats .AND. control%initial_status_provided ) THEN
        IF ( control%out > 0 .AND. control%print_level >= 1 )                  &
          WRITE( control%out, "( ' using provided initial status ' )" )

!  deal with the variable bounds

        n_free = 0
        DO i = 1, n_orig
          xl = prob%X_l( i ) ; xu = prob%X_u( i )
          IF ( X_stat( i ) == 0 ) THEN
            n_free = n_free + 1
            data%X( i ) = MAX( xl, MIN( xu, prob%X( i ) ) )
            data%C_free( n_free ) = data%C( i )
          ELSE IF ( X_stat( i ) < 0 ) THEN
            data%X( i ) = xl 
          ELSE
            data%X( i ) = xu
          END IF
        END DO

!  compute the value of the constraints

        data%Y = zero
        DO l = 1, a_ne_orig
          i = data%A%ROW( l )
          data%Y( i ) =                                                        &
            data%Y( i ) + data%A%val( l ) * data%X( data%A%col( l ) )
        END DO

!  deal with the constraint bounds

        n = n_orig
        DO i = 1, m
          cl = prob%c_l( i ) ; cu = prob%c_u( i )
          IF ( cu == cl ) THEN      !  equality constraint
          ELSE IF ( cl > - control%infinity ) THEN
            n = n + 1
            IF ( cu < control%infinity ) THEN !  bounded on both sides
              IF ( C_stat( i ) == 0 ) THEN
                n_free = n_free + 1
                data%X( n ) = MAX( cl, MIN( cu, data%Y( i ) ) )
                data%C_free( n_free ) = zero
              ELSE IF ( C_stat( i ) < 0 ) THEN
                data%X( n ) = cl
              ELSE
                data%X( n ) = cu
              END IF
            ELSE !  bounded from below
              IF ( C_stat( i ) == 0 ) THEN
                n_free = n_free + 1
                data%X( n ) = MAX( cl, data%Y( i ) )
                data%C_free( n_free ) = zero
              ELSE
                data%X( n ) = cl
              END IF
            END IF
          ELSE
            IF ( cu < control%infinity ) THEN !  bounded from above
              n = n + 1
              IF ( C_stat( i ) == 0 ) THEN
                n_free = n_free + 1
                data%X( n ) = MIN( cu, data%Y( i ) )
                data%C_free( n_free ) = zero
              ELSE
                data%X( n ) = cu
              END IF
            END IF
          END IF
        END DO
      ELSE

!  set the inital active set so that free variables are significantly away
!  from their bounds

        IF ( control%out > 0 .AND. control%print_level > 2 )                   &
          WRITE( control%out,                                                  &
            "( '     i     xl          x           xu          z' )" )
        n_free = 0 ; n_fixed = 0
        DO i = 1, n
          IF ( n_fixed == temporarily_fixed ) THEN
            DO j = i, n
              n_free = n_free + 1
              data%C_free( n_free ) = data%C( j )
            END DO
            EXIT
          END IF
          xl = data%X_l( i ) ; xu = data%X_u( i ) ; x = data%X( i )
          IF ( control%out > 0 .AND. control%print_level > 2 )                 &
            WRITE( control%out, "( I6, 4ES12.4 )" )  i, xl, x, xu, z
          IF ( xu == xl ) THEN      !  fixed variable
            CYCLE
          END IF
          IF ( xl > - control%infinity ) THEN
            IF ( xu < control%infinity ) THEN !  variable bounded on both sides
              data%X( i )  = half * ( xl + xu )
            ELSE  !  variable bounded from below
              data%X( i ) = xl + one
            END IF
          ELSE
            IF ( xu < control%infinity ) THEN !  variable bounded from above
              data%X( i )  = xu - one
            ELSE  ! free variable
              data%X( i ) = zero
            END IF
          END IF
          n_fixed = n_fixed + 1
        END DO
      END IF

!  initialize the dual variables

      data%Y( : m ) = - prob%Y( : m )

!  the problem is complete. Now start the iteration

      inform%iter = 0
      IF ( control%out > 0 .AND. control%print_level > 0 )                     &
        WRITE( control%out, "( /, '  iter    #low     #up   #free      ',      &
       & '   obj             #infeas' )" )

      rho = 100.0_wp
!     rho = 0.01_wp

!  -------------------
!  Main iteration loop
!  -------------------

      DO
        inform%iter = inform%iter + 1
        IF ( inform%iter > control%maxit ) THEN
          inform%status = GALAHAD_error_max_iterations ; GO TO 800
        END IF

!  compute z = c + H x + A(trans) y and the value of the objective function

        data%Z = data%C
        DO l = 1, data%H%ne
          i = data%H%ROW( l ) ; j = data%H%COL( l )
          val = data%H%val( l )
          data%Z( i ) = data%Z( i ) + val * data%X( j )
          IF ( i /= j ) data%Z( j ) = data%Z( j ) + val * data%X( i )
        END DO
  
!       inform%obj = DOT_PRODUCT( 0.5_wp * (data%Z + data%C), data%X ) + prob%f
        inform%obj = 0.5_wp * DOT_PRODUCT( data%Z + data%C, data%X ) + prob%f

!write(6,*) ' y ', data%Y( : m )
        DO l = 1, data%A%ne
          i = data%A%ROW( l ) ; j = data%A%COL( l )
          data%Z( j ) = data%Z( j ) + data%A%val( l ) * data%Y( i )
        END DO
!write(6,*) ' z ', data%X( : n_orig )

!  reset variables that are close to bounds to the bound values 

        DO i = 1, n
          IF ( ABS( data%X( i ) - data%X_l( i ) ) <= control%var_small )       &
            data%X( i ) = data%X_l( i )
          IF ( ABS( data%X( i ) - data%X_u( i ) ) <= control%var_small )       &
            data%X( i ) = data%X_u( i )
          IF ( ABS( data%Z( i ) ) <= control%var_small )                       &
            data%Z( i ) = zero
        END DO

!  print details of the primal and dual variables

!       IF ( inform%iter == 1 ) THEN
        IF ( .FALSE. ) THEN
!       IF ( .TRUE. ) THEN
          WRITE( control%out, "( /, 28X, '<------ Bounds ------>', /           &
         &         '      #  state    value   ',                               &
         &         '    Lower       Upper       Dual ' )" )
          DO i = 1, n
            st = '  FREE'
            IF ( ABS( data%X( i ) - data%X_l( i ) ) < ten ** ( - 5 ) ) THEN
              IF ( ABS( data%Z( i ) ) < ten ** ( - 5 ) ) THEN
                st = 'LDEGEN'
              ELSE
                st = ' LOWER'
              END IF
            END IF
            IF ( ABS( data%X( i ) - data%X_u( i ) ) < ten ** ( - 5 ) ) THEN
              IF ( ABS( data%Z( i ) ) < ten ** ( - 5 ) ) THEN
                st = 'UDEGEN'
              ELSE
                st = ' UPPER'
              END IF
            END IF
            IF ( ABS( data%X_l( i ) - data%X_u( i ) ) < ten ** ( - 10 ) )      &
              st = ' FIXED'
            WRITE( control%out, "( I7, A7, 4ES12.4 )" ) i,                     &
              st, data%X( i ), data%X_l( i ), data%X_u( i ), data%Z( i )
          END DO
        END IF

!  decide on the new active set: 
!   state of variable i =
!    -1 => on lower bound
!     0 => on upper bound
!     k > 0 => kth free variable

        n_free = 0
        n_low = 0 ; n_up = 0 ; n_infeas = 0
!       IF ( inform%iter == control%maxit )                                    &
        IF ( control%out > 0 .AND. control%print_level > 2 )                   &
          WRITE( control%out,                                                  &
            "( '     i     xl          x           xu          z     state' )" )
        DO i = 1, n
          x = data%X( i ) ; z = data%Z( i )
          xl = data%X_l( i ) ; xu = data%X_u( i )
          IF ( x < xl .OR. x > xu ) n_infeas = n_infeas + 1
          IF ( xu == xl ) THEN      !  fixed variable
            data%STATE( i ) = 0
            n_up = n_up + 1
          ELSE IF ( xl > - control%infinity ) THEN
            IF ( xu < control%infinity ) THEN !  variable bounded on both sides
              IF ( x - z / rho <= xl ) THEN
                data%STATE( i ) =  - 1
                n_low = n_low + 1
                IF ( n_low + n_up > n - m ) THEN
                  n_low = n_low - 1
                  DO j = i, n
                    n_free = n_free + 1
                    data%STATE( j ) = n_free
                  END DO
                  EXIT
                END IF
              ELSE IF ( x - z / rho >= xu ) THEN
                data%STATE( i ) =  0
                n_up = n_up + 1
                IF ( n_low + n_up > n - m ) THEN
                  n_up = n_up - 1
                  DO j = i, n
                    n_free = n_free + 1
                    data%STATE( j ) = n_free
                  END DO
                  EXIT
                END IF
              ELSE
                n_free = n_free + 1
                data%STATE( i ) = n_free
                data%C_free( n_free ) = data%C( i )
              END IF
            ELSE                    !  variable bounded from below
              IF ( x - z / rho <= xl ) THEN
                data%STATE( i ) =  - 1
                n_low = n_low + 1
                IF ( n_low + n_up > n - m ) THEN
                  n_low = n_low - 1
                  DO j = i, n
                    n_free = n_free + 1
                    data%STATE( j ) = n_free
                  END DO
                  EXIT
                END IF
              ELSE
                n_free = n_free + 1
                data%STATE( i ) = n_free
                data%C_free( n_free ) = data%C( i )
              END IF
            END IF
          ELSE
            IF ( xu < control%infinity ) THEN !  variable bounded from above
              IF ( x - z / rho >= xu ) THEN
                n_up = n_up + 1
                IF ( n_low + n_up > n - m ) THEN
                  n_up = n_up - 1
                  DO j = i, n
                    n_free = n_free + 1
                    data%STATE( j ) = n_free
                  END DO
                  EXIT
                END IF
                data%STATE( i ) = 0
              ELSE
                n_free = n_free + 1
                data%STATE( i ) = n_free
                data%C_free( n_free ) = data%C( i )
              END IF
            ELSE                    !  free variable
              n_free = n_free + 1
              data%STATE( i ) = n_free
              data%C_free( n_free ) = data%C( i )
            END IF
          END IF
!         IF ( inform%iter == control%maxit )                                  &
          IF ( control%out > 0 .AND. control%print_level > 2 )                 &
            WRITE( control%out, "( I6, 4ES12.4, I6 )" )                        &
              i, xl, x, xu, z, data%STATE( i )
        END DO

        IF ( control%out > 0 .AND. control%print_level > 0 )                   &
          WRITE( control%out, "( I6, 3I8, ES22.14, I10 )" )                    &
           inform%iter, n_low, n_up, n_free, inform%obj, n_infeas

!  Check to see if the state of the variables has changed
!  from that in the previous iteration

!  The state has not changed. Exit

        IF ( COUNT( data%STATE_old( : n ) /= data%STATE( : n ) ) == 0 ) THEN
          prob%X( : n_orig ) = data%X( : n_orig )
          prob%Z( : n_orig ) = data%Z( : n_orig )
          prob%Y( : m ) = - data%Y( : m )
          prob%C( : m ) = zero
          DO l = 1, a_ne_orig
            i = data%A%ROW( l ) ; j = data%A%COL( l )
            prob%C( i ) = prob%C( i ) + data%A%val( l ) * prob%X( j )
          END DO
          inform%status = GALAHAD_ok ; GO TO 800
        END IF
        data%STATE_old( : n ) = data%STATE( : n )

!  construct the matrix of free components of H and
!  add the free/fixed components of H * x from c

        data%H_free%n = n_free
        data%H_free%ne = 0
        DO l = 1, data%H%ne
          i = data%H%ROW( l ) ; j = data%H%COL( l )
          ii = data%STATE( i ) ; jj = data%STATE( j )
          IF ( ii > 0 .AND. jj > 0 ) THEN
            data%H_free%ne = data%H_free%ne + 1
            data%H_free%ROW( data%H_free%ne ) = ii
            data%H_free%COL( data%H_free%ne ) = jj
            data%H_free%VAL( data%H_free%ne ) = data%H%val( l )
          ELSE IF ( ii > 0 ) THEN
            IF ( jj == - 1 ) THEN
              data%C_free( ii ) =                                              &
                data%C_free( ii ) + data%H%val( l ) * data%X_l( j )
            ELSE
              data%C_free( ii ) =                                              &
                data%C_free( ii ) + data%H%val( l ) * data%X_u( j )
            END IF
          ELSE IF ( jj > 0 ) THEN
            IF ( ii == - 1 ) THEN
              data%C_free( jj ) =                                              &
                data%C_free( jj ) + data%H%val( l ) * data%X_l( i )
            ELSE
              data%C_free( jj ) =                                              &
                data%C_free( jj ) + data%H%val( l ) * data%X_u( i )
            END IF
          END IF
        END DO

!  similarly, construct the matrix of free components of A

        data%A_free%n = n_free
        data%A_free%m = m
        data%A_free%ne = 0
        data%B_free( : m ) = data%B( : m )
        DO l = 1, data%A%ne
          i = data%A%ROW( l ) ; j = data%A%COL( l )
          jj = data%STATE( j )
          IF ( jj > 0 ) THEN
            data%A_free%ne = data%A_free%ne + 1
            data%A_free%ROW( data%A_free%ne ) = i
            data%A_free%COL( data%A_free%ne ) = jj
            data%A_free%VAL( data%A_free%ne ) = data%A%val( l )
          ELSE  IF ( jj == - 1 ) THEN
            data%B_free( i )                                                   &
              = data%B_free( i ) - data%A%val( l ) * data%X_l( j )
          ELSE
            data%B_free( i )                                                   &
              = data%B_free( i ) - data%A%val( l ) * data%X_u( j )
          END IF
        END DO
     
!  factorize the matrix

!  (  H_free   A_free^T )
!  (  A_free      0     )

        IF ( m + n_free > 0 ) THEN
          CALL SBLS_form_and_factorize( n_free, m, data%H_free, data%A_free,   &
                                        data%C_null, data%sbls_data,           &
                                        control%sbls_control,                  &
                                        inform%sbls_inform )

!  solve the system

!  (  H_free   A_free^T ) ( x_free ) = ( - c - H_free/fixed * x_fixed )
!  (  A_free      0     ) (    y   )   (          b                   )

          data%SOL( : n_free ) = - data%C_free( : n_free )
          data%SOL( n_free + 1 : n_free + m ) = data%B_free( : m )
          CALL SBLS_solve( n_free, m, data%A_free, data%C_null,                &
                           data%sbls_data, control%sbls_control,               &
                           inform%sbls_inform, data%SOL( : n_free + m ) )
        END IF

!  recover x and y

        DO i = 1, n
          ii = data%STATE( i  )
          IF ( ii == - 1 ) THEN
            data%X( i ) = data%X_l( i )
          ELSE IF ( ii == 0 ) THEN
            data%X( i ) = data%X_u( i )
          ELSE
            data%X( i ) = data%SOL( ii )
          END IF
        END DO
        data%Y( : m ) = data%SOL( n_free + 1 : n_free + m )

        IF ( control%out > 0 .AND. control%print_level > 1 ) THEN
          prob%C( : m ) = zero
          DO l = 1, a_ne_orig
            i = data%A%ROW( l ) ; j = data%A%COL( l )
            prob%C( i ) = prob%C( i ) + data%A%val( l ) * data%X( j )
          END DO

          WRITE( control%out, "( /, ' state ',  60A1, /, ( 7X, 60A1 ) )" )     &
             ( STATE( data%STATE( i ) ), i = 1, n )
          WRITE( control%out, "( ' X  ', 6ES12.4, ( 4X, 6ES12.4 ) )" )         &
            data%X( : n_orig )
          WRITE( control%out, "( ' AX ', 6ES12.4, ( 4X, 6ES12.4 ) )" )         &
            prob%C( : m )
          WRITE( control%out, "( ' Y  ', 6ES12.4, ( 4X, 6ES12.4 ) )" )         &
            data%Y( : m )
        END IF

!  --------------------------
!  End of Main iteration loop
!  --------------------------

      END DO

  800 CONTINUE 
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving PDQP_solve with status', I0 )" )      &
          inform%status
      RETURN  

      CONTAINS

        FUNCTION STATE( i )

!  Returns the state  L, U or F depending on whether i is <0, 0 or >0

        CHARACTER ( LEN = 1 ) :: STATE
        INTEGER, INTENT( IN ) :: i
        IF ( i < 0 ) THEN
          STATE = 'L'
        ELSE IF ( i == 0 ) THEN
          STATE = 'U'
        ELSE
          STATE = 'F'
        END IF
        END FUNCTION STATE

!  End of PDQP_solve

      END SUBROUTINE PDQP_solve

!-*-*-*-*-*-*-   P D Q P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE PDQP_terminate( data, control, inform )

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
!   data    see Subroutine PDQP_initialize
!   control see Subroutine PDQP_initialize
!   inform  see Subroutine PDQP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( PDQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( PDQP_control_type ), INTENT( IN ) :: control        
      TYPE ( PDQP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate arrays allocated by SBLS

      CALL SBLS_terminate( data%sbls_data, control%sbls_control,             &
                           inform%sbls_inform )

!  Deallocate all other allocated arrays

      array_name = 'pdqp: data%A%type'
      CALL SPACE_dealloc_array( data%A%type,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%A%ROW'
      CALL SPACE_dealloc_array( data%A%ROW,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%A%COL'
      CALL SPACE_dealloc_array( data%A%COL,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%A%VAL'
      CALL SPACE_dealloc_array( data%A%VAL,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%H%type'
      CALL SPACE_dealloc_array( data%H%type,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%H%ROW'
      CALL SPACE_dealloc_array( data%H%ROW,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%H%COL'
      CALL SPACE_dealloc_array( data%H%COL,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%H%VAL'
      CALL SPACE_dealloc_array( data%H%VAL,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%A_free%type'
      CALL SPACE_dealloc_array( data%A_free%type,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%A_free%ROW'
      CALL SPACE_dealloc_array( data%A_free%ROW,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%A_free%COL'
      CALL SPACE_dealloc_array( data%A_free%COL,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%A_free%VAL'
      CALL SPACE_dealloc_array( data%A_free%VAL,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%H_free%type'
      CALL SPACE_dealloc_array( data%H_free%type,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%H_free%ROW'
      CALL SPACE_dealloc_array( data%H_free%ROW,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%H_free%COL'
      CALL SPACE_dealloc_array( data%H_free%COL,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%H_free%VAL'
      CALL SPACE_dealloc_array( data%H_free%VAL,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%C_null%type'
      CALL SPACE_dealloc_array( data%C_null%type,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%C_null%ROW'
      CALL SPACE_dealloc_array( data%C_null%ROW,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%C_null%COL'
      CALL SPACE_dealloc_array( data%C_null%COL,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%C_null%VAL'
      CALL SPACE_dealloc_array( data%C_null%VAL,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%STATE'
      CALL SPACE_dealloc_array( data%STATE,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%STATE_ole'
      CALL SPACE_dealloc_array( data%STATE_old,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%X'
      CALL SPACE_dealloc_array( data%X,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%X_l'
      CALL SPACE_dealloc_array( data%X_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%X_u'
      CALL SPACE_dealloc_array( data%X_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%B'
      CALL SPACE_dealloc_array( data%B,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%Y'
      CALL SPACE_dealloc_array( data%Y,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%Z'
      CALL SPACE_dealloc_array( data%Z,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN
      RETURN

      array_name = 'pdqp: data%B_free'
      CALL SPACE_dealloc_array( data%B_free,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%C_free'
      CALL SPACE_dealloc_array( data%C_free,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'pdqp: data%SOL'
      CALL SPACE_dealloc_array( data%SOL,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN
      RETURN

!  End of subroutine PDQP_terminate

      END SUBROUTINE PDQP_terminate

!  End of module PDQP

   END MODULE GALAHAD_PDQP_double
