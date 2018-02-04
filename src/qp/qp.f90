! THIS VERSION: GALAHAD 2.6 - 08/05/2013 AT 13:30 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ Q P    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released in GALAHAD Version 2.4. January 5th 2011

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_QP_double

!     ------------------------------------------------
!     |                                              |
!     | Minimize the quadratic objective function    |
!     |                                              |
!     |  1/2 x^T H x + g^T x + f                     |
!     |                                              |
!     | or linear/seprable objective function        |
!     |                                              |
!     |  1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f   |
!     |                                              |
!     | subject to the linear constraints and bounds |
!     |                                              |
!     |          c_l <= A x <= c_u                   |
!     |          x_l <=  x <= x_u                    |
!     |                                              |
!     | for some posibly indeinite Hessian H or      |
!     | (possibly zero) diagonal matrix W using a    |
!     | variety of methods. This provides a generic  |
!     | interface to all GALAHAD QP routines         |
!     |                                              |
!     ------------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS, ACTIVE => GALAHAD_ACTIVE, TRACE => GALAHAD_TRACE,   &
                           DEBUG => GALAHAD_DEBUG
      USE GALAHAD_SPACE_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_QPT_double
      USE GALAHAD_QPD_double, QP_data_type => QPD_data_type
      USE GALAHAD_SORT_double, ONLY: SORT_reorder_by_rows
      USE GALAHAD_SCALE_double
      USE GALAHAD_PRESOLVE_double
      USE GALAHAD_MOP_double
      USE GALAHAD_QPA_double
      USE GALAHAD_QPB_double
      USE GALAHAD_QPC_double
      USE GALAHAD_CQP_double
      USE GALAHAD_DQP_double
      USE GALAHAD_CCQP_double
      USE GALAHAD_LMS_double, ONLY: LMS_apply_lbfgs

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: QP_initialize, QP_read_specfile, QP_solve,                     &
                QP_terminate, QPT_problem_type, SMT_type, SMT_put, SMT_get,    &
                QP_data_type

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
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!  non-standard error returns


!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QP_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   scaling is controled by scale. Possible values are:
!     0  no scaling
!     1  scale to try to map all variables and constraints to [0,1]
!     2  normalize rows of K = ( H A(transpose) ) using Curtis and Reid'
!                              ( A      0       )  symmetric method
!     3  normalize rows & columns of A uing Curtis and Reid' unsymmetric method
!     4  normalize rows of A so that each has one-norm close to 1
!     5  normalize rows of K (cf 2) then normalize rows of A (cf 4)
!     6  normalize rows & columns of A (cf 3) then normalize rows of A (cf 4)
!     7  normalize rows & columns using Sinkhorn-Knopp equilibration
!    <0  apply -(scale = 1-7) above but before trying presolve

        INTEGER :: scale = 0

!    specifies the unit number to write generated SIF file describing the
!     current problem

        INTEGER :: sif_file_device = 52

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!   if %presolve true, the problem will be simplified by calling GALAHAD's
!     presolve package

        LOGICAL :: presolve = .FALSE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!   if %generate_sif_file is .true. if a SIF file describing the current
!    problem is to be generated

        LOGICAL :: generate_sif_file = .FALSE.

!  quadratic programming solver. Possible values are
!   qpa, qpb, qpc, cqp, dqp, ccqp

        CHARACTER ( LEN = 30 ) :: quadratic_programming_solver =               &
           "ccqp" // REPEAT( ' ', 26 )

!  name of generated SIF file containing input problem

        CHARACTER ( LEN = 30 ) :: sif_file_name =                              &
         "QPPROB.SIF"  // REPEAT( ' ', 19 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for SCALE

        TYPE ( SCALE_control_type ) :: SCALE_control

!  control parameters for PRESOLVE

        TYPE ( PRESOLVE_control_type ) :: PRESOLVE_control

!  control parameters for QPA

        TYPE ( QPA_control_type ) :: QPA_control

!  control parameters for QPB

        TYPE ( QPB_control_type ) :: QPB_control

!  control parameters for QPC

        TYPE ( QPC_control_type ) :: QPC_control

!  control parameters for CQP

        TYPE ( CQP_control_type ) :: CQP_control

!  control parameters for DQP

        TYPE ( DQP_control_type ) :: DQP_control

!  control parameters for CCQP

        TYPE ( CCQP_control_type ) :: CCQP_control

      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QP_time_type

!  the total cpu time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the cpu time spent presolving the problem

        REAL ( KIND = wp ) :: presolve = 0.0

!  the cpu time spent scaling the problem

        REAL ( KIND = wp ) :: scale = 0.0

!  the cpu time spent in the optimization

        REAL ( KIND = wp ) :: solve = 0.0

!  the total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent presolving the problem

        REAL ( KIND = wp ) :: clock_presolve = 0.0

!  the clock time spent scaling the problem

        REAL ( KIND = wp ) :: clock_scale = 0.0

!  the clock time spent in the optimization

        REAL ( KIND = wp ) :: clock_solve = 0.0

      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QP_inform_type

!  return status. See QP_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the value of the objective function at the best estimate of the solution
!   determined by QP_solve

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  the value of the primal infeasibility

        REAL ( KIND = wp ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

        REAL ( KIND = wp ) :: dual_infeasibility = HUGE( one )

!  the value of the complementary slackness

        REAL ( KIND = wp ) :: complementary_slackness = HUGE( one )

!  timings (see above)

        TYPE ( QP_time_type ) :: time

!  inform parameters for SCALE

        TYPE ( SCALE_inform_type ) :: SCALE_inform

!  inform parameters for PRESOLVE

        TYPE ( PRESOLVE_inform_type ) :: PRESOLVE_inform

!  inform parameters for QPA

        TYPE ( QPA_inform_type ) :: QPA_inform

!  inform parameters for QPB

        TYPE ( QPB_inform_type ) :: QPB_inform

!  inform parameters for QPC

        TYPE ( QPC_inform_type ) :: QPC_inform

!  inform parameters for CQP

        TYPE ( CQP_inform_type ) :: CQP_inform

!  inform parameters for DQP

        TYPE ( DQP_inform_type ) :: DQP_inform

!  inform parameters for CCQP

        TYPE ( CCQP_inform_type ) :: CCQP_inform
      END TYPE

   CONTAINS

!-*-*-*-*-*-   Q P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE QP_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for QP. This routine should be called before QP_solve
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

      TYPE ( QP_data_type ), INTENT( INOUT ) :: data
      TYPE ( QP_control_type ), INTENT( OUT ) :: control
      TYPE ( QP_inform_type ), INTENT( OUT ) :: inform

!  Set control parameters

      CALL SCALE_initialize( data%SCALE_data ,control%SCALE_control,           &
                             inform%SCALE_inform )
      control%SCALE_control%prefix    = '" - SCALE:"                   '
      CALL PRESOLVE_initialize( control%PRESOLVE_control,                      &
                                inform%PRESOLVE_inform, data%PRESOLVE_data )
!     control%PRESOLVE_control%prefix = '" - PRESOLVE:"                '
      CALL QPA_initialize( data, control%QPA_control, inform%QPA_inform )
      control%QPA_control%prefix = '" - QPA:"                     '
      CALL QPB_initialize( data, control%QPB_control, inform%QPB_inform )
      control%QPB_control%prefix = '" - QPB:"                     '
      CALL QPC_initialize( data, control%QPC_control, inform%QPC_inform  )
      control%QPC_control%prefix = '" - QPC:"                     '
      CALL CQP_initialize( data, control%CQP_control, inform%CQP_inform  )
      control%CQP_control%prefix = '" - CQP:"                     '
      CALL DQP_initialize( data, control%DQP_control, inform%DQP_inform  )
      control%DQP_control%prefix = '" - DQP:"                     '
      CALL CCQP_initialize( data, control%CCQP_control, inform%CCQP_inform  )
      control%CQP_control%prefix = '" - CCQP:"                    '

      inform%status = GALAHAD_ok

      RETURN

!  End of QP_initialize

      END SUBROUTINE QP_initialize

!-*-*-*-*-   Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE QP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by QP_initialize could (roughly)
!  have been set as:

! BEGIN QP SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  scale-problem                                     0
!  sif-file-device                                   52
!  infinity-value                                    1.0D+19
!  pre-solve-problem                                 F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  quadratic-programming-solver                      qpc
!  sif-file-name                                     QPPROB.SIF
!  output-line-prefix                                ""
! END QP SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( QP_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname


!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: scale = print_level + 1
      INTEGER, PARAMETER :: sif_file_device = scale + 1
      INTEGER, PARAMETER :: infinity = sif_file_device + 1
      INTEGER, PARAMETER :: presolve = infinity + 1
      INTEGER, PARAMETER :: space_critical = presolve + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: quadratic_programming_solver = generate_sif_file + 1
      INTEGER, PARAMETER :: sif_file_name = quadratic_programming_solver + 1
      INTEGER, PARAMETER :: prefix = sif_file_name + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'QP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      TYPE ( PRESOLVE_inform_type ) :: PRESOLVE_inform

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( scale )%keyword = 'scale-problem'
      spec( sif_file_device )%keyword = 'sif-file-device'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'

!  Logical key-words

      spec( presolve )%keyword = 'pre-solve-problem'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
      spec( generate_sif_file )%keyword = 'generate-sif-file'

!  Character key-words

      spec( quadratic_programming_solver )%keyword                             &
        = 'quadratic-programming-solver'
      spec( sif_file_name )%keyword = 'sif-file-name'
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
     CALL SPECFILE_assign_value( spec( scale ),                                &
                                 control%scale,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_device ),                      &
                                 control%sif_file_device,                      &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
!  Set logical values

     CALL SPECFILE_assign_value( spec( presolve ),                             &
                                 control%presolve,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( generate_sif_file ),                    &
                                 control%generate_sif_file,                    &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( quadratic_programming_solver ),         &
                                 control%quadratic_programming_solver,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_name ),                        &
                                 control%sif_file_name,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )


!  Make sure that inifinity is set consistently

      control%PRESOLVE_control%infinity = control%infinity
      control%QPA_control%infinity = control%infinity
      control%QPB_control%infinity = control%infinity
      control%QPC_control%infinity = control%infinity
      control%CQP_control%infinity = control%infinity
      control%DQP_control%infinity = control%infinity
      control%CCQP_control%infinity = control%infinity

!  Read the specfiles for PRESOLVE, QPA, QPB, QPC, CQP and DQP

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SCALE_read_specfile( control%SCALE_control, device,               &
                              alt_specname = TRIM( alt_specname ) // '-SCALE' )
        CALL PRESOLVE_read_specfile( device, control%PRESOLVE_control,         &
                                     PRESOLVE_inform, alt_specname =           &
                                     TRIM( alt_specname ) // '-PRESOLVE' )
        CALL QPA_read_specfile( control%QPA_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-QPA' )
        CALL QPB_read_specfile( control%QPB_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-QPB' )
        CALL QPC_read_specfile( control%QPC_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-QPC' )
        CALL CQP_read_specfile( control%CQP_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-CQP' )
        CALL DQP_read_specfile( control%DQP_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-DQP' )
        CALL CCQP_read_specfile( control%CCQP_control, device,                 &
                                alt_specname = TRIM( alt_specname ) // '-CCQP' )
      ELSE
        CALL SCALE_read_specfile( control%SCALE_control, device )
        CALL PRESOLVE_read_specfile( device, control%PRESOLVE_control,         &
                                     PRESOLVE_inform )
        CALL QPA_read_specfile( control%QPA_control, device )
        CALL QPB_read_specfile( control%QPB_control, device )
        CALL QPC_read_specfile( control%QPC_control, device )
        CALL CQP_read_specfile( control%CQP_control, device )
        CALL DQP_read_specfile( control%DQP_control, device )
        CALL CCQP_read_specfile( control%CCQP_control, device )
      END IF

      RETURN

!  End of QP_read_specfile

      END SUBROUTINE QP_read_specfile

!-*-*-*-*-*-*-*-*-*-   Q P _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE QP_solve( prob, data, control, inform, C_stat, B_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the quadratic
!
!        1/2 x^T H x + g^T x + f
!
!  or linear/separable objective
!
!        1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f
!
!  where
!
!             (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!  and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ),
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite. The subroutine is particularly
!  appropriate when H and/or A are sparse.
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
!    to be solved since the last call to QP_initialize, and .FALSE. if
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
!        feasible region will be found. %WEIGHT (see below) need not be set
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
!       The Hessian in this case is available via the component %H_lm below
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above,
!    except for scheme (ix), for which a permutation will be set within H_lm.
!    However, if scheme (i) is used for input, the output H%row will contain
!    the row numbers corresponding to the values in H%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %H_lm is a structure of type LMS_data_type, whose components hold the
!     L-BFGS Hessian. Access to this structure is via the module GALAHAD_LMS,
!     and this component needs only be set if %H%type( 1 : 5 ) = 'LBFGS.'
!
!   %WEIGHT is a REAL array, which need only be set if %Hessian_kind is larger
!    than 1. If this is so, it must be of length at least %n, and contain the
!    weights W for the objective function.
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
!   %G is a REAL array, which need only be set if %gradient_kind is not 0
!    or 1. If this is so, it must be of length at least %n, and contain the
!    linear terms g for the objective function.
!
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
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
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output A%row will contain
!    the row numbers corresponding to the values in A%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %C is a REAL array of length %m, which is used to store the values of
!    A x. It need not be set on entry. On exit, it will have been filled
!    with appropriate values.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to estimaes of the solution, x. On successful exit, it will contain
!    the required solution, x.
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
!  data is a structure of type QP_data_type which holds private internal data
!
!  control is a structure of type QP_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to QP_initialize. See QP_initialize
!   for details
!
!  inform is a structure of type QP_inform_type that provides
!    information on exit from QP_solve. The component status
!    has possible values:
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
!        prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated.
!
!    -4 The constraints are inconsistent.
!
!    -5 The constraints appear to have no feasible point.
!
!    -7 The objective function appears to be unbounded from below on the
!       feasible set.
!
!    -8 The analytic center appears to be unbounded.
!
!    -9 The analysis phase of the factorization failed; the return status
!       from the factorization package is given in the component factor_status.
!
!   -10 The factorization failed; the return status from the factorization
!       package is given in the component factor_status.
!
!   -11 The solve of a required linear system failed; the return status from
!       the factorization package is given in the component factor_status.
!
!   -16 The problem is so ill-conditoned that further progress is impossible.
!
!   -17 The step is too small to make further impact.
!
!   -18 Too many iterations have been performed.
!
!   -19 Too much elapsed CPU or system clock time has passed.
!
!  C_stat is an optional INTEGER array of length m, which if present will be
!   set on exit to indicate the likely ultimate status of the constraints.
!   Possible values are
!   C_stat( i ) < 0, the i-th constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th constraint is likely not in the active set
!
!  B_stat is an optional INTEGER array of length n, which if present will be
!   set on exit to indicate the likely ultimate status of the simple bound
!   constraints. Possible values are
!   B_stat( i ) < 0, the i-th bound constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is likely not in the active set
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( QP_data_type ), INTENT( INOUT ) :: data
      TYPE ( QP_control_type ), INTENT( IN ) :: control
      TYPE ( QP_inform_type ), INTENT( OUT ) :: inform
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: B_stat

!  Local variables

      INTEGER :: i, scale
      REAL :: time_start, time_now, time_end
      REAL ( KIND = wp ) :: clock_start, clock_now, clock_end
      REAL ( KIND = wp ) :: val
      LOGICAL :: printi, stat_required, presolve, lbfgs
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering QP_solve ' )" ) prefix

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( control%generate_sif_file ) THEN
        CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,    &
                      control%infinity, .TRUE. )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  initialize counts

      inform%status = GALAHAD_ok
      stat_required = PRESENT( C_stat ) .AND. PRESENT( B_stat )

!  basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1

!  ensure that input parameters are within allowed ranges

      IF ( prob%n < 1 .OR. prob%m < 0 .OR.                                     &
           .NOT. QPT_keyword_A( prob%A%type ) .OR.                             &
           .NOT. QPT_keyword_H( prob%H%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
           "( ' ', /, A, '   **  Error return ', I0, ' from QP ' )" )          &
          prefix, inform%status
        GO TO 800
      END IF

!  is an L-BFGS Hessian used?

      lbfgs = SMT_get( prob%H%type ) == 'LBFGS'

!  record matrix dimensions

      prob%A%m = prob%m
      prob%A%n = prob%n
      prob%H%m = prob%n
      prob%H%n = prob%n

      IF ( SMT_get( prob%H%type ) == 'NONE' .OR.                               &
           SMT_get( prob%H%type ) == 'ZERO' .OR.                               &
           SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
        prob%H%ne = 0
      ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
        prob%H%ne = 1
      ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
        prob%H%ne = prob%n
      ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
        prob%H%ne = prob%n * prob%n
      ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
        prob%H%ne = prob%H%ptr( prob%n + 1 ) - 1
      END IF
      IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
        prob%A%ne = prob%m * prob%n
      ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
        prob%A%ne = prob%A%ptr( prob%m + 1 ) - 1
      END IF

!  if required, write out problem

      IF ( control%out > 0 .AND. control%print_level >= 20 )                   &
        CALL QPT_summarize_problem( control%out, prob )

!  allocate workspace

      array_name = 'qp: data%SH'
      CALL SPACE_resize_array( prob%n, data%SH, inform%status,                 &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  -----------------------------
!  scale the problem if required
!  -----------------------------
!write(6,*) ' - scale ', control%scale

      IF ( lbfgs ) THEN ! to do: remove restriction
        scale = 0
      ELSE
        scale = control%scale
      END IF
      IF ( scale < 0 ) THEN
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        CALL SCALE_get( prob, - control%scale,                                 &
                        data%SCALE_trans, data%SCALE_data,                     &
                        control%SCALE_control, inform%SCALE_inform )
        IF ( inform%SCALE_inform%status < 0 ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, '  ERROR return from SCALE (status =', I0, ')' )" )          &
               prefix, inform%SCALE_inform%status
          CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
          inform%time%scale = inform%time%scale + time_end - time_now
          inform%time%clock_scale =                                            &
            inform%time%clock_scale + clock_end - clock_now
          inform%status = GALAHAD_error_scale ; GO TO 800
        END IF
        CALL SCALE_apply( prob, data%SCALE_trans, data%SCALE_data,             &
                          control%SCALE_control, inform%SCALE_inform )
        CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
        inform%time%scale = inform%time%scale + time_end - time_now
        inform%time%clock_scale =                                              &
          inform%time%clock_scale + clock_end - clock_now
        IF ( inform%SCALE_inform%status < 0 ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, '  ERROR return from SCALE (status =', I0, ')' )" )          &
               prefix, inform%SCALE_inform%status
          inform%status = GALAHAD_error_scale ; GO TO 800
        END IF
      END IF

!  if the presolver is to be used, allocate sufficient space

! to do: remove LBFGS restriction
      presolve = control%presolve .AND. .NOT. lbfgs
      IF ( presolve ) THEN
        array_name = 'qp: prob%X_status'
        CALL SPACE_resize_array( prob%n, prob%X_status, inform%status,         &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qp: prob%C_status'
        CALL SPACE_resize_array( prob%m, prob%C_status, inform%status,         &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qp: prob%Z_l'
        CALL SPACE_resize_array( prob%n, prob%Z_l, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qp: prob%Z_u'
        CALL SPACE_resize_array( prob%n, prob%Z_u, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qp: prob%Y_l'
        CALL SPACE_resize_array( prob%m, prob%Y_l, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qp: prob%Y_u'
        CALL SPACE_resize_array( prob%m, prob%Y_u, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        prob%X_status = ACTIVE
        prob%C_status = ACTIVE
        prob%Z_l( : prob%n ) = - control%infinity
        prob%Z_u( : prob%n ) =   control%infinity
        prob%Y_l( : prob%m ) = - control%infinity
        prob%Y_u( : prob%m ) =   control%infinity

!  --------------------
!  presolve if required
!  --------------------

        IF ( printi ) WRITE( control%out,                                      &
         "( A, ' dimensions prior to presolve:',                               &
       &    ' n = ', I0, ', m = ', I0, ', a_ne = ', I0, ', h_ne = ', I0 )" )   &
            prefix, prob%n, prob%m, MAX( 0, prob%A%ne ), MAX( 0, prob%H%ne )

!  overide some defaults

!        control%PRESOLVE_control%c_accuracy =                                 &
!          ten * QP_control%QPB_control%stop_p
!        control%PRESOLVE_control%z_accuracy =                                 &
!          ten * QP_control%QPB_control%stop_d

!  ensure that data will be restored after the presolve

        data%PRESOLVE_control = control%PRESOLVE_control
        data%PRESOLVE_control%get_q = .TRUE.
        data%PRESOLVE_control%get_f = .TRUE.
        data%PRESOLVE_control%get_g = .TRUE.
        data%PRESOLVE_control%get_H = .TRUE.
        data%PRESOLVE_control%get_A = .TRUE.
        data%PRESOLVE_control%get_x = .TRUE.
        data%PRESOLVE_control%get_x_bounds = .TRUE.
        data%PRESOLVE_control%get_z = .TRUE.
        data%PRESOLVE_control%get_z_bounds = .TRUE.
        data%PRESOLVE_control%get_c = .TRUE.
        data%PRESOLVE_control%get_c_bounds = .TRUE.
        data%PRESOLVE_control%get_y = .TRUE.
        data%PRESOLVE_control%get_y_bounds = .TRUE.

!  call the presolver

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        CALL PRESOLVE_initialize( data%PRESOLVE_control,                       &
                                inform%PRESOLVE_inform, data%PRESOLVE_data )
        IF ( inform%PRESOLVE_inform%status < 0 ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, '  ERROR return from PRESOLVE (status =', I0, ')' )" )       &
               prefix, inform%PRESOLVE_inform%status
          inform%status = GALAHAD_error_presolve ; GO TO 800
        END IF
        CALL PRESOLVE_apply( prob, data%PRESOLVE_control,                      &
                             inform%PRESOLVE_inform,                           &
                             data%PRESOLVE_data )
        CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
        inform%time%presolve = inform%time%presolve + time_end - time_now
        inform%time%clock_presolve =                                           &
            inform%time%clock_presolve + clock_end - clock_now
        IF ( inform%PRESOLVE_inform%status < 0 ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, '  ERROR return from PRESOLVE (status =', I0, ')' )" )       &
               prefix, inform%PRESOLVE_inform%status
          inform%status = GALAHAD_error_presolve ; GO TO 800
        END IF

        IF ( SMT_get( prob%H%type ) == 'NONE' .OR.                             &
             SMT_get( prob%H%type ) == 'ZERO' .OR.                             &
             SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
          prob%H%ne = 0
        ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
          prob%H%ne = 1
        ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
          prob%H%ne = prob%n
        ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          prob%H%ne = prob%n * prob%n
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          prob%H%ne = prob%H%ptr( prob%n + 1 ) - 1
        END IF

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          prob%A%ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          prob%A%ne = prob%A%ptr( prob%m + 1 ) - 1
        END IF

        IF ( printi ) WRITE( control%out,                                      &
          "( A, ' updated dimensions:  n = ', I0,                              &
       &    ', m = ', I0, ', a_ne = ', I0, ', h_ne = ', I0, /,                 &
       &    A, ' preprocessing time = ', F0.2,                                 &
       &    ', number of transformations = ', I0, / )" )                       &
            prefix, prob%n, prob%m, MAX( 0, prob%A%ne ), MAX( 0, prob%H%ne ),  &
            prefix, time_end - time_now, inform%PRESOLVE_inform%nbr_transforms
      END IF

      IF ( prob%n > 0 ) THEN

!  -----------------------------
!  scale the problem if required
!  -----------------------------

        IF ( scale > 0 ) THEN
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          CALL SCALE_get( prob, control%scale,                                 &
                          data%SCALE_trans, data%SCALE_data,                   &
                          control%SCALE_control, inform%SCALE_inform )
          IF ( inform%SCALE_inform%status < 0 ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, '  ERROR return from SCALE (status =', I0, ')' )" )        &
                 prefix, inform%SCALE_inform%status
            CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
            inform%time%scale = inform%time%scale + time_end - time_now
            inform%time%clock_scale =                                          &
              inform%time%clock_scale + clock_end - clock_now
            inform%status = GALAHAD_error_scale ; GO TO 800
          END IF
          CALL SCALE_apply( prob, data%SCALE_trans, data%SCALE_data,           &
                            control%SCALE_control, inform%SCALE_inform )
          CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
          inform%time%scale = inform%time%scale + time_end - time_now
          inform%time%clock_scale =                                            &
            inform%time%clock_scale + clock_end - clock_now
          IF ( inform%SCALE_inform%status < 0 ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, '  ERROR return from SCALE (status =', I0, ')' )" )        &
                 prefix, inform%SCALE_inform%status
            inform%status = GALAHAD_error_scale ; GO TO 800
          END IF
        END IF

!  ------------------
!  call the optimizer
!  ------------------

!  allocate additional workspace

        IF ( .NOT. stat_required ) THEN
          SELECT CASE( TRIM( control%quadratic_programming_solver ) )
          CASE ( 'qpa', 'QPA', 'qpc', 'QPC' )
            array_name = 'qp: data%C_stat'
            CALL SPACE_resize_array( prob%m, data%C_stat, inform%status,       &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            array_name = 'qp: data%B_stat'
            CALL SPACE_resize_array( prob%n, data%B_stat, inform%status,       &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END SELECT
        END IF

!  apply the slected solver

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        SELECT CASE( TRIM( control%quadratic_programming_solver ) )

!  == QPA ==

        CASE ( 'qpa', 'QPA' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD QPA solver used **' )" ) prefix
          IF ( stat_required ) THEN
            CALL QPA_solve( prob, C_stat, B_stat, data,                        &
                            control%QPA_control, inform%QPA_inform )
          ELSE
            CALL QPA_solve( prob, data%C_stat, data%B_stat, data,              &
                            control%QPA_control, inform%QPA_inform )
          END IF
          IF ( inform%QPA_inform%status /= GALAHAD_ok .AND.                    &
               inform%QPA_inform%status /= GALAHAD_error_ill_conditioned .AND. &
               inform%QPA_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD QPA solve error status = ', I0 )" ) prefix,      &
              inform%QPA_inform%status
            inform%status = GALAHAD_error_qpa ; GO TO 800
          END IF

!  == QPB ==

        CASE ( 'qpb', 'QPB' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD QPB solver used **' )" ) prefix
          CALL QPB_solve( prob, data, control%QPB_control, inform%QPB_inform,  &
                          C_stat, B_stat )
          IF ( inform%QPB_inform%status /= GALAHAD_ok .AND.                    &
               inform%QPB_inform%status /= GALAHAD_error_ill_conditioned .AND. &
               inform%QPB_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD QPB solve error status = ', I0 )" ) prefix,      &
              inform%QPB_inform%status
            inform%status = GALAHAD_error_qpb ; GO TO 800
          END IF

!  == QPC ==

        CASE ( 'qpc', 'QPC' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD QPC solver used **' )" ) prefix
          IF ( stat_required ) THEN
            CALL QPC_solve( prob, C_stat, B_stat, data,                        &
                            control%QPC_control, inform%QPC_inform )
          ELSE
            CALL QPC_solve( prob, data%C_stat, data%B_stat, data,              &
                            control%QPC_control, inform%QPC_inform )
          END IF
          IF ( inform%QPC_inform%status /= GALAHAD_ok .AND.                    &
               inform%QPC_inform%status /= GALAHAD_error_ill_conditioned .AND. &
               inform%QPC_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD QPC solve error status = ', I0 )" ) prefix,      &
              inform%QPC_inform%status
            inform%status = GALAHAD_error_qpc ; GO TO 800
          END IF

!  == CQP ==

        CASE ( 'cqp', 'CQP' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD CQP solver used **' )" ) prefix
          CALL CQP_solve( prob, data, control%CQP_control, inform%CQP_inform,  &
                          C_stat, B_stat )
          IF ( inform%CQP_inform%status /= GALAHAD_ok .AND.                    &
               inform%CQP_inform%status /= GALAHAD_error_ill_conditioned .AND. &
               inform%CQP_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD CQP solve error status = ', I0 )" ) prefix,      &
              inform%CQP_inform%status
            inform%status = GALAHAD_error_cqp ; GO TO 800
          END IF

!  == DQP ==

        CASE ( 'dqp', 'DQP' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD DQP solver used **' )" ) prefix
          CALL DQP_solve( prob, data, control%DQP_control, inform%DQP_inform,  &
                          C_stat, B_stat )
          IF ( inform%DQP_inform%status /= GALAHAD_ok .AND.                    &
               inform%DQP_inform%status /= GALAHAD_error_ill_conditioned .AND. &
               inform%DQP_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD DQP solve error status = ', I0 )" ) prefix,      &
              inform%DQP_inform%status
            inform%status = GALAHAD_error_dqp ; GO TO 800
          END IF

!  == CCQP ==

        CASE ( 'ccqp', 'CCQP' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD CCQP solver used **' )" ) prefix
          CALL CCQP_solve( prob, data, control%CCQP_control,                   &
                           inform%CCQP_inform, C_stat, B_stat )
          IF ( inform%CCQP_inform%status /= GALAHAD_ok .AND.                   &
               inform%CCQP_inform%status /= GALAHAD_error_ill_conditioned .AND.&
               inform%CCQP_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD CCQP solve error status = ', I0 )" ) prefix,     &
              inform%CCQP_inform%status
            inform%status = GALAHAD_error_ccqp ; GO TO 800
          END IF

!  = unavailable solver =

        CASE DEFAULT
          inform%status = GALAHAD_error_unknown_solver ; GO TO 800
        END SELECT

        inform%status = GALAHAD_ok
        CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
        inform%time%solve = inform%time%solve + time_end - time_now
        inform%time%clock_solve =                                              &
            inform%time%clock_solve + clock_end - clock_now

!  ---------------------
!  post-process the data
!  ---------------------

!  if the problem was scaled, unscale it

        IF ( scale > 0 ) THEN
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          CALL SCALE_recover( prob, data%SCALE_trans, data%SCALE_data,         &
                           control%SCALE_control, inform%SCALE_inform )
          CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
          inform%time%scale = inform%time%scale + time_end - time_now
          inform%time%clock_scale =                                            &
            inform%time%clock_scale + clock_end - clock_now
          IF ( inform%SCALE_inform%status < 0 ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, '  ERROR return from SCALE (status =', I0, ')' )" )        &
                prefix, inform%SCALE_inform%status
            inform%status = GALAHAD_error_scale ; GO TO 800
          END IF
        END IF
      ELSE
        inform%status = GALAHAD_ok
        inform%obj = prob%f
      END IF

!  restore from presolve

      IF ( presolve ) THEN
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        CALL PRESOLVE_restore( prob, data%PRESOLVE_control,                    &
                               inform%PRESOLVE_inform,                         &
                               data%PRESOLVE_data )
        IF ( inform%PRESOLVE_inform%status /= 0 .AND. printi )                 &
          WRITE( control%out, " ( /, A, ' Warning: status following',          &
       &  ' PRESOLVE_restore is ', I0, / )" )                                  &
           prefix, inform%PRESOLVE_inform%status
        CALL PRESOLVE_terminate( data%PRESOLVE_control,                        &
                                 inform%PRESOLVE_inform,                       &
                                 data%PRESOLVE_data )
        IF ( inform%PRESOLVE_inform%status /= 0 .AND. printi )                 &
          WRITE( control%out, " ( /, A, ' Warning: status following',          &
       &    ' PRESOLVE_terminate is ', I5, / ) " )                             &
          prefix, inform%PRESOLVE_inform%status
        CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
        inform%time%presolve = inform%time%presolve + time_end - time_now
        inform%time%clock_presolve =                                           &
          inform%time%clock_presolve + clock_end - clock_now
        IF ( printi ) WRITE( control%out,                                      &
         "( /, A, ' postprocessing time = ', F0.2)") prefix, time_end - time_now
      END IF

!  if the problem was scaled, unscale it

      IF ( scale < 0 ) THEN
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        CALL SCALE_recover( prob, data%SCALE_trans, data%SCALE_data,           &
                         control%SCALE_control, inform%SCALE_inform )
        CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
        inform%time%scale = inform%time%scale + time_end - time_now
        inform%time%clock_scale =                                              &
          inform%time%clock_scale + clock_end - clock_now
        IF ( inform%SCALE_inform%status < 0 ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, '  ERROR return from SCALE (status =', I0, ')' )" )          &
              prefix, inform%SCALE_inform%status
          inform%status = GALAHAD_error_scale ; GO TO 800
        END IF
      END IF

!  compute the constrain residuals

      CALL mop_AX( one, prob%A, prob%X( : prob%n ), zero, prob%C( : prob%m ),  &
                   symmetric = .FALSE., transpose = .FALSE. )

!  compute maximum contraint residual and complementary slackness

      inform%primal_infeasibility = zero ; inform%complementary_slackness = zero
      DO i = 1, prob%m
        val = prob%C( i )
        inform%primal_infeasibility = MAX( inform%primal_infeasibility,        &
           MAX( zero, prob%C_l( i ) - val,  val - prob%C_u( i ) ) )
        IF ( prob%C_l( i ) > - control%infinity ) THEN
          IF ( prob%C_u( i ) < control%infinity ) THEN
            inform%complementary_slackness =                                   &
              MAX( inform%complementary_slackness,                             &
                 MIN( ABS( ( prob%C_l( i ) - val ) * prob%Y( i ) ),            &
                      ABS( ( prob%C_u( i ) - val ) * prob%Y( i ) ) ) )
          ELSE
            inform%complementary_slackness =                                   &
              MAX( inform%complementary_slackness,                             &
                   ABS( ( prob%C_l( i ) - val ) * prob%Y( i ) ) )
          END IF
        ELSE IF ( prob%C_u( i ) < control%infinity ) THEN
          inform%complementary_slackness =                                     &
           MAX( inform%complementary_slackness,                                &
                ABS( ( prob%C_u( i ) - val ) * prob%Y( i ) ) )
        END IF
      END DO

      DO i = 1, prob%n
        val = prob%X( i )
        IF ( prob%X_l( i ) > - control%infinity ) THEN
          IF ( prob%X_u( i ) < control%infinity ) THEN
            inform%complementary_slackness =                                   &
              MAX( inform%complementary_slackness,                             &
                 MIN( ABS( ( prob%X_l( i ) - val ) * prob%Z( i ) ),            &
                      ABS( ( prob%X_u( i ) - val ) * prob%Z( i ) ) ) )
          ELSE
            inform%complementary_slackness =                                   &
              MAX( inform%complementary_slackness,                             &
                   ABS( ( prob%X_l( i ) - val ) * prob%Z( i ) ) )
          END IF
        ELSE IF ( prob%X_u( i ) < control%infinity ) THEN
          inform%complementary_slackness =                                     &
            MAX( inform%complementary_slackness,                               &
                 ABS( ( prob%X_u( i ) - val ) * prob%Z( i ) ) )
        END IF
      END DO

!  compute H * x

      IF ( lbfgs ) THEN
        CALL LMS_apply_lbfgs( prob%X( : prob%n ), prob%H_lm, i,                &
                              RESULT = data%SH( : prob%n ) )
      ELSE
        CALL mop_AX( one, prob%H, prob%X( : prob%n ), zero,                    &
                     data%SH( : prob%n ), symmetric = .TRUE.,                  &
                     transpose = .FALSE. )
      END IF

!  compute the objective function

      inform%obj = DOT_PRODUCT( prob%X( : prob%n ), prob%G( : prob%n ) )       &
         + half * DOT_PRODUCT( prob%X( : prob%n ), data%SH( : prob%n ) )       &
         + prob%f

!  compute the dual residual

      data%SH( : prob%n ) = data%SH( : prob%n ) - prob%Z( : prob%n )
      CALL mop_AX( - one, prob%A, prob%Y( : prob%m ), one, data%SH( : prob%n ),&
                   symmetric = .FALSE., transpose = .TRUE. )

      inform%dual_infeasibility = MAXVAL( ABS( data%SH( : prob%n ) ) )

!  return

  800 CONTINUE
      CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
      inform%time%total = inform%time%total + time_end - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_end - clock_start
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving QP_solve ' )" ) prefix

      RETURN

!  allocation error

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
!     CALL CPU_TIME( time_now ) ; inform%time%total = time_now - time_start
      IF ( printi ) WRITE( control%out,                                        &
        "( A, ' ** Message from -QP_solve-', /,  A,                            &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving QP_solve ' )" ) prefix

      RETURN

!  End of QP_solve

      END SUBROUTINE QP_solve

!-*-*-*-*-*-*-   Q P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE QP_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine QP_initialize
!   control see Subroutine QP_initialize
!   inform  see Subroutine QP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QP_data_type ), INTENT( INOUT ) :: data
      TYPE ( QP_control_type ), INTENT( IN ) :: control
      TYPE ( QP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by SCALE, PRESOLVE, QPA, QPB, QPC,
!  CQP, DQP and CCQP

      CALL SCALE_terminate( data%SCALE_data, control%SCALE_control,            &
                            inform%SCALE_inform, trans = data%SCALE_trans )
      IF ( inform%SCALE_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%SCALE_inform%alloc_status
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL PRESOLVE_terminate( data%PRESOLVE_control,                          &
                               inform%PRESOLVE_inform,                         &
                               data%PRESOLVE_data )
      IF ( inform%PRESOLVE_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
!       inform%alloc_status = inform%PRESOLVE_inform%alloc_status
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPA_terminate( data, control%QPA_control, inform%QPA_inform )
      IF ( inform%QPA_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%QPA_inform%alloc_status
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPB_terminate( data, control%QPB_control, inform%QPB_inform )
      IF ( inform%QPB_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%QPB_inform%alloc_status
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPC_terminate( data, control%QPC_control, inform%QPC_inform )
      IF ( inform%QPC_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%QPC_inform%alloc_status
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL CQP_terminate( data, control%CQP_control, inform%CQP_inform )
      IF ( inform%CQP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%CQP_inform%alloc_status
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL DQP_terminate( data, control%DQP_control, inform%DQP_inform )
      IF ( inform%DQP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%DQP_inform%alloc_status
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL CCQP_terminate( data, control%CCQP_control, inform%CCQP_inform )
      IF ( inform%CCQP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%CCQP_inform%alloc_status
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all remaing allocated arrays

      array_name = 'qp: data%SH'
      CALL SPACE_dealloc_array( data%SH,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qp: data%C_stat'
      CALL SPACE_dealloc_array( data%C_stat,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qp: data%B_stat'
      CALL SPACE_dealloc_array( data%B_stat,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine QP_terminate

      END SUBROUTINE QP_terminate

!  End of module QP

   END MODULE GALAHAD_QP_double











