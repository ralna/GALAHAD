! THIS VERSION: GALAHAD 2.4 - 08/12/2009 AT 12:00 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ C Q P S   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. January 1st 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_CQPS_double

!               ---------------------------------------------
!               |                                           |
!               | Solve the convex quadratic program        |
!               |                                           |
!               |    minimize     1/2 x(T) H x + g(T) x + f |
!               |    subject to     c_l <= A x <= c_u       |
!               |                   x_l <=  x  <= x_u       |
!               |                                           |
!               | using Spelucci's bound-constrained exact  |
!               | penalty reformuation and a preconditioned |
!               | projected conjugate-gradient BQP method   |
!               |                                           |
!               ---------------------------------------------

!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_SYMBOLS
     USE GALAHAD_NORMS_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_SORT_double
     USE GALAHAD_QPT_double
     USE GALAHAD_QPD_double, ONLY: QPD_SIF
     USE GALAHAD_BQP_double
     USE GALAHAD_BQPB_double
     USE GALAHAD_PSLS_double
     USE GALAHAD_SPECFILE_double

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: CQPS_initialize, CQPS_read_specfile, CQPS_solve, CQPS_terminate,&
               CQPS_reverse_h_type, CQPS_reverse_a_type, CQPS_data_type,       &
               GALAHAD_userdata_type, QPT_problem_type,                        &
               SMT_type, SMT_put, SMT_get

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
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: CQPS_control_type
        
!  unit number for error and warning diagnostics
      
       INTEGER :: error = 6

!  general output unit number

       INTEGER :: out  = 6

!  the level of output required

       INTEGER :: print_level = 0

!  the unit number to write generated SIF file describing the current problem

       INTEGER :: sif_file_device = 51

!  any bound larger than infinity in modulus will be regarded as infinite 

       REAL ( KIND = wp ) :: infinity = ten ** 19

!  any pair of constraint bounds (x_l,x_u) that are closer than i
!   dentical_bounds_tol will be reset to the average of their values
!
       REAL ( KIND = wp ) :: identical_bounds_tol = epsmch

!  initial value of the primal penalty parameter, rho
!
       REAL ( KIND = wp ) :: initial_rho = one

!  initial value of the dual penalty parameter, eta
!
       REAL ( KIND = wp ) :: initial_eta = two

!  the required accuracy for the primal infeasibility

       REAL ( KIND = wp ) :: stop_p = ten ** ( - 6 )

!  the required accuracy for the dual infeasibility

       REAL ( KIND = wp ) :: stop_d = ten ** ( - 6 )

!  the required accuracy for the complementary slackness

       REAL ( KIND = wp ) :: stop_c = ten ** ( - 6 )

!  the maximum CPU time allowed (-ve = no limit)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!  choose between projection (BQP) and interior-point (BQPB) BQP solvers

       LOGICAL :: use_bqp = .TRUE.

!  if space_critical is true, every effort will be made to use as little
!   space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!  if deallocate_error_fatal is true, any array/pointer deallocation error
!    will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!  if generate_sif_file is true, a SIF file describing the current problem
!  will be generated

       LOGICAL :: generate_sif_file = .FALSE.

!  name (max 30 characters) of generated SIF file containing input problem

       CHARACTER ( LEN = 30 ) :: sif_file_name =                               &
         "CQPSPROB.SIF"  // REPEAT( ' ', 18 )

!  all output lines will be prefixed by a string (max 30 characters)
!    prefix(2:LEN(TRIM(%prefix))-1)
!   where prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'
!
       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for BQP, BQPB and PSLS

       TYPE ( BQP_control_type ) :: BQP_control
       TYPE ( BQPB_control_type ) :: BQPB_control
       TYPE ( PSLS_control_type ) :: PSLS_control
     END TYPE CQPS_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: CQPS_time_type

!  total time

       REAL :: total = 0.0

!  time for the analysis phase

       REAL :: analyse = 0.0

!  time for the factorization phase

       REAL :: factorize = 0.0

!  time for the linear solution phase

       REAL :: solve = 0.0
     END TYPE CQPS_time_type
   
!  - - - - - - - - - - - - - - - - - - - - - - - 
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: CQPS_inform_type

!  reported return status:
!     0  success
!    -1  allocation error
!    -2  deallocation error
!    -3  matrix data faulty (%n < 1, %ne < 0)
!   -20  alegedly +ve definite matrix is not

       INTEGER :: status = 1

!  STAT value after allocate failure

       INTEGER :: alloc_status = 0

!  status return from factorization

       INTEGER :: factorization_status = 0

!  current value of the objective function

       REAL ( KIND = wp ) :: obj = infinity

!  current value of the norm of primal infeasibility

       REAL ( KIND = wp ) :: primal_infeasibility = infinity

!  current value of the norm of dual infeasibility

       REAL ( KIND = wp ) :: dual_infeasibility = infinity

!  current value of the norm of complementary slackness

       REAL ( KIND = wp ) :: complementary_slackness = infinity

!  name of array which provoked an allocate failure

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  times for various stages

       TYPE ( CQPS_time_type ) :: time

!  inform values from BQP, BQPB and PSLS

       TYPE ( BQP_inform_type ) :: BQP_inform
       TYPE ( BQPB_inform_type ) :: BQPB_inform
       TYPE ( PSLS_inform_type ) :: PSLS_inform
     END TYPE CQPS_inform_type

!  - - - - - - - - - - - -
!   reverse_h derived type
!  - - - - - - - - - - - -

     TYPE :: CQPS_reverse_h_type
       INTEGER :: nz_v_start, nz_v_end, nz_prod_end
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_v, NZ_prod
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V, PROD
     END TYPE CQPS_reverse_h_type

!  - - - - - - - - - - - -
!   reverse_a derived type
!  - - - - - - - - - - - -

     TYPE :: CQPS_reverse_a_type
       INTEGER :: nz_v_start, nz_v_end, nz_prod_end
       LOGICAL :: transpose
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_v, NZ_prod
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V, PROD
     END TYPE CQPS_reverse_a_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE :: CQPS_data_type
       INTEGER :: out, error, print_level, start_print, stop_print, print_gap
       INTEGER :: n_c_i, n_y_l, n_y_u, n_z_l, n_z_u
       INTEGER :: st_c_i, st_y, st_y_l, st_y_u, st_z_l, st_z_u
       INTEGER :: n_sub, branch
       REAL :: time_start
       REAL ( KIND = wp ) :: eta, rho, norm_g_2
       LOGICAL :: printt, printi, printm, printw, printd, printe 
       LOGICAL :: reverse_h, explicit_h, use_hprod
       LOGICAL :: reverse_a, explicit_a, use_aprod
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: SUB, B_stat
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_v, NZ_hprod, NZ_aprod
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V, PROD, HPROD, APROD
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PPROD, U, bqp_V
       TYPE ( CQPS_control_type ) :: control
       TYPE ( SMT_type ) :: H
       TYPE ( QPT_problem_type ) :: bqp
       TYPE ( BQP_data_type ) :: BQP_data
       TYPE ( BQPB_data_type ) :: BQPB_data
       TYPE ( PSLS_data_type ) :: PSLS_data
       TYPE ( GALAHAD_userdata_type ) :: bqp_userdata
       TYPE ( BQP_reverse_type ) :: bqp_reverse
       TYPE ( BQPB_reverse_type ) :: bqpb_reverse
     END TYPE CQPS_data_type

!-------------------------------
!   I n t e r f a c e  B l o c k
!-------------------------------

!     INTERFACE TWO_NORM
!
!       FUNCTION SNRM2( n, X, incx )
!       REAL :: SNRM2
!       INTEGER, INTENT( IN ) :: n, incx
!       REAL, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
!       END FUNCTION SNRM2
!
!       FUNCTION DNRM2( n, X, incx )
!       DOUBLE PRECISION :: DNRM2
!       INTEGER, INTENT( IN ) :: n, incx
!       DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
!       END FUNCTION DNRM2
!       
!     END INTERFACE 

   CONTAINS

!-*-*-*-*-*-   C Q P S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE CQPS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for CQPS. This routine should be called before
!  CQPS_solve
! 
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

     TYPE ( CQPS_data_type ), INTENT( INOUT ) :: data
     TYPE ( CQPS_control_type ), INTENT( OUT ) :: control
     TYPE ( CQPS_inform_type ), INTENT( OUT ) :: inform     

     inform%status = GALAHAD_ok       

!  initialize control parameters for BQP (see GALAHAD_BQP for details)

     CALL BQP_initialize( data%BQP_data, control%BQP_control,                  &
                          inform%BQP_inform )
     control%BQP_control%prefix = '" - BQP:"                     '

!  initialize control parameters for BQPB (see GALAHAD_BQPB for details)

     CALL BQPB_initialize( data%BQPB_data, control%BQPB_control,               &
                           inform%BQPB_inform )
     control%BQPB_control%prefix = '" - BQPB:"                    '

!  initialize control parameters for PSLS (see GALAHAD_PSLS for details)

     CALL PSLS_initialize( data%PSLS_data, control%PSLS_control,               &
                           inform%PSLS_inform )
     control%PSLS_control%prefix = '" - PSLS:"                    '
     control%PSLS_control%preconditioner = 1

!  added here to prevent for compiler bugs 

     control%stop_p = epsmch ** 0.33_wp
     control%stop_d = epsmch ** 0.33_wp
     control%stop_c = epsmch ** 0.33_wp

     RETURN

!  End of CQPS_initialize

     END SUBROUTINE CQPS_initialize

!-*-*-*-*-   C Q P S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

     SUBROUTINE CQPS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by CQPS_initialize could (roughly) 
!  have been set as:

! BEGIN CQPS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  sif-file-device                                   52
!  infinity-value                                    1.0D+19
!  identical-bounds-tolerance                        1.0D-15
!  initial-primal-penalty-parameter                  -1.0
!  initial-dual-penalty-parameter                    -1.0
!  primal-accuracy-required                          1.0D-5
!  dual-accuracy-required                            1.0D-5
!  complementary-slackness-accuracy-required         1.0D-5
!  maximum-cpu-time-limit                            -1.0
!  use-projection-based-bqp-solver                   T
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  sif-file-name                                     CQPSPROB.SIF
! END CQPS SPECIFICATIONS

!  Dummy arguments

     TYPE ( CQPS_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: sif_file_device = print_level + 1
     INTEGER, PARAMETER :: maxit = sif_file_device + 1
     INTEGER, PARAMETER :: infinity = maxit + 1
     INTEGER, PARAMETER :: identical_bounds_tol = infinity + 1
     INTEGER, PARAMETER :: initial_rho = identical_bounds_tol + 1
     INTEGER, PARAMETER :: initial_eta = initial_rho + 1
     INTEGER, PARAMETER :: stop_p = initial_eta + 1
     INTEGER, PARAMETER :: stop_d = stop_p + 1
     INTEGER, PARAMETER :: stop_c = stop_d + 1
     INTEGER, PARAMETER :: cpu_time_limit = stop_c + 1
     INTEGER, PARAMETER :: use_bqp = cpu_time_limit + 1
     INTEGER, PARAMETER :: space_critical = use_bqp + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: sif_file_name = generate_sif_file + 1
     INTEGER, PARAMETER :: prefix = sif_file_name + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'CQPS'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( sif_file_device )%keyword = 'sif-file-device'

!  Real key-words

     spec( infinity )%keyword = 'infinity-value'
     spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
     spec( initial_rho )%keyword = 'initial-primal-penalty-parameter'
     spec( initial_eta )%keyword = 'initial-dual-penalty-parameter'
     spec( stop_p )%keyword = 'primal-accuracy-required'
     spec( stop_d )%keyword = 'dual-accuracy-required'
     spec( stop_c )%keyword = 'complementary-slackness-accuracy-required'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'

!  Logical key-words

     spec( use_bqp )%keyword = 'use-projection-based-bqp-solver'
     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
     spec( generate_sif_file )%keyword = 'generate-sif-file'

!  Character key-words

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
     CALL SPECFILE_assign_value( spec( sif_file_device ),                      &
                                 control%sif_file_device,                      &
                                 control%error )

!  Set real value

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( identical_bounds_tol ),                 &
                                 control%identical_bounds_tol,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_rho ),                          &
                                 control%initial_rho,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_eta ),                          &
                                 control%initial_eta,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_p ),                               &
                                 control%stop_p,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_d ),                               &
                                 control%stop_d,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_c ),                               &
                                 control%stop_c,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( use_bqp ),                              &
                                 control%use_bqp,                              &
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

!  Set character value

     CALL SPECFILE_assign_value( spec( sif_file_name ),                        &
                                 control%sif_file_name,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  Read the specfiles for BQP, BQPB and PSLS

     IF ( PRESENT( alt_specname ) ) THEN
       CALL BQP_read_specfile( control%BQP_control, device,                    &
                                alt_specname = TRIM( alt_specname ) // '-BQP' )
       CALL BQPB_read_specfile( control%BQPB_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-BQPB' )
       CALL PSLS_read_specfile( control%PSLS_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-PSLS' )
     ELSE
       CALL BQP_read_specfile( control%BQP_control, device )
       CALL BQPB_read_specfile( control%BQPB_control, device )
       CALL PSLS_read_specfile( control%PSLS_control, device )
     END IF

     RETURN

     END SUBROUTINE CQPS_read_specfile

!-*-*-*-*-*-*-*-*-   C Q P S _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-

     SUBROUTINE CQPS_solve( prob, C_stat, B_stat, data, control, inform,       &
                           userdata, reverse_h, reverse_a,                     &
                           eval_HPROD, eval_APROD )

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
!  constant, g is an n-vector, H is a symmetric matrix, 
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!  The subroutine is particularly appropriate when A and H are sparse
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
!    to be solved since the last call to CQPS_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!                 
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
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
!    If H is not available explicitly, matrix-vector products must be
!      provided by the user using either reverse communication
!      (see reverse_h below) or a provided subroutine (see eval_HPROD below).
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
!    If A is not available explicitly, matrix-vector products with
!      A and its transpose must be provided by the user using either 
!      reverse communication (see reverse_a below) or a provided subroutine 
!      (see eval_APROD below).
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
!  C_stat is a INTEGER array of length m, which may be set by the user
!   on entry to QPA_solve to indicate which of the constraints are to
!   be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; C_stat
!   need not be set if control%cold_start is nonzero. On exit,
!   C_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are 
!   C_stat( i ) < 0, the i-th constraint is in the working set, 
!                    on its lower bound, 
!               > 0, the i-th constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th constraint is not in the working set
!
!  B_stat is a INTEGER array of length n, which may be set by the user
!   on entry to QPA_solve to indicate which of the simple bound constraints 
!   are to be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; B_stat
!   need not be set if control%cold_start is nonzero. On exit,
!   B_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are 
!   B_stat( i ) < 0, the i-th bound constraint is in the working set, 
!                    on its lower bound, 
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  data is a structure of type CQPS_data_type which holds private internal data
!
!  control is a structure of type CQPS_control_type that controls the 
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to CQPS_initialize. See CQPS_initialize 
!   for details
!
!  inform is a structure of type CQPS_inform_type that provides 
!    information on exit from CQPS_solve. The component status 
!    has possible values:
!  
!     0 Normal termination with a locally optimal solution.
!
!     2 The product H * v of the Hessian H with a given output vector v
!       is required from the user. The vector v will be stored in reverse_h%V
!       and the product H * v must be returned in reverse_h%PROD, and 
!       BQP_solve re-entered with all other arguments unchanged. 
!
!     3 The product A * v of the Jacobian A with a given output vector v
!       is required from the user. The vector v will be stored in reverse_a%V
!       and the product A * v must be returned in reverse_a%PROD, and 
!       BQP_solve re-entered with all other arguments unchanged. 
!
!     4 The product A^T * v of the transpose of the Jacobian A^T with a given 
!       output vector v is required from the user. The vector v will be stored 
!       in reverse_a%V and the product A^T * v must be returned in 
!       reverse_a%PROD, and BQP_solve re-entered with all other arguments 
!       unchanged. 
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
!        prob%H%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE', 'DIAGONAL' }
!       has been violated.
!
!    -4 The bound constraints are inconsistent.
!
!    -5 The constraints appear to have no feasible point.
!
!    -7 The objective function appears to be unbounded from below on the
!       feasible set.
!
!    -9 The factorization failed; the return status from the factorization
!       package is given in the component factorization_status.
!      
!    -13 The problem is so ill-conditoned that further progress is impossible.  
!
!    -16 The step is too small to make further impact.
!
!    -17 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of 
!       a badly scaled problem.
!
!    -18 Too much CPU time has passed. This may happen if control%cpu_time_limit
!        is too small, but may also be symptomatic of a badly scaled problem.
!
!    -23 an entry from the strict upper triangle of H has been input.
!
!  On exit from CQPS_solve, other components of inform give the 
!  following:
!
!     alloc_status = The status of the last attempted allocation/deallocation 
!     factorization_integer = The total integer workspace required for the 
!       factorization.
!     factorization_real = The total real workspace required for the 
!       factorization.
!     nfacts = The total number of factorizations performed.
!     nmods = The total number of factorizations which were modified to 
!       ensure that the matrix was an appropriate preconditioner. 
!     factorization_status = the return status from the matrix factorization
!       package.   
!     obj = the value of the objective function at the best estimate of the 
!       solution determined by CQPS_solve.
!     non_negligible_pivot = the smallest pivot which was not judged to be
!       zero when detecting linearly dependent constraints
!     bad_alloc = the name of the array for which an allocation/deallocation
!       error ocurred
!     time%total = the total time spent in the package.
!     time%preprocess = the time spent preprocessing the problem.
!     time%find_dependent = the time spent detecting linear dependencies
!     time%analyse = the time spent analysing the required matrices prior to
!       factorization.
!     time%factorize = the time spent factorizing the required matrices.
!     time%solve = the time spent computing the search direction.
!
!  userdata is a scalar variable of type GALAHAD_userdata_type which may be 
!   used to pass user data to and from the eval_* subroutines (see below)
!   Available coomponents which may be allocated as required are:
!
!    integer is a rank-one allocatable array of type default integer.
!    real is a rank-one allocatable array of type default real
!    complex is a rank-one allocatable array of type default comple.
!    character is a rank-one allocatable array of type default character.
!    logical is a rank-one allocatable array of type default logical.
!    integer_pointer is a rank-one pointer array of type default integer.
!    real_pointer is a rank-one pointer array of type default  real
!    complex_pointer is a rank-one pointer array of type default complex.
!    character_pointer is a rank-one pointer array of type default character.
!    logical_pointer is a rank-one pointer array of type default logical.
!
!  reverse_h is an OPTIONAL structure of type CQPS_reverse_h_type which is used 
!   to pass intermediate data to and from CQPS_solve. This will only be 
!   necessary if reverse-communication is to be used to form matrix-vector 
!   products of the form H * v or preconditioning steps of the form P * v. If 
!   reverse is present (and eval_HPROD is absent), reverse communication
!   will be used and the user must monitor the value of inform%status 
!   (see above) to await instructions about required matrix-vector products.
!
!  reverse_a is an OPTIONAL structure of type CQPS_reverse_a_type which is used
!   to pass intermediate data to and from CQPS_solve. This will only be 
!   necessary if reverse-communication is to be used to form matrix-vector 
!   products of the form A * v or A^T v. If reverse is present (and 
!   eval_APROD is absent), reverse communication will be used and the user 
!   must monitor the value of inform%status (see above) to await instructions 
!   about required matrix-vector products.
!
!  eval_HPROD is an OPTIONAL subroutine which if present must have the arguments
!   given below (see the interface blocks). The product H * v of the given 
!   matrix H and vector v stored in V must be returned in PROD; only the 
!   components NZ_v( nz_v_start : nz_v_end ) of V are nonzero. If either of 
!   the optional argeuments NZ_prod or nz_prod_end are absent, the WHOLE of H v 
!   including zeros should be returned in PROD. If NZ_prod and nz_prod_end are 
!   present, the NONZEROS in the product H * v must be placed in their 
!   appropriate comnponents of reverse%PROD, while a list of indices of the 
!   nonzeos placed in NZ_prod( 1 : nz_prod_end ). In both cases, the status 
!   variable should be set to 0 unless the product is impossible in which
!   case status should be set to a nonzero value. If eval_HPROD is not
!   present, CQPS_solve will either return to the user each time an evaluation 
!   is required (see reverse_h above) or form the product directly from 
!   user-provided %H.
!
!  eval_APROD is an OPTIONAL subroutine which if present must have the arguments
!   given below (see the interface blocks). The product A * v or A^T * v
!   of the given matrix A (or its trnspose if transpose =.TRUE.) and vector 
!   v stored in V must be returned in PROD. If eval_APROD is not present, 
!   CQPS_solve will either return to the user each time an evaluation is 
!   required (see reverse_a above) or form the product directly from 
!   user-provided %A.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
     INTEGER, INTENT( INOUT ), DIMENSION( prob%m ) :: C_stat
     INTEGER, INTENT( INOUT ), DIMENSION( prob%n ) :: B_stat
     TYPE ( CQPS_data_type ), INTENT( INOUT ) :: data
     TYPE ( CQPS_control_type ), INTENT( IN ) :: control
     TYPE ( CQPS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     TYPE ( CQPS_reverse_h_type ), OPTIONAL, INTENT( INOUT ) :: reverse_h
     TYPE ( CQPS_reverse_a_type ), OPTIONAL, INTENT( INOUT ) :: reverse_a
     OPTIONAL :: eval_HPROD, eval_APROD

!  interface blocks

     INTERFACE
       SUBROUTINE eval_HPROD( status, userdata, V, PROD, NZ_v, nz_v_start,     &
                              nz_v_end, NZ_prod, nz_prod_end )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: PROD
       INTEGER, OPTIONAL, INTENT( IN ) :: nz_v_start, nz_v_end
       INTEGER, OPTIONAL, INTENT( INOUT ) :: nz_prod_end
       INTEGER, DIMENSION( : ), OPTIONAL, INTENT( IN ) :: NZ_v
       INTEGER, DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: NZ_prod
       END SUBROUTINE eval_HPROD
     END INTERFACE
   
     INTERFACE
       SUBROUTINE eval_APROD( status, userdata, transpose, V, PROD )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: PROD
       LOGICAL, INTENT( IN ) :: transpose
       END SUBROUTINE eval_APROD
     END INTERFACE
   
!  Local variables

     INTEGER :: i, j, k, l, nnz
     INTEGER :: c_i, y, y_l, y_u, z_l, z_u
     REAL :: time
     REAL ( KIND = wp ) :: val, av_bnd, g_i, s_i, v_i, prod, rho_i
     REAL ( KIND = wp ) :: eta_new, ratio_eta, one_minus_ratio_eta
     REAL ( KIND = wp ) :: rho_new, ratio_rho
     LOGICAL :: reset_bnd
     CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Enter or re-enter the package and jump to appropriate re-entry point

     IF ( inform%status == 1 ) data%branch = 100

     SELECT CASE ( data%branch )
     CASE ( 100 ) ; GO TO 100
     CASE ( 200 ) ; GO TO 200
     CASE ( 300 ) ; GO TO 300
     CASE ( 410 ) ; GO TO 410
     CASE ( 420 ) ; GO TO 420
     CASE ( 430 ) ; GO TO 430
     CASE ( 440 ) ; GO TO 440
     CASE ( 450 ) ; GO TO 450
     CASE ( 460 ) ; GO TO 460
     CASE ( 610 ) ; GO TO 610
     CASE ( 620 ) ; GO TO 620
     CASE ( 630 ) ; GO TO 630
     CASE DEFAULT
       write( 6, * ) ' branch should not be here ... '
       stop
     END SELECT

 100 CONTINUE

     IF ( control%out > 0 .AND. control%print_level >= 5 )                     &
       WRITE( control%out, 2000 ) prefix, ' entering '

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed 

     IF ( control%generate_sif_file ) THEN
       CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,     &
                     control%infinity, .TRUE. )
     END IF

!  SIF file generated
! -------------------------------------------------------------------

     inform%status = - 1001   ! indicates under development

!  Initialize time

     CALL CPU_TIME( data%time_start )

!  Set initial timing breakdowns

     inform%time%total = 0.0 ; inform%time%analyse = 0.0
     inform%time%factorize = 0.0 ; inform%time%solve = 0.0

     data%use_hprod = PRESENT( eval_HPROD )
     data%use_aprod = PRESENT( eval_APROD )
     data%reverse_h = PRESENT( reverse_h ) .AND. .NOT. data%use_hprod
     data%reverse_a = PRESENT( reverse_a ) .AND. .NOT. data%use_aprod
     data%explicit_h = .NOT. ( data%use_hprod .OR. data%reverse_h )
     data%explicit_a = .NOT. ( data%use_aprod .OR. data%reverse_a )

!  start setting control parameters

     data%printe = control%error > 0 .AND. control%print_level >= 1
     data%printi = control%out > 0 .AND. control%print_level >= 1
     data%printt = control%out > 0 .AND. control%print_level >= 2
     data%printm = control%out > 0 .AND. control%print_level >= 3
     data%printw = control%out > 0 .AND. control%print_level >= 4
     data%printd = control%out > 0 .AND. control%print_level >= 5

!  Ensure that input parameters are within allowed ranges

     IF ( prob%n <= 0 .OR. prob%m < 0 ) THEN
       inform%status = GALAHAD_error_restrictions
       GO TO 910
     END IF

     IF ( data%explicit_h ) THEN
       IF ( .NOT. QPT_keyword_H( prob%H%type ) ) THEN
         inform%status = GALAHAD_error_restrictions
         GO TO 910
       END IF 
     END IF 

     IF ( data%explicit_a ) THEN
       IF ( .NOT. QPT_keyword_A( prob%A%type ) ) THEN
         inform%status = GALAHAD_error_restrictions
         GO TO 910
       END IF 
     END IF 

!  If required, write out problem 

     IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
       WRITE( control%out, "( ' n, m = ', I0, 1X, I0 )" ) prob%n, prob%m
       WRITE( control%out, "( ' f = ', ES12.4 )" ) prob%f
       WRITE( control%out, "( ' G = ', /, ( 5ES12.4 ) )" ) prob%G( : prob%n )
       IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
         WRITE( control%out, "( ' H (diagonal) = ', /, ( 5ES12.4 ) )" )        &
           prob%H%val( : prob%n )
       ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
         WRITE( control%out, "( ' H (dense) = ', /, ( 5ES12.4 ) )" )           &
           prob%H%val( : prob%n * ( prob%n + 1 ) / 2 )
       ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
         WRITE( control%out, "( ' H (row-wise) = ' )" )
         DO i = 1, prob%m
           WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                    &
             ( i, prob%H%col( j ), prob%H%val( j ),                            &
               j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1 )
         END DO
       ELSE
         WRITE( control%out, "( ' H (co-ordinate) = ' )" )
         WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                      &
         ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ), i = 1, prob%H%ne)
       END IF
       WRITE( control%out, "( ' X_l = ', /, ( 5ES12.4 ) )" )                   &
         prob%X_l( : prob%n )
       WRITE( control%out, "( ' X_u = ', /, ( 5ES12.4 ) )" )                   &
         prob%X_u( : prob%n )
       IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
         WRITE( control%out, "( ' A (dense) = ', /, ( 5ES12.4 ) )" )           &
           prob%A%val( : prob%n * prob%m )
       ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
         WRITE( control%out, "( ' A (row-wise) = ' )" )
         DO i = 1, prob%m
           WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                    &
             ( i, prob%A%col( j ), prob%A%val( j ),                            &
               j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1 )
         END DO
       ELSE
         WRITE( control%out, "( ' A (co-ordinate) = ' )" )
         WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                      &
         ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne)
       END IF
       WRITE( control%out, "( ' C_l = ', /, ( 5ES12.4 ) )" )                   &
         prob%C_l( : prob%m )
       WRITE( control%out, "( ' C_u = ', /, ( 5ES12.4 ) )" )                   &
         prob%C_u( : prob%m )
     END IF

!  Check that problem bounds are consistent; reassign any pair of bounds
!  that are "essentially" the same

     data%n_sub = 0
     reset_bnd = .FALSE.
     DO i = 1, prob%n
       IF ( prob%X_l( i ) - prob%X_u( i ) > control%identical_bounds_tol ) THEN
         inform%status = GALAHAD_error_bad_bounds
         GO TO 910 
       ELSE IF ( prob%X_u( i ) == prob%X_l( i ) ) THEN
       ELSE IF ( prob%X_u( i ) - prob%X_l( i )                                 &
                 <= control%identical_bounds_tol ) THEN
         av_bnd = half * ( prob%X_l( i ) + prob%X_u( i ) )
         prob%X_l( i ) = av_bnd ; prob%X_u( i ) = av_bnd
         reset_bnd = .TRUE.
       ELSE
         data%n_sub = data%n_sub + 1
       END IF
     END DO   
     IF ( reset_bnd .AND. data%printi ) WRITE( control%out,                    &
       "( ' ', /, '   **  Warning: one or more variable bounds reset ' )" )

     reset_bnd = .FALSE.
     DO i = 1, prob%m
       IF ( prob%C_l( i ) - prob%C_u( i ) > control%identical_bounds_tol ) THEN
         inform%status = GALAHAD_error_bad_bounds
         GO TO 910 
       ELSE IF ( prob%C_u( i ) == prob%C_l( i ) ) THEN
       ELSE IF ( prob%C_u( i ) - prob%C_l( i )                                 &
                 <= control%identical_bounds_tol ) THEN
         av_bnd = half * ( prob%C_l( i ) + prob%C_u( i ) )
         prob%C_l( i ) = av_bnd ; prob%C_u( i ) = av_bnd
         reset_bnd = .TRUE.
       END IF
     END DO   
     IF ( reset_bnd .AND. data%printi ) WRITE( control%out,                    &
       "( ' ', /, '   **  Warning: one or more constraint bounds reset ' )" )

     IF ( data%explicit_h ) THEN

!  allocate space to record the list of non-fixed variables

       array_name = 'cqps: data%SUB'
       CALL SPACE_resize_array( data%n_sub, data%SUB, inform%status,           &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  builld the list of non-fixed variables

       data%n_sub = 0
       DO i = 1, prob%n
         IF ( prob%X_l( i ) < prob%X_u( i ) ) THEN
           data%n_sub = data%n_sub + 1
           data%SUB( data%n_sub ) = i
         END IF
       END DO   

!  Form and factorize an approximation P(inverse) to H

       prob%H%n = prob%n
       CALL PSLS_form_and_factorize( prob%H, data%PSLS_data,                   &
                                     control%PSLS_control, inform%PSLS_inform, &
                                     SUB = data%SUB( : data%n_sub ) )

!  Build a copy of H stored by rows (both lower and upper triangles)

!  allocate space to record row lengths

       array_name = 'cqps: data%H%ptr'
       CALL SPACE_resize_array( prob%n + 1, data%H%ptr, inform%status,         &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  compute the number of nonzeros in each row

       SELECT CASE ( SMT_get( prob%H%type ) )
       CASE ( 'COORDINATE' )
         data%H%ptr = 0
         DO l = 1, prob%H%ne
           i = prob%H%row( l ) ; j = prob%H%col( l )
           data%H%ptr( i ) = data%H%ptr( i ) + 1
           IF ( i /= j ) data%H%ptr( j ) = data%H%ptr( j ) + 1
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         data%H%ptr = 0
         DO i = 1, prob%n
           DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
             j = prob%H%col( l )
             data%H%ptr( i ) = data%H%ptr( i ) + 1
             IF ( i /= j ) data%H%ptr( j ) = data%H%ptr( j ) + 1
           END DO
         END DO
       CASE ( 'DENSE' )
         data%H%ptr = prob%n
       CASE ( 'DIAGONAL' )
         data%H%ptr = 1
       END SELECT

!  set starting addresses for each row in the row-wise scheme

       nnz = 1
       DO i = 1, prob%n
         l = data%H%ptr( i )
         data%H%ptr( i ) = nnz
         nnz = nnz + l
       END DO
       data%H%ptr( prob%n + 1 ) = nnz

!  allocate space to hold the column indices and values in the row-wise scheme

       array_name = 'cqps: data%H%col'
       CALL SPACE_resize_array( nnz, data%H%col, inform%status,                &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: data%H%val'
       CALL SPACE_resize_array( nnz, data%H%val, inform%status,                &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  now copy H into the row-wise scheme

       SELECT CASE ( SMT_get( prob%H%type ) )
       CASE ( 'COORDINATE' )
         DO l = 1, prob%H%ne
           i = prob%H%row( l ) ; j = prob%H%col( l ) ; val = prob%H%val( l )
           data%H%col( data%H%ptr( i ) ) = j
           data%H%val( data%H%ptr( i ) ) = val
           data%H%ptr( i ) = data%H%ptr( i ) + 1
           IF ( i /= j ) THEN
             data%H%col( data%H%ptr( j ) ) = i
             data%H%val( data%H%ptr( j ) ) = val
             data%H%ptr( j ) = data%H%ptr( j ) + 1
           END IF
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, prob%n
           DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
             j = prob%H%col( l ) ; val = prob%H%val( l )
             data%H%col( data%H%ptr( i ) ) = j
             data%H%val( data%H%ptr( i ) ) = val
             data%H%ptr( i ) = data%H%ptr( i ) + 1
             IF ( i /= j ) THEN
               data%H%col( data%H%ptr( j ) ) = i
               data%H%val( data%H%ptr( j ) ) = val
               data%H%ptr( j ) = data%H%ptr( j ) + 1
             END IF
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, prob%n
           DO j = 1, i
             l = l + 1
             val = prob%H%val( l )
             data%H%col( data%H%ptr( i ) ) = j
             data%H%val( data%H%ptr( i ) ) = val
             data%H%ptr( i ) = data%H%ptr( i ) + 1
             IF ( i /= j ) THEN
               data%H%col( data%H%ptr( j ) ) = i
               data%H%val( data%H%ptr( j ) ) = val
               data%H%ptr( j ) = data%H%ptr( j ) + 1
             END IF
           END DO
         END DO
       CASE ( 'DIAGONAL' )
         DO i = 1, prob%n
           data%H%col( data%H%ptr( i ) ) = i
           data%H%val( data%H%ptr( i ) ) =  prob%H%val( i )
           data%H%ptr( i ) = data%H%ptr( i ) + 1
         END DO
       END SELECT

!  finally, reset the starting addresses for each row

       DO i = prob%n, 1, - 1
         data%H%ptr( i + 1 ) = data%H%ptr( i )
       END DO
       data%H%ptr( 1 ) = 1
     END IF

!  allocate workspace arrays

     array_name = 'cqps: data%U'
     CALL SPACE_resize_array( prob%n, data%U, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'cqps: data%PPROD'
     CALL SPACE_resize_array( prob%n, data%PPROD, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     IF ( data%reverse_h ) THEN
       array_name = 'cqps: reverse_h%NZ_v'
       CALL SPACE_resize_array( prob%n, reverse_h%NZ_v, inform%status,         &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: reverse_h%NZ_prod'
       CALL SPACE_resize_array( prob%n, reverse_h%NZ_prod, inform%status,      &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: reverse_h%V'
       CALL SPACE_resize_array( prob%n, reverse_h%V, inform%status,            &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: reverse_h%PROD'
       CALL SPACE_resize_array( prob%n, reverse_h%PROD, inform%status,         &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     ELSE
       array_name = 'cqps: data%NZ_v'
       CALL SPACE_resize_array( prob%n, data%NZ_v, inform%status,              &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: data%NZ_hprod'
       CALL SPACE_resize_array( prob%n, data%NZ_hprod, inform%status,          &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: data%V'
       CALL SPACE_resize_array( prob%n, data%V, inform%status,                 &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: data%HPROD'
       CALL SPACE_resize_array( prob%n, data%HPROD, inform%status,             &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

     IF ( data%reverse_a ) THEN
       array_name = 'cqps: reverse_a%NZ_v'
       CALL SPACE_resize_array( MAX( prob%n, prob%m ), reverse_a%NZ_v,         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: reverse_a%NZ_prod'
       CALL SPACE_resize_array( MAX( prob%n, prob%m ), reverse_a%NZ_prod,      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: reverse_a%V'
       CALL SPACE_resize_array( MAX( prob%n, prob%m ), reverse_a%V,            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: reverse_a%PROD'
       CALL SPACE_resize_array( MAX( prob%n, prob%m ), reverse_a%PROD,         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     ELSE
       array_name = 'cqps: data%NZ_v'
       CALL SPACE_resize_array( MAX( prob%n, prob%n ), data%NZ_v,              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: data%NZ_aprod'
       CALL SPACE_resize_array( MAX( prob%n, prob%n ), data%NZ_aprod,          &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: data%V'
       CALL SPACE_resize_array( MAX( prob%n, prob%n ), data%V,                 &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'cqps: data%APROD'
       CALL SPACE_resize_array( MAX( prob%n, prob%n ), data%APROD,             &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

!  =====================  BQP problem construction  =======================
!
!  Data for the BQP is stored in data%bqp. Variables and their bounds are 
!  as follows:
!
!      w = ( x c_i y_e y_i y_l y_u z_l z_u )
!
!  where
!
!      x_l <=  x  <= x_u
!      c_l <= c_i <= c_u
!      y = (  y_e )
!          (  y_i )
!       0  <= y_l
!       0  <= y_u
!       0  <= z_l
!       0  <= z_u
!
!  and the index sets e and i refer to the general equality and inequality
!  constraints, respectively. The BQP is then to minimize 
!
!   phi(w) = 
!
!    - [ g^T x + 1/2 x^T H x 
!        - y_e^T ( A_e x - c_e ) - y_i^T (A_i x - c_i ) 
!        - ( c_i - c_l )^T y_l - ( c_u - c_i )^T y_u
!        - ( x - x_l )^T z_l - ( x_u - x )^T z_u ] 
!    + 1/2 eta || g + H x - A_e^T y_e - A_i^T y_i - ( z_l - z_u ) ||_P^2
!    + 1/2 eta || y_i - ( y_l - y_u ) ||^2
!    + 1/2 rho || A_i x - c_i ||^2
!
!  =========================================================================

!  count how many variables will be used in total, along with how many
!  of each category

     data%n_c_i = 0 ; data%n_y_l = 0 ; data%n_y_u = 0
     data%n_z_l = 0 ; data%n_z_u = 0

!  x variables

     DO i = 1, prob%m

!  c_i variables

       IF ( ( prob%C_l( i ) > - control%infinity .OR.                          &
              prob%C_u( i ) < control%infinity ) .AND.                         &
              prob%C_l( i ) /= prob%C_u( i ) ) THEN
         data%n_c_i = data%n_c_i + 1

!  y_l variables

         IF ( prob%C_l( i ) > - control%infinity ) data%n_y_l = data%n_y_l + 1

!  y_u variables

         IF ( prob%C_u( i ) < control%infinity ) data%n_y_u = data%n_y_u + 1
       END IF
     END DO

     DO i = 1, prob%n

!  z_l variables

       IF ( prob%X_l( i ) > - control%infinity ) data%n_z_l = data%n_z_l + 1

!  z_u variables

       IF ( prob%X_u( i ) < control%infinity ) data%n_z_u = data%n_z_u + 1
     END DO   

!  record the starting addresses of each variable sub-vector

     data%st_c_i = prob%n
     data%st_y = data%st_c_i + data%n_c_i
     data%st_y_l = data%st_y + prob%m
     data%st_y_u = data%st_y_l + data%n_y_l
     data%st_z_l = data%st_y_u + data%n_y_u
     data%st_z_u = data%st_z_l + data%n_z_l

!  count how many variables will be used in total

     data%bqp%n = data%st_z_u + data%n_z_u

     IF ( data%printi ) THEN
       IF ( prob%n > 1 ) THEN
         WRITE( control%out,                                                   &
           "( /, A, ' variables x: ', I0, '-', I0 )", advance = 'no' )         &
             prefix, 1, prob%n
       ELSE IF ( prob%n == 1 ) THEN
         WRITE( control%out,                                                   &
           "( /, A, ' variables x: ', I0 )", advance = 'no' )                  &
             prefix, 1
       END IF
       IF ( data%n_c_i > 1 ) THEN
         WRITE( control%out,                                                   &
         "( A, ', c_i: ', I0, '-', I0 )", advance = 'no' )                     &
           prefix, data%st_c_i + 1, data%st_y
       ELSE IF ( data%n_c_i == 1 ) THEN
          WRITE( control%out,                                                  &
         "( A, ', c_i: ', I0 )", advance = 'no' )                              &
           prefix, data%st_c_i + 1
       END IF
       IF ( prob%m > 1 ) THEN
         WRITE( control%out,                                                   &
         "( A, ', y: ', I0, '-', I0 )", advance = 'no' )                       &
           prefix, data%st_y + 1, data%st_y_l
       ELSE IF ( prob%m == 1 ) THEN
         WRITE( control%out,                                                   &
         "( A, ', y: ', I0 )", advance = 'no' )                                &
           prefix, data%st_y + 1
       END IF
       IF ( data%n_y_l > 1 ) THEN
         WRITE( control%out,                                                   &
         "( A, ', y_l: ', I0, '-', I0 )", advance = 'no' )                     &
           prefix, data%st_y_l + 1, data%st_y_u
       ELSE IF ( data%n_y_l == 1 ) THEN
         WRITE( control%out,                                                   &
         "( A, ', y_l: ', I0 )", advance = 'no' )                              &
           prefix, data%st_y_l + 1
       END IF
       IF ( data%n_y_u > 1 ) THEN
         WRITE( control%out,                                                   &
         "( A, ', y_u: ', I0, '-', I0 )", advance = 'no' )                     &
           prefix, data%st_y_u + 1, data%st_z_l
       ELSE IF ( data%n_y_u == 1 ) THEN
         WRITE( control%out,                                                   &
         "( A, ', y_u: ', I0 )", advance = 'no' )                              &
           prefix, data%st_y_u + 1
       END IF
       IF ( data%n_z_l > 1 ) THEN
         WRITE( control%out,                                                   &
         "( A, ', z_l: ', I0, '-', I0 )", advance = 'no' )                     &
           prefix, data%st_z_l + 1, data%st_z_u
       ELSE IF ( data%n_z_l == 1 ) THEN
         WRITE( control%out,                                                   &
         "( A, ', z_l: ', I0 )", advance = 'no' )                              &
           prefix, data%st_z_l + 1
       END IF
       IF ( data%n_z_u > 1 ) THEN
         WRITE( control%out,                                                   &
         "( A, ', z_u: ', I0, '-', I0 )" )                                     &
           prefix, data%st_z_u + 1, data%bqp%n
       ELSE IF ( data%n_z_u == 1 ) THEN
         WRITE( control%out,                                                   &
         "( A, ', z_u: ', I0 )" )                                              &
           prefix, data%st_z_u + 1
       ELSE
         WRITE( control%out, "( '')" )
       END IF
     END IF

!  allocate space for the BQP

     array_name = 'cqps: data%bqp%G'
     CALL SPACE_resize_array( data%bqp%n, data%bqp%G, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'cqps: data%bqp%X_l'
     CALL SPACE_resize_array( data%bqp%n, data%bqp%X_l, inform%status,         &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'cqps: data%bqp%X_u'
     CALL SPACE_resize_array( data%bqp%n, data%bqp%X_u, inform%status,         &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'cqps: data%bqp%X'
     CALL SPACE_resize_array( data%bqp%n, data%bqp%X, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'cqps: data%bqp%Z'
     CALL SPACE_resize_array( data%bqp%n, data%bqp%Z, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'cqps: data%B_stat'
     CALL SPACE_resize_array( data%bqp%n, data%B_stat, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'cqps: data%bqp_V'
     CALL SPACE_resize_array( data%bqp%n, data%bqp_V, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'cqps: data%PROD'
     CALL SPACE_resize_array( data%bqp%n, data%PROD, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  assign bounds and primal and dual starting values for the BQP

     c_i = data%st_c_i
     y = data%st_y
     y_l = data%st_y_l
     y_u = data%st_y_u
     z_l = data%st_z_l
     z_u = data%st_z_u

!  x variables

     data%bqp%X_l( : prob%n ) = prob%X_l( : prob%n )
     data%bqp%X_u( : prob%n ) = prob%X_u( : prob%n )
     data%bqp%X( : prob%n ) = prob%X( : prob%n )
     data%bqp%Z( : prob%n ) = prob%Z( : prob%n )

     DO i = 1, prob%m

!  y_e variables

       IF ( prob%C_l( i ) == prob%C_u( i ) ) THEN
         y = y + 1
         data%bqp%X_l( y ) = - ten * control%infinity
         data%bqp%X_u( y ) = ten * control%infinity
         data%bqp%X( y ) = zero
         data%bqp%Z( y ) = zero

!  y_i variables

       ELSE
         y = y + 1
         data%bqp%X_l( y ) = - ten * control%infinity
         data%bqp%X_u( y ) = ten * control%infinity
         data%bqp%X( y ) = zero
         data%bqp%Z( y ) = zero

!  c_i variables

         IF ( prob%C_l( i ) > - control%infinity .OR.                          &
              prob%C_u( i ) < control%infinity ) THEN 
           c_i = c_i + 1
           data%bqp%X_l( c_i ) = prob%C_l( i )
           data%bqp%X_u( c_i ) = prob%C_u( i )
           data%bqp%X( c_i ) = zero
           data%bqp%Z( c_i ) = zero

!  y_l variables

           IF ( prob%C_l( i ) > - control%infinity ) THEN
             y_l = y_l + 1
             data%bqp%X_l( y_l ) = zero
             data%bqp%X_u( y_l ) = ten * control%infinity
             data%bqp%X( y_l ) = MAX( prob%Y( i ), zero ) 
             data%bqp%Z( y_l ) = zero
           END IF

!  y_u variables

           IF ( prob%C_u( i ) < control%infinity ) THEN
             y_u = y_u + 1
             data%bqp%X_l( y_u ) = zero
             data%bqp%X_u( y_u ) = ten * control%infinity
             data%bqp%X( y_u ) = MAX( - prob%Y( i ), zero ) 
             data%bqp%Z( y_u ) = zero
           END IF
         END IF
       END IF
     END DO

     DO i = 1, prob%n

!  z_l variables

       IF ( prob%X_l( i ) > - control%infinity ) THEN
         z_l = z_l + 1
         data%bqp%X_l( z_l ) = zero
         data%bqp%X_u( z_l ) = ten * control%infinity
         data%bqp%X( z_l ) = MAX( prob%Z( i ), zero ) 
         data%bqp%Z( z_l ) = zero
       END IF

!  z_u variables

       IF ( prob%X_u( i ) < control%infinity ) THEN
         z_u = z_u + 1
         data%bqp%X_l( z_u ) = zero
         data%bqp%X_u( z_u ) = ten * control%infinity
         data%bqp%X( z_u ) = MAX( - prob%Z( i ), zero ) 
         data%bqp%Z( z_u ) = zero
       END IF
     END DO   

!  assign the initial regularisation parameters

     IF ( control%initial_rho > zero ) THEN
       data%rho = control%initial_rho
     ELSE
       data%rho = one
     END IF

     IF ( control%initial_eta > zero ) THEN
       data%eta = control%initial_eta
     ELSE
       data%eta = one
     END IF

!  set up the gradient for the BQP. Since 
!
!   grad_w phi(w) =
!
!     x: [  - u + eta H u + rho A_i^T ( A_i x - c_i )                         ]
!   c_i: [  - y_i + y_l - y_u + eta ( y_i - y_l + y_u ) - rho ( A_i x - c_i ) ]
!   y_e: [  A_e x - c_e - eta A_e u                                           ]
!   y_i  [  A_i x - c_i - eta A_i u + eta ( y_i - y_l + y_u )                 ]
!   y_l: [  c_i - c_l - eta ( y_i - y_l + y_u )                               ]
!   y_u: [  c_u - c_i + eta ( y_i - y_l + y_u )                               ]
!   z_l: [  x - x_l - eta u_l                                                 ]
!   z_u: [  x_u - x + eta u_u                                                 ]
!
!   where u = P( g + H x - A_e^T y_e - A_i^T y_i - z_l + z_u )
!
!   grad_w phi(0 ) =
!
!     x: [    - g + eta H P g    ]
!   c_i: [      0                ]
!   y_e: [  - c_e - eta A_e P g  ]
!   y_i  [        - eta A_i P g  ]
!   y_l: [  - c_l                ]
!   y_u: [    c_u                ]
!   z_l: [  - x_l - eta P g_l    ]
!   z_u: [    x_u + eta P g_u    ]

!  first compute P g

     data%PPROD( : prob%n ) = prob%G( : prob%n )
     IF ( data%explicit_h )                                                    &
       CALL  PSLS_solve( data%PPROD( : prob%n ), data%PSLS_data,               &
                         control%PSLS_control, inform%PSLS_inform )

!  now compute H P g ...

     IF ( data%reverse_h ) THEN
       reverse_h%V( : prob%n ) = data%PPROD( : prob%n )
       data%branch = 200 ; inform%status = 2 ; RETURN
     ELSE IF ( data%use_hprod ) THEN
       CALL eval_HPROD( i, userdata, data%PPROD( : prob%n ),                   &
                        data%HPROD( : prob%n ) )
     ELSE
       data%HPROD = zero
       DO i = 1, prob%n
         g_i = data%PPROD( i )
         DO k = data%H%ptr( i ), data%H%ptr( i + 1 ) - 1
           data%HPROD( data%H%col( k ) )                                       &
             = data%HPROD( data%H%col( k ) ) + data%H%val( k ) * g_i
         END DO
       END DO
     END IF
 200 CONTINUE

!  .. and A P g

     IF ( data%reverse_a ) THEN
       reverse_a%V( : prob%n ) = data%PPROD( : prob%n )
       data%branch = 300 ; inform%status = 3 ; RETURN
     ELSE IF ( data%use_aprod ) THEN
       CALL eval_APROD( i, userdata, .FALSE., data%PPROD( : prob%n ),          &
                        data%APROD( : prob%m ) )
     ELSE
       data%APROD( : prob%m ) = zero
       SELECT CASE ( SMT_get( prob%A%type ) )
       CASE ( 'COORDINATE' )
         DO l = 1, prob%A%ne
           i = prob%A%row( l ) ; j = prob%A%col( l )
           data%APROD( i ) =                                                   &
             data%APROD( i ) + prob%A%val( l ) * data%PPROD( j )
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, prob%m
           DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
             j = prob%A%col( l )
             data%APROD( i ) =                                                 &
               data%APROD( i ) + prob%A%val( l ) * data%PPROD( j )
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, prob%m
           DO j = 1, prob%n
             l = l + 1
             data%APROD( i ) =                                                 &
               data%APROD( i ) + prob%A%val( l ) * data%PPROD( j )
           END DO
         END DO
       END SELECT
     END IF
 300 CONTINUE

!  increase the regularisation parameter if necessary

     data%norm_g_2 = DOT_PRODUCT( data%PPROD, prob%G )
     IF ( data%norm_g_2 > zero ) THEN
       IF ( data% reverse_h ) THEN
         data%eta = MAX( data%eta, DOT_PRODUCT( data%PPROD( : prob%n ),        &
                                                data%PPROD( : prob%n ) ) /     &
                                   DOT_PRODUCT( reverse_h%PROD( : prob%n ),    &
                                                data%PPROD( : prob%n ) ) )
       ELSE
         data%eta = MAX( data%eta, DOT_PRODUCT( data%PPROD( : prob%n ),        &
                                                data%PPROD( : prob%n ) ) /     &
                                   DOT_PRODUCT( data%HPROD( : prob%n ),        &
                                                data%PPROD( : prob%n ) ) )
       END IF
     END IF

!  set up the constant term for the BQP

     data%bqp%f = - prob%f + half * data%eta * data%norm_g_2

!  now contruct the gradient

     c_i = data%st_c_i
     y = data%st_y
     y_l = data%st_y_l
     y_u = data%st_y_u
     z_l = data%st_z_l
     z_u = data%st_z_u

!  wrt to the x variables

     IF ( data% reverse_h ) THEN
       data%bqp%G( : prob%n ) =                                                &
         - prob%G( : prob%n ) + data%eta * reverse_h%PROD( : prob%n )
     ELSE
       data%bqp%G( : prob%n ) =                                                &
         - prob%G( : prob%n ) + data%eta * data%HPROD( : prob%n )
     END IF

     DO i = 1, prob%m
       IF ( data%reverse_a ) THEN
         prod = reverse_a%PROD( i )
       ELSE
         prod = data%APROD( i )
       END IF

!  wrt to the y_e variables

       IF ( prob%C_l( i ) == prob%C_u( i ) ) THEN
         y = y + 1
         data%bqp%G( y ) = - prob%C_l( i ) - data%eta * prod

!  wrt to the y_i variables

       ELSE
         y = y + 1
         data%bqp%G( y ) = - data%eta * prod

!  wrt to the c_i variables

         IF ( prob%C_l( i ) > - control%infinity .OR.                          &
              prob%C_u( i ) < control%infinity ) THEN 
           c_i = c_i + 1
           data%bqp%G( c_i ) = zero

!  wrt to the y_l variables

           IF ( prob%C_l( i ) > - control%infinity ) THEN
             y_l = y_l + 1
             data%bqp%G( y_l ) = - prob%C_l( i )
           END IF

!  wrt to the y_u variables

           IF ( prob%C_u( i ) < control%infinity ) THEN
             y_u = y_u + 1
             data%bqp%G( y_u ) = prob%C_u( i )
           END IF
         END IF
       END IF
     END DO

     DO i = 1, prob%n

!  wrt to the z_l variables

       IF ( prob%X_l( i ) > - control%infinity ) THEN
         z_l = z_l + 1
         data%bqp%G( z_l ) =  - prob%X_l( i ) - data%eta * data%PPROD( i )
       END IF

!  wrt to the z_u variables

       IF ( prob%X_u( i ) < control%infinity ) THEN
         z_u = z_u + 1
         data%bqp%G( z_u ) = prob%X_u( i ) + data%eta * data%PPROD( i )
       END IF
     END DO   

     IF ( data%printi ) THEN
       WRITE( control%out,                                                     &
         "( ' initial eta and rho are', ES11.4, ' and', ES11.4 )" )            &
           data%eta, data%rho
     END IF

!  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!                               solve the BQP
!  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

     data%control = control
     inform%BQP_inform%status = 1 ; inform%BQPB_inform%status = 1

 390 CONTINUE

!  use the projection-based BQP solver

       IF ( data%control%use_bqp ) THEN
         CALL BQP_solve( data%bqp, data%B_stat, data%BQP_data,                 &
                         data%control%BQP_control, inform%BQP_inform,          &
                         data%bqp_userdata, reverse = data%bqp_reverse )

!  successful exit with the solution

         IF ( inform%BQP_inform%status == GALAHAD_ok ) THEN
           GO TO 600

!  not convex - indicates infeasibility

         ELSE IF ( inform%BQP_inform%status == GALAHAD_error_inertia ) THEN
           GO TO 700

!  unbounded from below - indicates infeasibility

         ELSE IF ( inform%BQP_inform%status == GALAHAD_error_unbounded ) THEN
           GO TO 700

!  error exit due to CPU or iteration limits

         ELSE IF ( inform%BQP_inform%status == GALAHAD_error_max_iterations    &
                   .OR.                                                        &
                   inform%BQP_inform%status == GALAHAD_error_cpu_limit ) THEN
           inform%status = inform%BQP_inform%status
           GO TO 600 

!  error exit without the solution

         ELSE IF ( inform%BQP_inform%status < 0 ) THEN
           IF ( data%printi ) THEN
             WRITE( control%out, 2010 )                                        &
               prefix, inform%BQP_inform%status, 'BQP_solve'
           ELSE IF ( data%printe ) THEN
             WRITE( control%error, 2010 )                                      &
               prefix, inform%BQP_inform%status, 'BQP_solve'
           END IF
           inform%status = inform%BQP_inform%status
           GO TO 910
         END IF

!  use the interior-point-based BQP solver

       ELSE
         CALL BQPB_solve( data%bqp, data%B_stat, data%BQPB_data,               &
                         data%control%BQPB_control, inform%BQPB_inform,        &
                         data%BQP_userdata, reverse = data%BQPB_reverse )

!  successful exit with the solution

         IF ( inform%bqpb_inform%status == GALAHAD_ok ) THEN
           GO TO 600

!  not convex - indicates infeasibility

         ELSE IF ( inform%bqpb_inform%status == GALAHAD_error_inertia ) THEN
           GO TO 700

!  unbounded from below - indicates infeasibility

         ELSE IF ( inform%bqpb_inform%status == GALAHAD_error_unbounded ) THEN
           GO TO 700

!  error exit due to CPU or iteration limits

         ELSE IF ( inform%bqpb_inform%status == GALAHAD_error_max_iterations   &
                   .OR.                                                        &
                   inform%bqpb_inform%status == GALAHAD_error_cpu_limit ) THEN
           inform%status = inform%BQP_inform%status
           GO TO 600 

!  error exit without the solution

         ELSE IF ( inform%BQP_inform%status < 0 ) THEN
           IF ( data%printi ) THEN
             WRITE( control%out, 2010 )                                        &
               prefix, inform%bqpb_inform%status, 'BQPB_solve'
           ELSE IF ( data%printe ) THEN
             WRITE( control%error, 2010 )                                      &
               prefix, inform%bqpb_inform%status, 'BQPB_solve'
           END IF
           inform%status = inform%bqpb_inform%status
           GO TO 910
         END IF
       END IF

!  compute the Hessian-vector product
!
!   Hess_ww phi(w) =
!
!              x     c_i     y_e    y_i    y_l      y_u     z_l    z_u
!     x: [  - H       0     A_e^T  A_i^T    0        0       I     - I  ] 
!   c_i: [    0       0       0     - I     I       -I       0      0   ]
!   y_e: [   A_e      0       0      0      0        0       0      0   ]
!   y_i: [   A_i     - I      0      0      0        0       0      0   ]
!   y_l: [    0       I       0      0      0        0       0      0   ]
!   y_u: [    0     - I       0      0      0        0       0      0   ]
!   z_l: [    I       0       0      0      0        0       0      0   ]
!   z_u: [  - I       0       0      0      0        0       0      0   ]
!
!                             x   c_i    y_e     y_i     y_l   y_u   z_l   z_u
!     x: + eta [  H    ] P [  H    0  - A_e^T  - A_i^T    0     0    - I    I  ]
!   c_i:       [  0    ]  
!   y_e:       [ - A_e ]
!   y_i:       [ - A_i ]
!   y_l:       [   0   ]
!   y_u:       [   0   ]
!   z_l:       [ - I   ]   
!   z_u:       [   I   ]
!
!                             x   c_i    y_e     y_i     y_l   y_u   z_l   z_u
!     x: + eta [   0   ]   [  0    0      0       I      - I    I     0     0 ] 
!   c_i:       [   0   ]  
!   y_e:       [   0   ]
!   y_i:       [   I   ]
!   y_l:       [ - I   ]
!   y_u:       [   I   ]
!   z_l:       [   0   ]   
!   z_u:       [   0   ]
!
!                             x   c_i    y_e     y_i     y_l   y_u   z_l   z_u
!     x: + rho [ A_i^T ]   [  A   -I      0       0       0     0     0     0 ]
!   c_i:       [  - I  ]  
!   y_e:       [   0   ]
!   y_i:       [   0   ]
!   y_l:       [   0   ]
!   y_u:       [   0   ]
!   z_l:       [   0   ]
!   z_u:       [   0   ]
!
!  so if v = ( v_x  v_c_i  v_y_e  v_y_i   v_y_l  v_y_u  v_z_l  v_z_u ), 
!
!  Hess_ww phi(w) v = 
!
!     x: [    - u     + eta H P u           + rho A_i^T r_i ]  
!   c_i: [                                  - rho r_i       ]
!   y_e: [   A_e v_x  - eta A_e P u                         ]
!   y_i: [   A_i v_x  - eta A_i P u + eta s_i               ]
!   y_l: [                          - eta s_i               ]
!   y_u: [                            eta s_i               ]
!   z_l: [    v_x     - eta  P u                            ]
!   z_u: [  - v_x     + eta  P u                            ]            
!
!  where
!
!   u = ( H v_x - A_e^T v_y_e - A_i^T v_y_i - v_z_l + v_z_u )
!   r_i = A_i v_x - v_c_i 
!   s_i = v_y_i - v_y_l + v_y_u

!  the products should be over sparse v  ** replace later **

       IF ( data%control%use_bqp ) THEN
         SELECT CASE ( inform%BQP_inform%status )
         CASE ( 2 ) 
           data%BQP_V( : data%bqp%n ) = data%BQP_reverse%V( : data%bqp%n )
         CASE ( 3, 4 ) 
           data%BQP_V( : data%bqp%n ) = zero
           DO l = data%BQP_reverse%nz_v_start, data%BQP_reverse%nz_v_end
             data%BQP_V( data%BQP_reverse%NZ_v( l ) ) =                        &
               data%BQP_reverse%V( data%BQP_reverse%NZ_v( l ) )
           END DO
         END SELECT
       ELSE
         data%BQP_V( : data%bqp%n ) = data%BQPB_reverse%V( : data%bqp%n )
       END IF

!  record v_x

       IF ( data%reverse_h ) THEN
         reverse_h%V( : prob%n ) = data%BQP_V( : prob%n )
         data%branch = 410 ; inform%status = 2 ; RETURN
       END IF

!  compute H v_x

       IF ( data%use_hprod ) THEN
         CALL eval_HPROD( i, userdata, data%BQP_V( : prob%n ),                 &
                          data%U( : prob%n ) )
       ELSE
         data%U( : prob%n ) = zero
         DO i = 1, prob%n
           v_i = data%BQP_V( i )
           DO k = data%H%ptr( i ), data%H%ptr( i + 1 ) - 1
             data%U( data%H%col( k ) )                                         &
               = data%U( data%H%col( k ) ) + data%H%val( k ) * v_i
           END DO
         END DO
       END IF

!  initialize u = H v_x

 410   CONTINUE
       IF ( data%reverse_h ) data%U( : prob%n ) = reverse_h%PROD( : prob%n )

!  compute A v_x

       IF ( data%reverse_a ) THEN
         reverse_a%V( : prob%n ) = data%BQP_V( : prob%n )
         data%branch = 420 ; inform%status = 3 ; RETURN
       ELSE IF ( data%use_aprod ) THEN
         CALL eval_APROD( i, userdata, .FALSE., data%BQP_V( : prob%n ),        &
                   data%PROD( data%st_y + 1 : data%st_y + prob%m))
       ELSE
         data%PROD( data%st_y + 1 : data%st_y + prob%m ) = zero
         SELECT CASE ( SMT_get( prob%A%type ) )
         CASE ( 'COORDINATE' )
           DO l = 1, prob%A%ne
             i = prob%A%row( l ) ; j = prob%A%col( l )
             data%PROD( data%st_y + i ) = data%PROD( data%st_y + i ) +         &
                 prob%A%val( l ) * data%BQP_V( j )
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           DO i = 1, prob%m
             DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
               j = prob%A%col( l )
               data%PROD( data%st_y + i ) = data%PROD( data%st_y + i ) +       &
                   prob%A%val( l ) * data%BQP_V( j )
             END DO
           END DO
         CASE ( 'DENSE' )
           l = 0
           DO i = 1, prob%m
             DO j = 1, prob%n
               l = l + 1
               data%PROD( data%st_y + i ) = data%PROD( data%st_y + i ) +       &
                   prob%A%val( l ) * data%BQP_V( j )
             END DO
           END DO
         END SELECT
       END IF
 420   CONTINUE

!  initialize the y-th components of Hess_ww phi(w) v

       IF ( data%reverse_a ) data%PROD( data%st_y + 1 : data%st_y + prob%m )   &
           = reverse_a%PROD( : prob%m )

!  set the c_i-th components of Hess_ww phi(w) v and record rho * r

       c_i = data%st_c_i
       DO i = 1, prob%m
         IF ( ( prob%C_l( i ) > - control%infinity .OR.                        &
                prob%C_u( i ) < control%infinity ) .AND.                       &
                prob%C_l( i ) /= prob%C_u( i ) ) THEN
           c_i = c_i + 1
           rho_i = data%rho * ( data%PROD( data%st_y + i ) - data%BQP_V( c_i ) )
           data%PROD( c_i ) = - rho_i
           IF ( data%reverse_a ) THEN
             reverse_a%V( i ) = rho_i
           ELSE
             data%V( i ) = rho_i
           END IF
         ELSE
           IF ( data%reverse_a ) THEN
             reverse_a%V( i ) = zero
           ELSE
             data%V( i ) = zero
           END IF
         END IF
       END DO

!  compute rho A_i^T r

       IF ( data%reverse_a ) THEN
         data%branch = 430 ; inform%status = 4 ; RETURN
       ELSE IF ( data%use_aprod ) THEN
         CALL eval_APROD( i, userdata, .TRUE., data%V( : prob%m ),           &
                          data%PROD( : prob%n ) )
       ELSE
         data%PROD( : prob%n ) = zero
         SELECT CASE ( SMT_get( prob%A%type ) )
         CASE ( 'COORDINATE' )
           DO l = 1, prob%A%ne
             i = prob%A%row( l ) ; j = prob%A%col( l )
             data%PROD( j ) = data%PROD( j ) + prob%A%val( l ) * data%V( i )
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           DO i = 1, prob%m
             DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
               j = prob%A%col( l )
               data%PROD( j ) = data%PROD( j ) + prob%A%val( l ) * data%V( i )
             END DO
           END DO
         CASE ( 'DENSE' )
           l = 0
           DO i = 1, prob%m
             DO j = 1, prob%n
               l = l + 1
               data%PROD( j ) = data%PROD( j ) + prob%A%val( l ) * data%V( i )
             END DO
           END DO
         END SELECT
       END IF

!  initialize the x-th components of Hess_ww phi(w) v

 430   CONTINUE
       IF ( data%reverse_a ) data%PROD( : prob%n ) = reverse_a%PROD( : prob%n )

!  record (  v_y_e ) 
!         (  v_y_i )

!      y = data%st_y ; y_l = data%st_y_l ; y_u = data%st_y_u
       IF ( data%reverse_a ) THEN
         reverse_a%V( : prob%m ) =                                             &
           data%BQP_V( data%st_y + 1 : data%st_y + prob%m )
         data%branch = 440 ; inform%status = 4 ; RETURN
       END IF

!  compute A^T ( v_y_e ) 
!              ( v_y_i )

       IF ( data%use_aprod ) THEN
         CALL eval_APROD( i, userdata, .TRUE.,                                 &
                          data%BQP_V( data%st_y + 1 : data%st_y + prob%m ),    &
                          data%APROD( : prob%n ) )
       ELSE
         data%APROD( : prob%n ) = zero
         SELECT CASE ( SMT_get( prob%A%type ) )
         CASE ( 'COORDINATE' )
           DO l = 1, prob%A%ne
             i = prob%A%row( l ) ; j = prob%A%col( l )
             data%APROD( j ) =                                                 &
               data%APROD( j ) + prob%A%val( l ) * data%BQP_V( data%st_y + i )
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           DO i = 1, prob%m
             DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
               j = prob%A%col( l )
               data%APROD( j ) =                                               &
                 data%APROD( j ) + prob%A%val( l ) * data%BQP_V( data%st_y + i )
             END DO
           END DO
         CASE ( 'DENSE' )
           l = 0
           DO i = 1, prob%m
             DO j = 1, prob%n
               l = l + 1
               data%APROD( j ) =                                               &
                 data%APROD( j ) + prob%A%val( l ) * data%BQP_V( data%st_y + i )
             END DO
           END DO
         END SELECT
       END IF

!  update u <- u - A_e^T v_y_e - A_i^T v_y_i ...

 440   CONTINUE
       IF ( data%reverse_a ) THEN
         data%U( : prob%n ) = data%U( : prob%n ) - reverse_a%PROD( : prob%n )
       ELSE
         data%U( : prob%n ) = data%U( : prob%n ) - data%APROD( : prob%n )
       END IF

!  ... and u <- u - v_z_l + v_z_u

       z_l = data%st_z_l ; z_u = data%st_z_u
       DO i = 1, prob%n
         IF ( prob%X_l( i ) > - control%infinity .AND.                         &
              prob%X_u( i ) < control%infinity ) THEN
           z_l = z_l + 1 ; z_u = z_u + 1
           data%U( i ) = data%U( i ) - data%BQP_V( z_l ) + data%BQP_V( z_u )
         ELSE IF ( prob%X_l( i ) > - control%infinity ) THEN
           z_l = z_l + 1
           data%U( i ) = data%U( i ) - data%BQP_V( z_l )
         ELSE IF ( prob%X_u( i ) < control%infinity ) THEN
           z_u = z_u + 1
           data%U( i ) = data%U( i ) + data%BQP_V( z_u )
         END IF
       END DO  

!  continue the x-th components of Hess_ww phi(w) v

       data%PROD( : prob%n ) = data%PROD( : prob%n ) - data%U( : prob%n )

!  reset u <- P u

!write(6,"( 'u', /, ( 3ES24.16 ) )" ) data%U( : prob%n )
       IF ( data%explicit_h )                                                  &
         CALL PSLS_solve( data%U( : prob%n ), data%PSLS_data,                  &
                          control%PSLS_control, inform%PSLS_inform )

!  set the z_l and z_u-th components of Hess_ww phi(w) v

       z_l = data%st_z_l ; z_u = data%st_z_u
       DO i = 1, prob%n
         IF ( prob%X_l( i ) > - control%infinity .AND.                         &
              prob%X_u( i ) < control%infinity ) THEN
           z_l = z_l + 1 ; z_u = z_u + 1
           data%PROD( z_l ) = data%BQP_V( i ) - data%eta * data%U( i )
           data%PROD( z_u ) = - data%BQP_V( i ) + data%eta * data%U( i )
         ELSE IF ( prob%X_l( i ) > - control%infinity ) THEN
           z_l = z_l + 1
           data%PROD( z_l ) = data%BQP_V( i ) - data%eta * data%U( i )
         ELSE IF ( prob%X_u( i ) < control%infinity ) THEN
           z_u = z_u + 1
           data%PROD( z_u ) = - data%BQP_V( i ) + data%eta * data%U( i )
         END IF
       END DO  

!  record P u 

       IF ( data%reverse_h ) THEN
         reverse_h%V( : prob%n ) = data%U( : prob%n )
         data%branch = 450 ; inform%status = 2 ; RETURN

!  compute H P u

       ELSE IF ( data%use_hprod ) THEN
         CALL eval_HPROD( i, userdata, data%U( : prob%n ),                     &
                          data%HPROD( : prob%n ) )
       ELSE
         data%HPROD( : prob%n ) = zero
         DO i = 1, prob%n
           v_i = data%U( i )
           DO k = data%H%ptr( i ), data%H%ptr( i + 1 ) - 1
             data%HPROD( data%H%col( k ) )                                     &
               = data%HPROD( data%H%col( k ) ) + data%H%val( k ) * v_i
           END DO
         END DO
       END IF
!write(6,"( 'HPu', /, ( 3ES24.16 ) )" ) data%HPROD( : prob%n )

!  finish the x-th components of Hess_ww phi(w) v

 450   CONTINUE
       IF ( data%reverse_h ) THEN
         data%PROD( : prob%n ) = data%PROD( : prob%n )                         &
             + data%eta * reverse_h%PROD( : prob%n )
       ELSE
         data%PROD( : prob%n ) = data%PROD( : prob%n )                         &
             + data%eta * data%HPROD( : prob%n )
       END IF

!  compute A P u

       IF ( data%reverse_a ) THEN
         reverse_a%V( : prob%n ) = data%U( : prob%n )
         data%branch = 460 ; inform%status = 3 ; RETURN
       ELSE IF ( data%use_aprod ) THEN
         CALL eval_APROD( i, userdata, .FALSE., data%U( : prob%n ),            &
                          data%APROD( : prob%m ) )
       ELSE
         data%APROD( : prob%m ) = zero
         SELECT CASE ( SMT_get( prob%A%type ) )
         CASE ( 'COORDINATE' )
           DO l = 1, prob%A%ne
             i = prob%A%row( l ) ; j = prob%A%col( l )
             data%APROD( i ) = data%APROD( i ) + prob%A%val( l ) * data%U( j )
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           DO i = 1, prob%m
             DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
               j = prob%A%col( l )
               data%APROD( i ) = data%APROD( i ) + prob%A%val( l ) * data%U( j )
             END DO
           END DO
         CASE ( 'DENSE' )
           l = 0
           DO i = 1, prob%m
             DO j = 1, prob%n
               l = l + 1
               data%APROD( i ) = data%APROD( i ) + prob%A%val( l ) * data%U( j )
             END DO
           END DO
         END SELECT
       END IF

!  continue the y-th components of Hess_ww phi(w) v

 460   CONTINUE
       IF ( data%reverse_a ) THEN
         data%PROD( data%st_y + 1 : data%st_y + prob%m ) =                     &
           data%PROD( data%st_y + 1 : data%st_y + prob%m ) -                   &
             data%eta * reverse_a%PROD( : prob%m )
       ELSE
         data%PROD( data%st_y + 1 : data%st_y + prob%m ) =                     &
           data%PROD( data%st_y + 1 : data%st_y + prob%m ) -                   &
             data%eta * data%APROD( : prob%m )
       END IF

!  compute s_i

       y = data%st_y ; y_l = data%st_y_l ; y_u = data%st_y_u
       DO i = 1, prob%m
         IF ( prob%C_l( i ) == prob%C_u( i ) ) THEN
           y = y + 1
         ELSE
           y = y + 1
           s_i = data%BQP_V( y )
           IF ( prob%C_l( i ) > - control%infinity .OR.                        &
                prob%C_u( i ) < control%infinity ) THEN 
             IF ( prob%C_l( i ) > - control%infinity ) THEN
               y_l = y_l + 1
               s_i = s_i - data%BQP_V( y_l )
             END IF
             IF ( prob%C_u( i ) < control%infinity ) THEN
               y_u = y_u + 1
               s_i = s_i + data%BQP_V( y_u )
             END IF
           END IF
           s_i = data%eta * s_i

!  finish the y_i-th components of Hess_ww phi(w) v ...

           data%PROD( y ) = data%PROD( y ) + s_i

!  ... and set the y_l and y_u-th components

           IF ( prob%C_l( i ) > - control%infinity ) data%PROD( y_l ) = - s_i
           IF ( prob%C_u( i ) < control%infinity ) data%PROD( y_u ) = s_i
         END IF
       END DO

!  this should be replaced by the nonzeros for efficiency ** replace later **

       IF ( data%control%use_bqp ) THEN
         data%BQP_reverse%PROD( : data%bqp%n ) = data%PROD( : data%bqp%n )
         IF ( inform%BQP_inform%status == 4 ) THEN
           data%BQP_reverse%nz_prod_end = 0
           DO i = 1, data%bqp%n
             IF ( data%BQP_reverse%PROD( i ) /= zero ) THEN
               data%BQP_reverse%nz_prod_end = data%BQP_reverse%nz_prod_end + 1
               data%BQP_reverse%nz_prod( data%BQP_reverse%nz_prod_end ) = i
!              data%BQP_reverse%PROD( data%BQP_reverse%nz_prod_end ) =         &
!                data%BQP_reverse%PROD( i )
             END IF
           END DO
         END IF
       ELSE
         data%BQPB_reverse%PROD( : data%bqp%n ) = data%PROD( : data%bqp%n )
       END IF
       GO TO 390

!  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!                                  BQP solved
!  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

 600 CONTINUE

!  record the primal variables and lagrange multipliers

     prob%X( : prob%n ) = data%bqp%X( : prob%n )
     prob%Y( : prob%m ) = data%bqp%X( data%st_y + 1 : data%st_y + prob%m )

!  record the dual variables

     z_l = data%st_z_l ; z_u = data%st_z_u
     DO i = 1, prob%n
       prob%Z( i ) = zero
       IF ( prob%X_l( i ) > - control%infinity ) THEN
         z_l = z_l + 1
         prob%Z( i ) = prob%Z( i ) + data%bqp%X( z_l )
       END IF
       IF ( prob%X_u( i ) < control%infinity ) THEN
         z_u = z_u + 1
         prob%Z( i ) = prob%Z( i ) - data%bqp%X( z_u )
       END IF
     END DO   

!  compute H x

     IF ( data%reverse_h ) THEN
       reverse_h%V( : prob%n ) = data%bqp%X( : prob%n )
       data%branch = 610 ; inform%status = 2 ; RETURN
     ELSE IF ( data%use_hprod ) THEN
       CALL eval_HPROD( i, userdata, prob%X( : prob%n ),                       &
                        data%HPROD( : prob%n ) )
     ELSE
       data%HPROD( : prob%n ) = zero
       DO i = 1, prob%n
         v_i = prob%X( i )
         DO k = data%H%ptr( i ), data%H%ptr( i + 1 ) - 1
           data%HPROD( data%H%col( k ) )                                       &
             = data%HPROD( data%H%col( k ) ) + data%H%val( k ) * v_i
         END DO
       END DO
     END IF

!  compute the objective value

 610 CONTINUE
     IF ( data%reverse_h ) THEN
       inform%obj = half * DOT_PRODUCT( reverse_h%PROD( : prob%n ),            &
                                        prob%X( : prob%n ) ) +                 &
                           DOT_PRODUCT( prob%G( : prob%n ),                    &
                                        prob%X( : prob%n ) ) + prob%f
     ELSE
       inform%obj = half * DOT_PRODUCT( data%HPROD( : prob%n ),                &
                                        prob%X( : prob%n ) ) +                 &
                           DOT_PRODUCT( prob%G( : prob%n ),                    &
                                        prob%X( : prob%n ) ) + prob%f
     END IF

!  compute A x

     IF ( data%reverse_a ) THEN
       reverse_a%V( : prob%n ) = prob%X( : prob%n )
       data%branch = 620 ; inform%status = 3 ; RETURN
     ELSE IF ( data%use_aprod ) THEN
       CALL eval_APROD( i, userdata, .FALSE., prob%X( : prob%n ),              &
                        prob%C( : prob%m ) )
     ELSE
       prob%C( : prob%m ) = zero
       SELECT CASE ( SMT_get( prob%A%type ) )
       CASE ( 'COORDINATE' )
         DO l = 1, prob%A%ne
           i = prob%A%row( l ) ; j = prob%A%col( l )
           prob%C( i ) = prob%C( i ) + prob%A%val( l ) * prob%X( j )
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, prob%m
           DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
             j = prob%A%col( l )
             prob%C( i ) = prob%C( i ) + prob%A%val( l ) * prob%X( j )
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, prob%m
           DO j = 1, prob%n
             l = l + 1
             prob%C( i ) = prob%C( i ) + prob%A%val( l ) * prob%X( j )
           END DO
         END DO
       END SELECT
     END IF

!  record the constraint value

 620 CONTINUE
     IF ( data%reverse_h ) prob%C( : prob%m ) = reverse_a%PROD( : prob%m )

!  evaluate the norms of the primal infeasibility and complementary slackness

     inform%primal_infeasibility = zero
     inform%complementary_slackness = zero

!  for the bound constraints ...

     DO i = 1, prob%n
       IF ( prob%X_l( i ) > - control%infinity ) THEN
           inform%primal_infeasibility = MAX( inform%primal_infeasibility,     &
             prob%X_l( i ) -  prob%X( i ) )
           inform%complementary_slackness =                                    &
             MAX( inform%complementary_slackness,                              &
                  ( prob%X( i ) - prob%X_l( i ) ) * MAX( prob%Z( i ), zero ) )
       END IF
       IF ( prob%X_u( i ) < control%infinity ) THEN
           inform%primal_infeasibility = MAX( inform%primal_infeasibility,     &
             prob%X( i ) -  prob%X_u( i ) )
           inform%complementary_slackness =                                    &
             MAX( inform%complementary_slackness,                              &
                  ( prob%X_u( i ) - prob%X( i ) ) * MAX( - prob%Z( i ), zero ) )
       END IF
     END DO   

!  ... and the linear constraints

     DO i = 1, prob%m
       IF ( prob%C_l( i ) == prob%C_u( i ) ) THEN
         inform%primal_infeasibility = MAX( inform%primal_infeasibility,       &
           ABS(  prob%C( i ) -  prob%C_l( i ) ) )
       ELSE
         IF ( prob%C_l( i ) > - control%infinity ) THEN
           inform%primal_infeasibility = MAX( inform%primal_infeasibility,     &
             prob%C_l( i ) -  prob%C( i ) )
           inform%complementary_slackness =                                    &
             MAX( inform%complementary_slackness,                              &
                  ( prob%C( i ) - prob%C_l( i ) ) * MAX( prob%Y( i ), zero ) )
         END IF
         IF ( prob%C_u( i ) < control%infinity ) THEN
           inform%primal_infeasibility = MAX( inform%primal_infeasibility,     &
             prob%C( i ) -  prob%C_u( i ) )
           inform%complementary_slackness =                                    &
             MAX( inform%complementary_slackness,                              &
                  ( prob%C_u( i ) - prob%C( i ) ) * MAX( - prob%Y( i ), zero ) )
         END IF
       END IF
     END DO

!  compute A^T y

     IF ( data%reverse_a ) THEN
       reverse_a%V( : prob%m ) = prob%Y( : prob%m )
       data%branch = 630 ; inform%status = 4 ; RETURN
     ELSE IF ( data%use_aprod ) THEN
       CALL eval_APROD( i, userdata, .TRUE., prob%Y( : prob%m ),               &
                        data%APROD( : prob%n ) )
     ELSE
       data%APROD( : prob%n ) = zero
       SELECT CASE ( SMT_get( prob%A%type ) )
       CASE ( 'COORDINATE' )
         DO l = 1, prob%A%ne
           i = prob%A%row( l ) ; j = prob%A%col( l )
           data%APROD( j ) = data%APROD( j ) + prob%A%val( l ) * prob%Y( + i )
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, prob%m
           DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
             j = prob%A%col( l )
             data%APROD( j ) = data%APROD( j ) + prob%A%val( l ) * prob%Y( + i )
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, prob%m
           DO j = 1, prob%n
             l = l + 1
             data%APROD( j ) = data%APROD( j ) + prob%A%val( l ) * prob%Y( + i )
           END DO
         END DO
       END SELECT
     END IF

!  evaluate the norm of dual infeasibility

 630 CONTINUE
     IF ( data%reverse_h ) THEN
       IF ( data%reverse_a ) THEN
         inform%dual_infeasibility = MAXVAL( ABS(                              &
           prob%G( : prob%n ) + reverse_h%PROD( : prob%n ) -                   &
           reverse_a%PROD( : prob%n ) - prob%Z( : prob%n ) ) )
       ELSE
         inform%dual_infeasibility = MAXVAL( ABS(                              &
           prob%G( : prob%n ) + reverse_h%PROD( : prob%n ) -                   &
           data%APROD( : prob%n ) - prob%Z( : prob%n ) ) )
       END IF
     ELSE
       IF ( data%reverse_a ) THEN
         inform%dual_infeasibility = MAXVAL( ABS(                              &
           prob%G( : prob%n ) + data%HPROD( : prob%n ) -                       &
           reverse_a%PROD( : prob%n ) - prob%Z( : prob%n ) ) )
       ELSE
         inform%dual_infeasibility = MAXVAL( ABS(                              &
           prob%G( : prob%n ) + data%HPROD( : prob%n ) -                       &
           data%APROD( : prob%n ) - prob%Z( : prob%n ) ) )
       END IF
     END IF

     IF ( inform%status == GALAHAD_error_max_iterations .OR.                   &
          inform%status == GALAHAD_error_cpu_limit ) GO TO 910

!  check for convergence

     IF ( inform%primal_infeasibility <= control%stop_p .AND.                  &
          inform%dual_infeasibility <= control%stop_d .AND.                    &
          inform%complementary_slackness <= control%stop_c ) GO TO 900

!  not yet optimal: increase eta and rho

 700 CONTINUE
     eta_new = ten * data%eta
!    eta_new = data%eta
     IF ( data%eta > zero ) THEN
       ratio_eta = eta_new / data%eta
     ELSE
       ratio_eta = zero
     END IF
     one_minus_ratio_eta = one - ratio_eta

     rho_new = ten * data%rho
!    rho_new = data%rho
     IF ( data%rho > zero ) THEN
       ratio_rho = rho_new / data%rho
     ELSE
       ratio_rho = zero
     END IF

!  set up the constant term for the new BQP to reflect the new eta and rho

     data%bqp%f = - prob%f + half * eta_new * data%norm_g_2

!  now contruct the new gradient

     y = data%st_y ; z_l = data%st_z_l ; z_u = data%st_z_u

!  wrt to the x variables

     data%bqp%G( : prob%n ) = ratio_eta * data%bqp%G( : prob%n ) -             &
         one_minus_ratio_eta * prob%G( : prob%n )

!  wrt to the y_e variables

     DO i = 1, prob%m
       IF ( prob%C_l( i ) == prob%C_u( i ) ) THEN
         y = y + 1
         data%bqp%G( y ) =  ratio_eta * data%bqp%G( y ) -                      &
           one_minus_ratio_eta * prob%C_l( i ) 

!  wrt to the y_i variables

       ELSE
         y = y + 1
         data%bqp%G( y ) = ratio_eta * data%bqp%G( y )
       END IF
     END DO

!  wrt to the z_l variables

     DO i = 1, prob%n
       IF ( prob%X_l( i ) > - control%infinity ) THEN
         z_l = z_l + 1
         data%bqp%G( z_l ) = ratio_eta * data%bqp%G( z_l ) -                   &
           one_minus_ratio_eta * prob%X_l( i )
       END IF

!  wrt to the z_u variables

       IF ( prob%X_u( i ) < control%infinity ) THEN
         z_u = z_u + 1
         data%bqp%G( z_u ) = ratio_eta * data%bqp%G( z_u ) +                   &
           one_minus_ratio_eta * prob%X_u( i )
       END IF
     END DO   

!  re-solve the BQP

     data%eta = eta_new
     data%rho = rho_new
     inform%BQP_inform%status = 10
     inform%BQPB_inform%status = 10
!     IF ( .NOT. data%control%use_bqp ) inform%BQP_inform%status = 1
!     data%control%use_bqp = .TRUE.
data%control%BQPB_control%stop_d = 0.1_wp * data%control%BQPB_control%stop_d
if (data%rho > ten ** 20 ) then
write(6,*) ' eta too large. stopping'
stop
end if
     IF ( data%printi ) THEN
       WRITE( control%out,                                                     &
         "( /, A, ' primal infeasibility = ', ES11.4, /,                       &
        &      A, ' dual   infeasibility = ', ES11.4, /,                       &
        &      A, ' complementarity      = ', ES11.4 )" )                      &
         prefix, inform%primal_infeasibility,                                  &
         prefix, inform%dual_infeasibility,                                    &
         prefix, inform%complementary_slackness
       WRITE( control%out,                                                     &
          "( ' increasing eta and rho to', ES11.4, ' and', ES11.4 )" )         &
          data%eta, data%rho
     END IF
     GO TO 390

!  Successful return

 900 CONTINUE
     inform%status = 0
     CALL CPU_TIME( time ) ; inform%time%total = time - data%time_start 
     IF ( data%printi ) THEN
       WRITE( control%out,                                                     &
         "( /, A, ' primal infeasibility = ', ES11.4, /,                       &
        &      A, ' dual   infeasibility = ', ES11.4, /,                       &
        &      A, ' complementarity      = ', ES11.4 )" )                      &
         prefix, inform%primal_infeasibility, prefix,                          &
         inform%dual_infeasibility, prefix, inform%complementary_slackness
       IF ( control%use_bqp ) THEN
         WRITE( control%out, 2040 ) prefix
       ELSE
         WRITE( control%out, 2050 ) prefix
       END IF
       WRITE( control%out, 2060 ) prefix,                                      &
         TRIM( PSLS_name( control%PSLS_control%preconditioner,                 &
                          control%PSLS_control%semi_bandwidth,                 &
                          control%PSLS_control%icfs_vectors ) )
     END IF
     IF ( data%printd ) WRITE( control%out, 2000 ) prefix, ' leaving '
     RETURN

!  Error returns

 910 CONTINUE 
     CALL CPU_TIME( time ) ; inform%time%total = time - data%time_start 
     IF ( data%printi ) THEN
       SELECT CASE ( inform%status )
       CASE ( GALAHAD_ok )
       CASE ( GALAHAD_error_allocate )
         WRITE( control%out, 2020 ) prefix, 'allocation error'
         WRITE( control%out, 2030 ) prefix, inform%alloc_status,               &
                                            inform%bad_alloc
       CASE ( GALAHAD_error_deallocate )
         WRITE( control%out, 2020 ) prefix, 'de-allocation error'
         WRITE( control%out, 2030 ) prefix, inform%alloc_status,               &
                                            inform%bad_alloc
       CASE ( GALAHAD_error_restrictions )
         WRITE( control%out, 2020 ) prefix, 'input restriction violated'
       CASE ( GALAHAD_error_dual_infeasible )
         WRITE( control%out, 2020 ) prefix, 'no feasible point'
       CASE ( GALAHAD_error_unbounded )
         WRITE( control%out, 2020 ) prefix, 'problem unbounded'
       CASE ( GALAHAD_error_max_iterations )
         WRITE( control%out, 2020 ) prefix, 'iteration limit exceeded'
       CASE ( GALAHAD_error_cpu_limit )
         WRITE( control%out, 2020 ) prefix, 'CPU time limit exceeded'
       CASE ( GALAHAD_error_inertia )
         WRITE( control%out, 2020 ) prefix, 'problem is not strictly convex'
       CASE DEFAULT
         WRITE( control%out, 2020 ) prefix, 'undefined error'
       END SELECT
       IF ( control%use_bqp ) THEN
         WRITE( control%out, 2040 ) prefix
       ELSE
         WRITE( control%out, 2050 ) prefix
       END IF
       WRITE( control%out, 2060 ) prefix,                                      &
         TRIM( PSLS_name( control%PSLS_control%preconditioner,                 &
                          control%PSLS_control%semi_bandwidth,                 &
                          control%PSLS_control%icfs_vectors ) )
     ELSE IF ( data%printe ) THEN
       WRITE( control%error, 2010 ) prefix, inform%status, 'CQPS_solve'
     END IF
     IF ( data%printd ) WRITE( control%out, 2000 ) prefix, ' leaving '
     RETURN  

!  Non-executable statements

2000 FORMAT( /, A, ' --', A, ' CQPS_solve' ) 
2010 FORMAT( A, '   **  Error return ', I0, ' from ', A ) 
2020 FORMAT( /, A, ' CQPS error exit: ', A )
2030 FORMAT( /, A, ' allocation error status ', I0, ' for ', A )
2040 FORMAT( /, A, ' projection-based BQP solver used' )
2050 FORMAT( /, A, ' interior-point BQP solver used' )
2060 FORMAT( A, 1X, A, ' used' )

!  End of CQPS_solve

     END SUBROUTINE CQPS_solve

!-*-*-*-*-*-*-   C Q P S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE CQPS_terminate( data, control, inform, reverse_h, reverse_a )

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
!   data      see Subroutine CQPS_initialize
!   control   see Subroutine CQPS_initialize
!   inform    see Subroutine CQPS_solve
!   reverse_h see Subroutine BQP_solve
!   reverse_a see Subroutine BQP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( CQPS_data_type ), INTENT( INOUT ) :: data
     TYPE ( CQPS_control_type ), INTENT( IN ) :: control        
     TYPE ( CQPS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( CQPS_reverse_h_type ), OPTIONAL, INTENT( INOUT ) :: reverse_h
     TYPE ( CQPS_reverse_a_type ), OPTIONAL, INTENT( INOUT ) :: reverse_a

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by BQP

     CALL BQP_terminate( data%BQP_data, control%BQP_control,                   &
                         inform%BQP_inform, reverse = data%BQP_reverse )
     IF ( inform%BQP_inform%status /= GALAHAD_ok ) THEN
       inform%status = GALAHAD_error_deallocate
       inform%alloc_status = inform%BQP_inform%alloc_status
!      inform%bad_alloc = inform%BQP_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

     CALL BQPB_terminate( data%bqpb_data, control%BQPB_control,                &
          inform%BQPB_inform, reverse = data%bqpb_reverse )
     IF ( inform%BQP_inform%status /= GALAHAD_ok ) THEN
       inform%status = GALAHAD_error_deallocate
       inform%alloc_status = inform%BQP_inform%alloc_status
!      inform%bad_alloc = inform%BQP_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all remaining allocated arrays

     array_name = 'bqp: data%H%ptr'
     CALL SPACE_dealloc_array( data%H%ptr,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%H%col'
     CALL SPACE_dealloc_array( data%H%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%H%val'
     CALL SPACE_dealloc_array( data%H%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     IF ( PRESENT( reverse_h ) ) THEN
       array_name = 'bqp: reverse_h%V'
       CALL SPACE_dealloc_array( reverse_h%V,                                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN

       array_name = 'bqp: reverse_h%PROD'
       CALL SPACE_dealloc_array( reverse_h%PROD,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN

       array_name = 'bqp: reverse_h%NZ_v'
       CALL SPACE_dealloc_array( reverse_h%NZ_v,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN

       array_name = 'bqp: reverse_h%NZ_prod'
       CALL SPACE_dealloc_array( reverse_h%NZ_prod,                            &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN
     END IF

     IF ( PRESENT( reverse_a ) ) THEN
       array_name = 'bqp: reverse_a%V'
       CALL SPACE_dealloc_array( reverse_a%V,                                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN

       array_name = 'bqp: reverse_a%PROD'
       CALL SPACE_dealloc_array( reverse_a%PROD,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN

       array_name = 'bqp: reverse_a%NZ_v'
       CALL SPACE_dealloc_array( reverse_a%NZ_v,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN

       array_name = 'bqp: reverse_a%NZ_prod'
       CALL SPACE_dealloc_array( reverse_a%NZ_prod,                            &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN
     END IF

     array_name = 'bqp: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%SUB'
     CALL SPACE_dealloc_array( data%SUB,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%PPROD'
     CALL SPACE_dealloc_array( data%PPROD,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%HPROD'
     CALL SPACE_dealloc_array( data%HPROD,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%APROD'
     CALL SPACE_dealloc_array( data%APROD,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%NZ_v'
     CALL SPACE_dealloc_array( data%NZ_v,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%NZ_hprod'
     CALL SPACE_dealloc_array( data%NZ_hprod,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%NZ_aprod'
     CALL SPACE_dealloc_array( data%NZ_aprod,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%BQP_V'
     CALL SPACE_dealloc_array( data%BQP_V,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%U'
     CALL SPACE_dealloc_array( data%U,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%PROD'
     CALL SPACE_dealloc_array( data%PROD,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine CQPS_terminate

     END SUBROUTINE CQPS_terminate

!  End of module CQPS

   END MODULE GALAHAD_CQPS_double
