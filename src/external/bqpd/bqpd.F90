! THIS VERSION: GALAHAD 5.6 - 2026-10-06 AT 07:40 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-*-  G A L A H A D _ B Q P D   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released in GALAHAD Version 5.6. October 4th 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_BQPD_precision

!     ------------------------------------------------
!     |                                              |
!     | Minimize the quadratic objective function    |
!     |                                              |
!     |         1/2 x^T H x + g^T x + f              |
!     |                                              |
!     | subject to the linear constraints and bounds |
!     |                                              |
!     |          c_l <= A x <= c_u                   |
!     |          x_l <=  x <= x_u                    |
!     |                                              |
!     | for some positive definite Hessian using     !
!     | Roger Fletcher's BQPD package (which must be |
!     | provided externally). See README.bqpd for    |
!     | details of how this is possible              |
!     |                                              |
!     ------------------------------------------------

      USE GALAHAD_KINDS_precision, ONLY: i4_, ip_, rp_
      USE GALAHAD_CLOCK, ONLY: CLOCK_time
      USE GALAHAD_SYMBOLS, ONLY: GALAHAD_ok
      USE GALAHAD_SPACE_precision, ONLY: SPACE_resize_array, SPACE_dealloc_array
      USE GALAHAD_SPECFILE_precision, ONLY: SPECFILE_item_type, SPECFILE_read, &
                                            SPECFILE_assign_value
      USE GALAHAD_SMT_precision, ONLY: SMT_TYPE, SMT_GET, SMT_PUT
      USE GALAHAD_QPT_precision, ONLY: QPT_problem_type, QPT_keyword_A,        &
                                       QPT_keyword_H, QPT_summarize_problem
      USE GALAHAD_QPD_precision, ONLY: QPD_data_type, QPD_SIF
      USE GALAHAD_CONVERT_precision, ONLY: CONVERT_control_type,               &
                                           CONVERT_inform_type,                &
                                           CONVERT_to_sparse_row_format,       &
                                          CONVERT_to_symmetric_coordinate_format

      IMPLICIT NONE ( TYPE, EXTERNAL )

      PRIVATE 
      PUBLIC :: BQPD_initialize, BQPD_read_specfile, BQPD_solve,               &
                BQPD_terminate, QPT_problem_type, SMT_type, SMT_put, SMT_get

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE BQPD_initialize
       MODULE PROCEDURE BQPD_initialize, BQPD_full_initialize
     END INTERFACE BQPD_initialize

     INTERFACE BQPD_terminate
       MODULE PROCEDURE BQPD_terminate, BQPD_full_terminate
     END INTERFACE BQPD_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: BQPD_control_type

!   error and warning diagnostics occur on stream error

        INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

        INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

        INTEGER ( KIND = ip_ ) :: print_level = 0

!   the maximum number of levels of recursion allowed

        INTEGER ( KIND = ip_ ) :: max_levels = 100

!   the space required for storing the row spikes of the L matrix

        INTEGER ( KIND = ip_ ) :: spike_space = 100000

!  the maximum number of iterative refinements per linear solve

        INTEGER ( KIND = ip_ ) :: nrep = 2

!  the minimum number of iterations before iterative refinements are used

        INTEGER ( KIND = ip_ ) :: npiv = 3

!  the maximum number of unsuccessful restarts allowed

#ifdef REAL_32
        INTEGER ( KIND = ip_ ) :: nres = 3
#else
        INTEGER ( KIND = ip_ ) :: nres = 2
#endif

!  the maximum interval between refactorizations

#ifdef REAL_32
        INTEGER ( KIND = ip_ ) :: nfreq = 100
#else
        INTEGER ( KIND = ip_ ) :: nfreq = 500
#endif

!   if the objective function value is smaller than obj_unbounded, it will be
!    flagged as unbounded from below.

        REAL ( KIND = rp_ ) :: obj_unbounded = - one / epsmch ** 2

!   the hoped-for relative accuracy in the solution

        REAL ( KIND = rp_ ) :: tol = epsmch ** 0.75

!  the maximum allowable relative error in two numbers that would be equal 
!  in exact arithmetic

#ifdef REAL_32
        REAL ( KIND = rp_ ) :: sgnf = ten ** ( - 1 )
#elif REAL_128
        REAL ( KIND = rp_ ) :: sgnf = ten ** ( - 8 )
#else
        REAL ( KIND = rp_ ) :: sgnf = ten ** ( - 4 )
#endif

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

     END TYPE BQPD_control_type

!  - - - - - - - - - - - - - - - - - -
!   inform derived type with defaults
!  - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: BQPD_inform_type

!  return status from Fletcher's BQPD:
!     0 = solution obtained
!     1 = unbounded problem detected (f(x)<=fmin would occur)
!     2 = bl(i) > bu(i) for some i
!     3 = infeasible problem detected in Phase 1
!     4 = incorrect setting of m, n, kmax, mlp, mode or tol
!     5 = not enough space in lp
!     6 = not enough space for reduced Hessian matrix (increase kmax)
!     7 = not enough space for sparse factors (sparse code only)
!     8 = maximum number of unsuccessful restarts taken
!    >8 = possible use by later sparse matrix codes

        INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER ( KIND = ip_ ) :: iter = - 1

!  the value of the objective function at the best estimate of the solution
!   determined by BQPD_solve

        REAL ( KIND = rp_ ) :: obj = HUGE( one )

      END TYPE BQPD_inform_type

!  - - - - - - - - - - - - - -
!   data derived type for BQPD
!  - - - - - - - - - - - - - -

      TYPE, PUBLIC :: BQPD_data_type
        INTEGER ( KIND = ip_ ) :: n, m, h_ne, a_ne
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IGA, LP, LS, LWS
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G, B_l, B_u
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: GA, ALP, R, E, W, WS
        LOGICAL :: original_a, original_h
        LOGICAL :: new_structure = .TRUE.
        TYPE ( SMT_type ) :: A, H
      END TYPE BQPD_data_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: BQPD_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( BQPD_data_type ) :: BQPD_data
        TYPE ( BQPD_control_type ) :: BQPD_control
        TYPE ( BQPD_inform_type ) :: BQPD_inform
        TYPE ( QPT_problem_type ) :: prob
      END TYPE BQPD_full_data_type

   CONTAINS

!-*-*-*-*-*-   B Q P D _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE BQPD_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for BQPD. This routine should be called before
!  BQPD_solve
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

      TYPE ( BQPD_data_type ), INTENT( INOUT ) :: data
      TYPE ( BQPD_control_type ), INTENT( OUT ) :: control
      TYPE ( BQPD_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

      RETURN

!  End of BQPD_initialize

      END SUBROUTINE BQPD_initialize

!- G A L A H A D -  B Q P D _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE BQPD_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for BQPD controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BQPD_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( BQPD_control_type ), INTENT( OUT ) :: control
     TYPE ( BQPD_inform_type ), INTENT( OUT ) :: inform

     CALL BQPD_initialize( data%bqpd_data, control, inform )

     RETURN

!  End of subroutine BQPD_full_initialize

     END SUBROUTINE BQPD_full_initialize

!-*-*-*-*-   B Q P D _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE BQPD_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by BQPD_initialize could (roughly)
!  have been set as:

! BEGIN BQPD SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  max-levels                                        100
!  spike-space                                       100000
!  max-refinements                                   2
!  min-iterations-before-refinement                  3
!  max-restarts                                      2
!  max-refactorization-interval                      500
!  minimum-objective-before-unbounded                -1.0D+32
!  accuracy-requied                                  1.0D-12
!  max-allowable-error                               1.0D-4
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END BQPD SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( BQPD_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: max_levels = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: spike_space = max_levels + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: nrep = spike_space + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: npiv = nrep + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: nres = npiv + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: nfreq = nres + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: obj_unbounded = nfreq + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tol = obj_unbounded + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: sgnf = tol + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = sgnf + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'BQPD'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( max_levels )%keyword = 'max-levels'
      spec( spike_space )%keyword = 'spike-space'
      spec( nrep )%keyword = 'max-refinements'
      spec( npiv )%keyword = 'min-refinements'
      spec( nres )%keyword = 'max-restarts'
      spec( nfreq )%keyword = 'max-refactorization-interval'

!  Real key-words

      spec( obj_unbounded )%keyword = 'minimum-objective-before-unbounded'
      spec( tol )%keyword = 'accuracy-requied'
      spec( sgnf )%keyword = 'max-allowable-error'

!  Logical key-words

      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' bqpd: ', alt_specname

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
     CALL SPECFILE_assign_value( spec( max_levels ),                           &
                                 control%max_levels,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( spike_space ),                          &
                                 control%spike_space,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( nrep ),                                 &
                                 control%nrep,                                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( npiv ),                                 &
                                 control%npiv,                                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( nres ),                                 &
                                 control%nres,                                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( nfreq ),                                &
                                 control%nfreq,                                &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( tol ),                                  &
                                 control%tol,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sgnf ),                                 &
                                 control%sgnf,                                 &
                                 control%error )
!  Set logical values

     CALL SPECFILE_assign_value( spec( obj_unbounded ),                        &
                                 control%obj_unbounded,                        &
                                 control%error )
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

      END SUBROUTINE BQPD_read_specfile

!-*-*-*-*-*-*-*-*-*-   B Q P D _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE BQPD_solve( prob, data, control, inform )

!  solve the quadratic program using the BQPD package

!  A - by rows (actually A -> A' in the BQPD notation)
!  H - upper triangle by coordinates

!  dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( BQPD_data_type ), INTENT( INOUT ) :: data
      TYPE ( BQPD_control_type ), INTENT( IN ) :: control
      TYPE ( BQPD_inform_type ), INTENT( OUT ) :: inform

!  local variables

      INTEGER ( KIND = ip_ ) :: n, m, np1, npm, a_ne, h_ne, maxa
      INTEGER ( KIND = ip_ ) :: i, j, k, k_max, peq, max_levels, spike_space
      REAL ( KIND = rp_ ) :: f_opt
      CHARACTER ( LEN = 80 ) :: array_name
      INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: INFO
      TYPE ( CONVERT_control_type ) :: control_convert
      TYPE ( CONVERT_inform_type ) :: inform_convert

!  common blocks (ouch!)

      INTEGER ( KIND = ip_ ) :: nout, print_level, iprint, nrep, npiv, nres
      INTEGER ( KIND = ip_ ) :: nup, nfreq, len_ws, len_lws
      INTEGER :: len_ws_gdotx, len_lws_gdotx, len_ws_bqpd, len_lws_bqpd
      INTEGER :: len_ws_sparsel, len_lws_sparsel
      REAL ( KIND = rp_ ) :: eps, tol, emin, sgnf

      COMMON / noutc / nout
      COMMON / iprintc / print_level
      COMMON / epsc / eps, tol, emin
      COMMON / repc / sgnf, nrep, npiv, nres
      COMMON / refactorc / nup, nfreq
      COMMON / wsc / len_ws_gdotx, len_lws_gdotx, len_ws_bqpd, len_lws_bqpd,   &
                     len_ws, len_lws

!  external

      EXTERNAL :: BQPD

!  transfer the data into BQPD's QP format. Start with bound data

      n = prob%n ; m = prob%m ; np1 = n + 1 ; npm = n + m
      IF ( data%new_structure ) THEN
        array_name = 'bqpd: data%B_l'
        CALL SPACE_resize_array( npm, data%B_l, inform%status,                 &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%B_u'
        CALL SPACE_resize_array( npm, data%B_u, inform%status,                 &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF
      data%B_l( : n ) = prob%X_l( : n )
      data%B_l( np1 : npm ) = prob%C_l( : m )
      data%B_u( : n ) = prob%X_u( : n )
      data%B_u( np1 : npm ) = prob%C_u( : m )

!  set space for gradients

      a_ne = prob%A%ne ; h_ne = prob%H%ne ; maxa = a_ne + n
      IF ( data%new_structure ) THEN
        array_name = 'bqpd: data%GA'
        CALL SPACE_resize_array( maxa, data%GA, inform%status,                 &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%IGA'
        CALL SPACE_resize_array( 0, maxa + m + 3, data%IGA, inform%status,     &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

!  include the gradient g in GA and IGA

      data%GA( 1 : n ) = prob%G( 1 : n )
      data%IGA( 0 ) = maxa + 1
      data%IGA( 1 : n ) = [ ( i, i = 1, n ) ]

!  include the Jacobian A in GA and IGA

      data%original_a = SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS'
      IF ( data%original_a ) THEN
        data%GA( n + 1 : n + a_ne ) = prob%A%val( : a_ne )
        data%IGA( n + 1 : n + a_ne ) = prob%A%col( : a_ne )
        data%IGA( n + a_ne + 1 ) = 1
        data%IGA( n + a_ne + 2 : n + a_ne + m + 2 ) = prob%A%ptr( : m + 1 ) + n

!  if necessary, convert the input A into by sparse-row format

      ELSE
        CALL CONVERT_to_sparse_row_format( prob%A, data%A, control_convert,    &
                                           inform_convert )
        data%GA( n + 1 : n + a_ne ) = data%A%val( : a_ne )
        data%IGA( n + 1 : n + a_ne ) = data%A%col( : a_ne )
        data%IGA( n + a_ne + 1 ) = 1
        data%IGA( n + a_ne + 2 : n + a_ne + m + 2 ) = data%A%ptr( : m + 1 ) + n
     END IF

!  if necessary, convert the input H into by sparse-co-ordinate format

      data%original_h = SMT_get( prob%H%type ) == 'COORDINATE'
      IF ( .NOT. data%original_h )                                             &
        CALL CONVERT_to_symmetric_coordinate_format( prob%H, data%H,           &
                                                     control_convert,          &
                                                     inform_convert )

!  assign default and non-default settings prior to solution

      nout = control%out
      iprint = control%print_level
      max_levels = control%max_levels
      spike_space = control%spike_space
      nrep = control%nrep
      npiv = control%npiv
      nres = control%nres
      nfreq = control%nfreq
      tol = control%tol
      sgnf = control%sgnf
      k = 0 ! dimension of reduced space (not used for cold start)
      k_max = MIN( 2000, n ) ! max allowed value of k (not used)

!  allocate workspace

      IF ( data%new_structure ) THEN
        array_name = 'bqpd: data%ALP'
        CALL SPACE_resize_array( max_levels, data%ALP, inform%status,          &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%LP'
        CALL SPACE_resize_array( max_levels, data%LP, inform%status,           &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        len_ws_gdotx = h_ne
        len_lws_gdotx = 2 * h_ne + 1
        len_ws_bqpd = k_max * ( k_max + 9 ) / 2 + npm + m
        len_lws_bqpd = k_max
        len_ws_sparsel = 5 * n + spike_space
        len_lws_sparsel = 9 * n + m
        len_ws = len_ws_gdotx + len_ws_bqpd + len_ws_sparsel
        len_lws = len_lws_gdotx + len_lws_bqpd + len_lws_sparsel

        array_name = 'bqpd: data%G'
        CALL SPACE_resize_array( n, data%G, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%R'
        CALL SPACE_resize_array( npm, data%R, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%W'
        CALL SPACE_resize_array( npm, data%W, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%E'
        CALL SPACE_resize_array( npm, data%E, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%LS'
        CALL SPACE_resize_array( npm, data%LS, inform%status,                  &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%LWS'
        CALL SPACE_resize_array( len_lws, data%LWS, inform%status,             &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%WS'
        CALL SPACE_resize_array( len_ws, data%WS, inform%status,               &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

!  fill bqpd workspace arrays

      data%LWS( 1 ) = h_ne
      data%LWS( 2 : h_ne + 1 ) = prob%H%row( : h_ne )
      data%LWS( h_ne + 2 : 2 * h_ne + 1 ) = prob%H%col( : h_ne )
      data%WS( 1 : h_ne ) = prob%H%val( : h_ne )

!  solve the problem

      CALL BQPD( n, m, k, k_max, data%GA, data%IGA, prob%X,                    &
                 data%B_l, data%B_u, f_opt, control%obj_unbounded, data%G,     &
                 data%R, data%W, data%E, data%LS, data%ALP, data%LP,           &
                 max_levels, peq, data%WS, data%LWS, 0_ip_,                    &
                 inform%status, INFO, control%print_level, control%out )

!  if successful, record the solution

      IF ( inform%status == 0 ) THEN
        DO i = 1, n - k ! active constraints
          j = data%LS( i )
          IF ( j < 0 ) THEN ! active at upper bound
            data%R( - j ) = - data%R( - j )
          END IF
        END DO
        data%R( ABS( data%LS( n - k + 1 : npm ) ) ) = zero

        array_name = 'bqpd: prob%Y'
        CALL SPACE_resize_array( m, prob%Y, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: prob%Z'
        CALL SPACE_resize_array( n, prob%Z, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        prob%Y( : m ) = data%R( np1 : npm )
        prob%Z( : n ) = data%R( : n )
        inform%iter = INFO( 1 )
        inform%obj = f_opt + prob%f
      END IF

!  ensure that the existing structure remains for subsequent calls until
!  QP_BQPD_terminate removes it

      data%new_structure = .FALSE.

      RETURN

!  End of BQPD_solve

      END SUBROUTINE BQPD_solve

!-*-*-*-*-*-*-*-*-*-*-*-   G D O T X   S U B R O U T I N E   -*-*-*-*-*-*-*-*-*

!!$      SUBROUTINE GDOTX( n, X, WS, LWS, V ) BIND( C, NAME = "gdotx_" ) 
!!$
!!$!  given H and x, form v = H x
!!$
!!$!  internal subroutine required by BQPD
!!$
!!$!  Dummy arguments
!!$
!!$      INTEGER ( ip_ ), INTENT( IN ) :: n
!!$      INTEGER ( ip_ ), INTENT( IN ), DIMENSION( 0 : * ) :: LWS
!!$      REAL ( rp_ ), INTENT( IN ), DIMENSION( n ) :: X
!!$      REAL ( rp_ ), INTENT( IN ), DIMENSION( * ) :: WS
!!$      REAL ( rp_ ), INTENT( OUT ), DIMENSION( n ) :: V
!!$
!!$!  Local variables
!!$
!!$      INTEGER ( ip_ ) :: i, ij, j, ng
!!$      REAL ( rp_ ) :: a, b
!!$
!!$!  special case for tridiagonal Toeplitz H
!!$
!!$      IF ( LWS( 0 ) == 0 ) THEN
!!$        a = WS( 1 )
!!$        b = WS( 2 )
!!$        V( 1 ) = a * X( 1 ) + b * x( 2 )
!!$        DO i = 2, n - 1
!!$           v(i) = a * X( i ) + b * ( X( i - 1 ) + X( i + 1 ) )
!!$        END DO
!!$        V( n ) = b * X( n - 1 ) + a * X( n )
!!$
!!$! normal case where the upper triangle of H is in co-ordinate format
!!$
!!$      ELSE
!!$        DO i = 1, n
!!$          V( i ) =  0.0_rp_
!!$        enddo
!!$        ng = LWS( 0 )
!!$!       WRITE(6,*) ' new product'
!!$        DO ij = 1,ng
!!$          i = LWS( ij )
!!$          j = LWS( ng + ij )
!!$!         WRITE(6,*) ' i, j, val ', i, j, WS( ij )
!!$          V( i ) = v( i ) + WS( ij ) * X( j )
!!$          IF ( i /= j ) V( j ) = V( j ) + WS( ij ) * X( i )
!!$        END DO
!!$!       WRITE(6,*) 'v  = ', V
!!$      END IF
!!$      RETURN
!!$      END SUBROUTINE GDOTX

!-*-*-*-*-*-*-   B Q P D _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE BQPD_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine BQPD_initialize
!   control see Subroutine BQPD_initialize
!   inform  see Subroutine BQPD_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( BQPD_data_type ), INTENT( INOUT ) :: data
      TYPE ( BQPD_control_type ), INTENT( IN ) :: control
      TYPE ( BQPD_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

      array_name = 'bqpd: data%IGA'
      CALL SPACE_dealloc_array( data%IGA, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%LP'
      CALL SPACE_dealloc_array( data%LP, inform%status,                        &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%LS'
      CALL SPACE_dealloc_array( data%LS, inform%status,                        &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%LWS'
      CALL SPACE_dealloc_array( data%LWS, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%GA'
      CALL SPACE_dealloc_array( data%GA, inform%status,                        &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%G'
      CALL SPACE_dealloc_array( data%G, inform%status,                         &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%B_l'
      CALL SPACE_dealloc_array( data%B_l, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%B_u'
      CALL SPACE_dealloc_array( data%B_u, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%ALP'
      CALL SPACE_dealloc_array( data%ALP, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%R'
      CALL SPACE_dealloc_array( data%R, inform%status,                         &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%E'
      CALL SPACE_dealloc_array( data%E, inform%status,                         &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%W'
      CALL SPACE_dealloc_array( data%W, inform%status,                         &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpd: data%WS'
      CALL SPACE_dealloc_array( data%WS, inform%status,                        &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      IF ( .NOT. data%original_a ) THEN
        array_name = 'bqpd: data%A%ptr'
        CALL SPACE_dealloc_array( data%A%ptr, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%A%col'
        CALL SPACE_dealloc_array( data%A%row, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%A%val'
        CALL SPACE_dealloc_array( data%A%val, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%A%type'
        CALL SPACE_dealloc_array( data%A%type, inform%status,                  &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN
      END IF

      IF ( .NOT. data%original_h ) THEN
        array_name = 'bqpd: data%H%row'
        CALL SPACE_dealloc_array( data%H%row, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%H%col'
        CALL SPACE_dealloc_array( data%H%col, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%H%val'
        CALL SPACE_dealloc_array( data%H%val, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bqpd: data%H%type'
        CALL SPACE_dealloc_array( data%H%type, inform%status,                  &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN
      END IF

!  ensure that the structure will be re-initialised on any subsequent call

      data%new_structure = .TRUE.
      RETURN

!  End of subroutine BQPD_terminate

      END SUBROUTINE BQPD_terminate

! - G A L A H A D -  B Q P D _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE BQPD_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BQPD_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( BQPD_control_type ), INTENT( IN ) :: control
     TYPE ( BQPD_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL BQPD_terminate( data%bqpd_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'cqp: data%prob%X'
     CALL SPACE_dealloc_array( data%prob%X,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%X_l'
     CALL SPACE_dealloc_array( data%prob%X_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%X_u'
     CALL SPACE_dealloc_array( data%prob%X_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%G'
     CALL SPACE_dealloc_array( data%prob%G,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%Y'
     CALL SPACE_dealloc_array( data%prob%Y,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%Z'
     CALL SPACE_dealloc_array( data%prob%Z,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%C'
     CALL SPACE_dealloc_array( data%prob%C,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%C_l'
     CALL SPACE_dealloc_array( data%prob%C_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%C_u'
     CALL SPACE_dealloc_array( data%prob%C_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%WEIGHT'
     CALL SPACE_dealloc_array( data%prob%WEIGHT,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%H%type'
     CALL SPACE_dealloc_array( data%prob%H%type,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%H%ptr'
     CALL SPACE_dealloc_array( data%prob%H%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%H%row'
     CALL SPACE_dealloc_array( data%prob%H%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%H%col'
     CALL SPACE_dealloc_array( data%prob%H%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%H%val'
     CALL SPACE_dealloc_array( data%prob%H%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%A%type'
     CALL SPACE_dealloc_array( data%prob%A%type,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%A%ptr'
     CALL SPACE_dealloc_array( data%prob%A%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%A%row'
     CALL SPACE_dealloc_array( data%prob%A%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%A%col'
     CALL SPACE_dealloc_array( data%prob%A%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%A%val'
     CALL SPACE_dealloc_array( data%prob%A%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine BQPD_full_terminate

     END SUBROUTINE BQPD_full_terminate

!  End of module BQPD

   END MODULE GALAHAD_BQPD_precision
