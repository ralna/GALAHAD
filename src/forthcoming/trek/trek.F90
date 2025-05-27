! THIS VERSION: GALAHAD 5.3 - 2025-05-24 AT 14:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ T R E K    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Fowkes/Gould/Montoison/Orban, for GALAHAD productions
!  Principal author: Hussam Al Daas and Nick Gould

!  History -
!   originally released in GALAHAD Version 5.2. February 15th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_TREK_precision

     USE GALAHAD_KINDS_precision
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPACE_precision
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_SMT_precision
     USE GALAHAD_NORMS_precision, ONLY: TWO_NORM
     USE GALAHAD_MOP_precision, ONLY: TREK_Hv => MOP_Ax
     USE GALAHAD_BLAS_inter_precision, ONLY: TBSV
     USE GALAHAD_LAPACK_inter_precision, ONLY: LAENV, GEQRF, ORGQR, SYEV,      &
                                               PBTRF, PBTRS
     USE GALAHAD_SLS_precision, ONLY: SLS_control_type, SLS_inform_type,       &
                                      SLS_data_type, SLS_initialize,           &
                                      SLS_read_specfile, SLS_analyse,          &
                                      SLS_factorize, SLS_solve, SLS_terminate
     USE GALAHAD_TRS_precision, ONLY: TRS_control_type, TRS_inform_type,       &
                                      TRS_read_specfile, TRS_solve_diagonal

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: TREK_initialize, TREK_read_specfile,                            &
               TREK_solve, TREK_resolve, TREK_solve_all, TREK_terminate,       &
               TREK_full_initialize, TREK_full_terminate,                      &
               SMT_type, SMT_put, SMT_get

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
     REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = rp_ ), PARAMETER :: violation_max = 0.0001_rp_

     INTEGER ( KIND = ip_ ) :: ldp = 3 ! P is symmetric, pentadiagonal band

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: TREK_control_type

!   error and warning diagnostics occur on stream error

       INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

       INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

       INTEGER ( KIND = ip_ ) :: print_level = 0

!  maximum iteration count

       INTEGER ( KIND = ip_ ) :: maxit = - 1

!  reduction factor for subsequent trust-region radii, i = 2,...,m

       REAL ( KIND = rp_ ) :: reduction = 0.5_rp_

!  stopping tolerance for the cheaply-computed residual

!      ||A X + X D + ce'|| < tol ||c e'||

       REAL ( KIND = rp_ ) :: stop_residual = ten * SQRT( epsmch )

!  should the incoming Lanczos vectors be re-orthogonalised against the
!  existing ones (this can be expensive)

       LOGICAL :: reorthogonalize = .FALSE.

!  should the exact shifts for each radius be computed at every iteration
!  (this can be expensive)

       LOGICAL :: exact_shift = .FALSE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

       LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal = .FALSE.

!  positive-definite linear equation solver

       CHARACTER ( LEN = 30 ) :: solver = "ssids" // REPEAT( ' ', 25 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""' // REPEAT( ' ', 28 )

!  control parameters for SLS

       TYPE ( SLS_control_type ) :: sls_control

!  control parameters for TRS

       TYPE ( TRS_control_type ) :: trs_control

     END TYPE TREK_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: TREK_inform_type

!  return status. See TREK_solve for details

        INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER ( KIND = ip_ ) :: iter = - 1

!  the number of orthogonal vectors required

        INTEGER ( KIND = ip_ ) :: n_vec = - 1

!  the value of the quadratic function

        REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the M-norm of x, ||x||_M

        REAL ( KIND = rp_ ) :: x_norm = zero

!  the Lagrange multiplier corresponding to the trust-region constraint

        REAL ( KIND = rp_ ) :: multiplier = zero

!  the current radius

        REAL ( KIND = rp_ ) :: radius = - one

!  the proposed next radius to be used

        REAL ( KIND = rp_ ) :: next_radius = - one

!  the maximum relative residual error

        REAL ( KIND = rp_ ) :: error = - one

!  inform parameters for SLS

        TYPE ( SLS_inform_type ) :: sls_inform

!  inform parameters for TRS

        TYPE ( TRS_inform_type ) :: trs_inform

      END TYPE TREK_inform_type

!  ...................
!   data derived type
!  ...................

      TYPE, PUBLIC :: TREK_data_type

        INTEGER ( KIND = ip_ ) :: is_max, k_max, lwork_syev, n_v
        INTEGER ( KIND = ip_ ) :: k_exit = - 1
        REAL ( KIND = rp_ ) :: c_norm, last_radius, last_shift
        LOGICAL :: allocated_arrays = .FALSE.

!  common workspace arrays

        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: V, P, P_shift
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: Q, S, S2
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: ALPHA, BETA, DELTA
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: U, U1, S1, W, SHIFT
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C, X, D, WORK_syev

!  local copy of control

        TYPE ( TREK_control_type ) :: control

!  data for SLS

        TYPE ( SLS_data_type ) :: sls_data
      END TYPE TREK_data_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: TREK_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( TREK_data_type ) :: TREK_data
        TYPE ( TREK_control_type ) :: TREK_control
        TYPE ( TREK_inform_type ) :: TREK_inform
      END TYPE TREK_full_data_type

   CONTAINS

!-*-*-*-*-*-   T R E K _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE TREK_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for TREK. This routine should be called before
!  TREK_solve
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

      TYPE ( TREK_data_type ), INTENT( INOUT ) :: data
      TYPE ( TREK_control_type ), INTENT( OUT ) :: control
      TYPE ( TREK_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Initalize SLS components

      CALL SLS_initialize( control%solver, data%SLS_data,                     &
                           control%SLS_control, inform%SLS_inform )
!     control%SLS_control%perturb_to_make_definite = .FALSE.
!     control%SLS_control%preconditioner = 2
      control%SLS_control%prefix = '" - SLS:"                    '

      RETURN

!  End of TREK_initialize

      END SUBROUTINE TREK_initialize

!- G A L A H A D -  T R E K _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE TREK_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for TREK controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( TREK_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( TREK_control_type ), INTENT( OUT ) :: control
     TYPE ( TREK_inform_type ), INTENT( OUT ) :: inform

     CALL TREK_initialize( data%trek_data, control, inform )

     RETURN

!  End of subroutine TREK_full_initialize

     END SUBROUTINE TREK_full_initialize

!-*-*-*-*-   T R E K _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE TREK_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by TREK_initialize could (roughly)
!  have been set as:

! BEGIN TREK SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  maximum-number-of-iterations                      -1
!  start-error-search                                0
!  radius-reduction-factor                           0.5E+0
!  residual-accuracy                                 1.0E-8
!  reorthogonalize-vectors                           F
!  exact-shift                                       F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  symmetric-linear-equation-solver                  ssids
!  output-line-prefix                                ""
! END TREK SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( TREK_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxit = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: reduction = maxit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_residual = reduction + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: reorthogonalize = stop_residual + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: exact_shift = reorthogonalize + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = exact_shift + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: solver = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = solver + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'TREK'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( maxit )%keyword = 'maximum-number-of-iterations'

!  Real key-words

      spec( reduction )%keyword = 'radius-reduction-factor'
      spec( stop_residual )%keyword = 'residual-accuracy'

!  Logical key-words

      spec( reorthogonalize )%keyword = 'reorthogonalize-vectors'
      spec( exact_shift )%keyword = 'exact-shift'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( solver )%keyword = 'symmetric-linear-equation-solver'
      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' trek: ', alt_specname

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

!  Set real values

     CALL SPECFILE_assign_value( spec( reduction ),                            &
                                 control%reduction,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_residual ),                        &
                                 control%stop_residual,                        &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( reorthogonalize ),                      &
                                 control%reorthogonalize,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( exact_shift ),                          &
                                 control%exact_shift,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
!  Set character values

     CALL SPECFILE_assign_value( spec( solver ),                               &
                                 control%solver,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  Read the specfiles for SLS and TRS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-SLS')
        CALL TRS_read_specfile( control%TRS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-TRS')
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
        CALL TRS_read_specfile( control%TRS_control, device )
      END IF

      RETURN

      END SUBROUTINE TREK_read_specfile

!-*-*-*-*-*-*-*-*-   T R E K _ S O L V E    S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE TREK_solve( n, H, C, radius, X, data, control, inform )

!  Given an n x n symmetric matrix H, an n-vector c, and a radius > 0,
!  approximately solve the trust-region subproblem
!
!    min 1/2 x'Hx + c'x : ||x|| <= radius
!
!  using an extended Krylov subspace method.
!
!  The method uses the "backward" estended-Krylov subspace
!
!    K_2k = { c, H^{-1} c, H c, ..., H^{-k} c },
!
!  (see module EKS)
!
!  Input:
!   n - number of unknowns
!   H - symmetric coefficient matrix, H, from the quadratic term, 
!       in any symmetric format supported by the SMT type
!   C - vector c from linear term
!   radius - scalar trust-region radius > 0
!   control - parameters structure (see preamble)
!   inform - output structure (see preamble)
!   data - prvate internal workspace
!
!   Output:
!   X - solution vector x

!  dummy arguments

     INTEGER, INTENT( IN ) :: n
     TYPE ( SMT_type ), INTENT( IN ) :: H
     REAL ( KIND = rp_), INTENT( IN ), DIMENSION( n ) :: C
     REAL ( KIND = rp_), INTENT( IN ) :: radius
     REAL ( KIND = rp_), INTENT( OUT ), DIMENSION( n  ) :: X
     TYPE ( TREK_data_type ), INTENT( INOUT ) :: data
     TYPE ( TREK_control_type ), INTENT( IN ) :: control
     TYPE ( TREK_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER :: i, j, jj, j_max, k, km1, k2, k2m1, k2p1, mkp1, nb
     INTEGER :: lapack_info, out
     REAL ( KIND = rp_) :: alpha, beta, gamma, delta, newton_norm
     REAL ( KIND = rp_) :: e11, e12, e22, error, error_j, s_norm, w_norm
     REAL ( KIND = rp_) :: violation, shift
     LOGICAL :: printi, printm, printd
     LOGICAL :: termination_test = .TRUE.
     CHARACTER ( LEN = 80 ) :: array_name

!  temporary debug variables - ultimately remove

     INTEGER :: ii
     INTEGER, PARAMETER :: p_dim = 10
     REAL ( KIND = rp_) :: P_calc( p_dim, p_dim )
     REAL ( KIND = rp_) :: R( n ), SOL( n )

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  check input dimensions are consistent

     IF ( n /= H%n ) THEN
       inform%status = GALAHAD_error_restrictions ; GO TO 920
     END IF

!  check initial radius value is positive

     IF ( radius <= zero ) THEN
       inform%status = GALAHAD_error_restrictions ; GO TO 920
     END IF
     inform%radius = radius
     data%last_radius = radius

!  record output values

     out = control%out
     printi = control%print_level > 0 .AND. out > 0
     printm = control%print_level > 2 .AND. out > 0
     printd = control%print_level > 5 .AND. out > 0

!  record the iteration limit

     data%control = control
     data%k_max = MAX( control%maxit, 1 )
     data%is_max = 2 * data%k_max
     data%k_exit = - 1

!   provide space for, and initiate, the components of the projected solution

     IF ( .NOT. data%allocated_arrays ) THEN
       data%allocated_arrays = .TRUE.

       array_name = 'trek: data%V'
       CALL SPACE_resize_array(  1_ip_, n, - data%k_max, data%k_max, data%V,   &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

!  set the projected matrix P and solution s

       array_name = 'trek: data%P'
       CALL SPACE_resize_array( ldp, data%is_max, data%P,                      &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%P_shift'
       CALL SPACE_resize_array( ldp, data%is_max, data%P_shift,                &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%S1'
       CALL SPACE_resize_array( data%is_max, data%S1,                          &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%S2'
       CALL SPACE_resize_array( data%is_max, 1, data%S2,                       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%W'
       CALL SPACE_resize_array( data%is_max, data%W,                           &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

!  set workspace arrays

       array_name = 'trek: data%ALPHA'
       CALL SPACE_resize_array( - data%k_max, data%k_max, data%ALPHA,          &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%BETA'
       CALL SPACE_resize_array( - data%k_max, data%k_max, data%BETA,           &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%DELTA'
       CALL SPACE_resize_array( - data%k_max, data%k_max, data%DELTA,          &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%U'
       CALL SPACE_resize_array( n, data%U,                                     &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       IF ( control%exact_shift ) THEN
         array_name = 'trek: data%Q'
         CALL SPACE_resize_array( data%is_max, data%is_max, data%Q,            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%D'
         CALL SPACE_resize_array( data%is_max, data%D,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%C'
         CALL SPACE_resize_array( data%is_max, data%C,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%X'
         CALL SPACE_resize_array( data%is_max, data%X,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

!  discover the size of, and allocate, workspace needed for the eigensolver

         nb = LAENV( 1_ip_, 'DSYTRD', 'L', data%is_max,                        &
                    - 1_ip_, - 1_ip_, - 1_ip_ )
         data%lwork_syev                                                       &
           = MAX( 1_ip_, 3 * data%is_max - 1, ( nb + 2 ) * data%is_max )

         ALLOCATE( data%WORK_syev( data%lwork_syev ),                          &
                   STAT = inform%alloc_status )
         IF ( inform%alloc_status /= 0 ) THEN
           inform%bad_alloc = 'trek: data%WORK_syev'
           inform%status = GALAHAD_error_allocate ; GO TO 910
         END IF
       END IF
     END IF

     IF ( printd ) THEN
       array_name = 'trek: data%U1'
       CALL SPACE_resize_array( n, data%U1,                                    &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910
     END IF

!   initialize, analyse and factorize H

     CALL SLS_initialize( control%solver, data%sls_data,                       &
                          data%control%sls_control,                            &
                          inform%sls_inform, check = .TRUE. )

     CALL SLS_analyse( H, data%sls_data, data%control%sls_control,             &
                       inform%sls_inform )
     IF ( inform%sls_inform%status < 0 ) THEN
       inform%status = GALAHAD_error_analysis ; GO TO 920
     END IF

     CALL SLS_factorize( H, data%sls_data, data%control%sls_control,           &
                         inform%sls_inform )
     IF ( inform%sls_inform%status < 0 ) THEN
       inform%status = GALAHAD_error_factorization ; GO TO 920
     END IF

!  ****************************************************************************
!
!  build and use the backward extended Krylov space
!
!    K_2k = { c, H^{-1} c, H c, ..., H^{-k} c }
!
!  Do this by building an orthogonal basis matrix
!
!    V_2k = [ v_0, v_{-1}, v_1, ... , v_{-k} ]
!
!  and the pentadiagonal projected matrix P_2k = V_2k' A V_2k, with 
!  columns p_i, whose subdiagonal columns are 
!
!    p_2k-1 = ( ( 1 - beta_{-k+1} delta_k - alpha_k delta_{-k} ) / beta_{k-1} )
!             (                     alpha_{k-1}                               )
!             (           - delta_{-k} delta_{k+1} /beta_{k-1}                )
!
!  and
!
!    p_2k = ( alpha_{-k} )
!           (  delta_k   )
!           (     0      )
!
!  where the alpha_i, beta_i and delta_i are defined below
!
!  For details, see
!    H. Al Daas & N. I. M. Gould
!    Extended-Krylov-subspace trust-region methods
!    Working note STFC-RAL 2025
!
!  ****************************************************************************

!  record the initial error

     data%c_norm = TWO_NORM( C( 1 : n ) )

!  normalize the input vector, delta_0 = ||c|| and v_0 = -c/delta_0

     data%DELTA( 0 ) = data%c_norm
     data%V( 1 : n, 0 ) = - C( 1 : n ) / data%c_norm

!  set u = H^{−1} v_0

     data%U = data%V( 1 : n, 0 )
     CALL SLS_solve( H, data%U, data%sls_data, data%control%sls_control,       &
                     inform%sls_inform )
     IF ( inform%sls_inform%status < 0 ) THEN
       inform%status = GALAHAD_error_solve ; GO TO 920
     END IF

!  compute ||x_newton|| = ||H^{-1} c||

     newton_norm = data%c_norm * TWO_NORM( data%U ) 

!  set beta_0 = v_0' u and u = u - beta_0 v_0
 
     beta = DOT_PRODUCT( data%V( 1 : n, 0 ), data%U )
     data%BETA( 0 ) = beta
     data%U = data%U - beta * data%V( 1 : n, 0 )

!  set delta_{-1} = ||u||; v_{-1} = u / delta_{-1}

     data%DELTA( - 1 ) = TWO_NORM( data%U )
     data%V( 1 : n, - 1 ) = data%U / data%DELTA( - 1 )
     data%n_v = 2

!  if the Newton step is inside the trust region, exit

     IF ( newton_norm <= radius ) THEN
       X( 1 : n ) = data%c_norm * data%U

!  compute the next potential radius depending on whether ||x_newton|| <= radius

       inform%next_radius = radius
       DO
         inform%next_radius = inform%next_radius * data%control%reduction
         IF ( newton_norm > inform%next_radius ) EXIT
       END DO
       data%k_exit = 1
       data%last_shift = zero
       inform%status = GALAHAD_ok ; GO TO 900
     ELSE
       inform%next_radius = radius * data%control%reduction
     END IF

!  initialise the shift as zero

     shift = zero

!  ------------------------------------------------------------
!  start of main forward iteration loop (in comments, iter = k)
!  ------------------------------------------------------------

     IF ( printi ) WRITE( out, 2000 ) prefix
     inform%status = GALAHAD_error_max_iterations

     DO k = 1, data%k_max
       inform%iter = k
       km1 = k - 1
       mkp1 = - k + 1
       k2 = 2 * k
       k2m1 = k2 - 1
       k2p1 = k2 + 1

!  print iteration details if required

       IF ( printm ) WRITE( out, "( A, ' iteration ', I0 )" ) prefix, k

! for testing

       IF ( printd ) THEN 
         CALL TREK_Hv( one, H, data%V( 1 : n, mkp1 ), zero, data%U,            &
                       control%out, control%error, symmetric = .TRUE. )
         CALL TREK_Hv( one, H, data%V( 1 : n, k ), zero, data%U1,              &
                       control%out, control%error, symmetric = .TRUE. )
         e11 = DOT_PRODUCT( data%V( 1 : n, mkp1 ), data%U )
         e12 = DOT_PRODUCT( data%V( 1 : n, mkp1 ), data%U1 )
         e22 = DOT_PRODUCT( data%V( 1 : n, k ), data%U1 )
       END IF

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  segment involving the product with A
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  set u = H v_{-k}

       CALL TREK_Hv( one, H, data%V( 1 : n, - k ), zero, data%U,               &
                     out, control%error, symmetric = .TRUE. )

!  set alpha_{-k} = v_{-k}' u and u = u _ alpha_{-k} v_{-k}

       alpha = DOT_PRODUCT( data%V( 1 : n, - k ), data%U )
       data%ALPHA( - k ) = alpha
       data%U = data%U - alpha * data%V( 1 : n, - k )

!  set alpha {k-1} = v_{k-1}' u and u = u - alpha_{k-1} v_{k-1}

       alpha = DOT_PRODUCT( data%V( 1 : n, km1 ), data%U )
       data%ALPHA( km1 ) = alpha
       data%U = data%U - alpha * data%V( 1 : n, km1 )

!  orthogonalise wrt the remaining vectors

!      IF ( .true. ) THEN
       IF ( control%reorthogonalize ) THEN
         DO i = mkp1, km1 - 1
           beta = DOT_PRODUCT( data%V( 1 : n, i ), data%U )
           data%U = data%U - beta * data%V( 1 : n, i )
         END DO
       END IF

!   set delta_{k} = ||u|| and v_{k} = u/delta_{k}

       delta = TWO_NORM( data%U )
       gamma = - data%DELTA( - k ) * delta / data%BETA( km1 )
       data%DELTA( k ) = delta
       data%V( 1 : n, k ) = data%U / delta

!  save column 2k-1 of P

       IF ( k > 1 ) THEN
         data%P( 1, k2m1 ) = ( one - data%DELTA( - k ) * data%ALPHA( km1 )     &
                                   - data%BETA( mkp1 ) * data%DELTA( km1 ) )   &
                               / data%BETA( km1 )
       ELSE
         data%P( 1, k2m1 ) = ( one - data%DELTA( - k ) * data%ALPHA( km1 ) )   &
                               / data%BETA( km1 )
       END IF
       data%P( 2, k2m1 ) = data%ALPHA( km1 )
       data%P( 3, k2m1 ) = gamma

!  ....................---------------------
!  test system of order 2k-1 for termination
!  ....................---------------------

!  compute the projected solution ( P + shift(j) I ) s(j) = delta_0 e_1
!  for j in [1,...,m]

!      IF ( k > 1 .AND. termination_test ) THEN   
       IF ( k > 1 .AND. .FALSE. ) THEN   

!  form and factorize the shifted matrix P + shift(j) I = LL'

         data%P_shift( 1, : k2m1 ) = data%P( 1, : k2m1 ) + shift
         data%P_shift( 2 : 3, : k2m1 ) = data%P( 2 : 3, : k2m1 )
         CALL PBTRF( 'L', k2m1, 2, data%P_shift, ldp, lapack_info )
         IF ( lapack_info < 0 ) THEN
           inform%status = GALAHAD_error_lapack ; GO TO 920
         END IF

!  solve the shifted system to find the solution s(j)

         data%S2( 1, 1 ) = data%c_norm ; data%S2( 2 : k2m1, 1 ) = zero
         CALL PBTRS( 'L', k2m1, 2, 1, data%P_shift, ldp,                       &
                     data%S2, data%is_max, lapack_info )
         IF ( lapack_info < 0 ) THEN
           inform%status = GALAHAD_error_lapack ; GO TO 920
         END IF
         data%S1( : k2m1 ) = data%S2( : k2m1, 1 )

!  record ||r_2k(t)||^2 = ( alpha_k^2 + gamma^2 ) s_2k(j)^2 

         error = SQRT( data%ALPHA( km1 ) ** 2 + gamma ** 2 )                   &
                   * ABS( data%S1( k2m1 ) )

!  compute ||s(j)||

         s_norm = TWO_NORM( data%S1( 1 : k2m1 ) )
         WRITE( out, "('k = ', I3, ' ||s||, radius = ', 2ES12.4 )" )           &
           k, s_norm, radius

!  if ||s(j)|| > radius_j, update shift(j). First, solve L w_j = s_j ...

         violation = s_norm - radius
         IF ( s_norm > radius ) THEN
           data%W( : k2m1 ) = data%S1( : k2m1 )
           CALL TBSV( 'L', 'N', 'N', k2m1, 2, data%P_shift, ldp,               &
                      data%W, 1_ip_)

!  ... then use the norm of w_j to improve the shift

           w_norm = TWO_NORM( data%W( 1 : k2m1 ) )
           shift = shift + ( s_norm - radius ) * s_norm ** 2                   &
                     / ( radius * w_norm ** 2 )
         END IF

!  debug - compare predicted and actual error

         IF ( printd ) THEN
           SOL = zero
           DO ii = 1, k2m1
             IF ( ii == 1 ) THEN
               i = 0
             ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
               i = - i - 1
             ELSE
               i = - i
             END IF
             SOL = SOL + data%V( : n, i ) * data%S1( ii )
           END DO
           R = shift * SOL + C
           CALL TREK_Hv( one, H, SOL, one, R, out, control%error,              &
                         symmetric = .TRUE. )
           WRITE( out, "( ' ||r||, est = ', 2ES12.4 )" )                       &
             TWO_NORM( R ), error_j
         END IF

         IF ( printi ) WRITE( out, 2010 ) prefix, k, k2m1,                     &
              s_norm, radius, shift, error, inform%obj
         error = error / data%c_norm

!  check for termination

         IF ( error < data%control%stop_residual .AND.                         &
              violation <= violation_max ) THEN
           inform%n_vec = k2m1

!  recover the solution

           X( : n ) = zero
           DO ii = 1, k2m1
             IF ( ii == 1 ) THEN
               i = 0
             ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
               i = - i - 1
             ELSE
               i = - i
             END IF
             X( : n ) = X( : n ) + data%V( : n, i ) * data%S1( ii )
           END DO

!  exit the main loop

           inform%status = GALAHAD_ok ; EXIT
         END IF
       END IF

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!  segment involving the product with A inverse
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  set u = H^{−1} v_k

       data%U = data%V( 1 : n, k )
       CALL SLS_solve( H, data%U, data%sls_data, data%control%sls_control,     &
                       inform%sls_inform )
       IF ( inform%sls_inform%status < 0 ) THEN
         inform%status = GALAHAD_error_solve ; GO TO 920
       END IF

!  set beta_{-k} := v_{-k}' u and u = u - beta_{-k} v_{-k}
 
       beta = DOT_PRODUCT( data%V( 1 : n, - k ), data%U )
       data%BETA( - k ) = beta
       data%U = data%U - beta * data%V( 1 : n, - k )

!  set beta k = v_k' u and w := w - beta_k v_k

       beta = DOT_PRODUCT( data%V( 1 : n, k ), data%U )
       data%BETA( k ) = beta
       data%U = data%U - beta * data%V( 1 : n, k )

!  orthogonalise wrt the remaining vectors

!      IF ( .true. ) THEN
       IF ( control%reorthogonalize ) THEN
         DO i = mkp1 + 1, k - 1
           beta = DOT_PRODUCT( data%V( 1 : n, i ), data%U )
           data%U = data%U - beta * data%V( 1 : n, i )
         END DO
       END IF

!  test

       IF ( printd ) THEN
         DO i = - k, k
           DO j = - k, i
             alpha =  DOT_PRODUCT( data%V( 1 : n, i ), data%V( 1 : n, j ) )
             IF ( ( i == j  .AND. ABS( alpha - one ) > ten ** ( -5 ) ) .OR.    &
                  ( i /= j  .AND. ABS( alpha ) > ten ** ( -5 ) ) )  &
               WRITE( out, "( ' i, j, v_ij ', 2I3, ES12.4 )" ) i, j, alpha
           END DO
         END DO
       END IF

!  save column 2k of P

       data%P( 1, k2 ) = data%ALPHA( - k )
       data%P( 2, k2 ) = delta
       data%P( 3, k2 ) = zero

       IF ( printd ) THEN
         IF ( ABS( data%P( 1, k2m1 ) - e11 ) > ten ** ( - 5 ) .OR.             &
              ABS( data%P( 2, k2m1 ) - e12 ) > ten ** ( - 5 ) .OR.             &
              ABS( data%P( 1, k2 ) - e22 ) > ten ** ( - 5 ) ) THEN
           WRITE( out, "( ' theoretical (P) and computed (C) Hessian:', /,     &
          &               ' P_11, C_11 = ', 2ES12.4, /,                        &
          &               ' P_21, C_21 = ', 2ES12.4, /,                        &
          &               ' P_22, C_22 = ', 2ES12.4 )" )                       &
            data%P( 1, k2m1 ), e11, data%P( 2, k2m1 ), e12,                    &
            data%P( 1, k2 ), e22
          END IF
        END IF

!  ....................-------------------
!  test system of order 2k for termination
!  ....................-------------------

!  compute the projected solution ( P + shift(j) I ) s(j) = delta_0 e_1
!  for j in [1,...,m]

       IF (  termination_test ) THEN   
         j_max = 1

!  exact shifts are required

         IF ( control%exact_shift ) THEN

!  save the lower triangle of P in Q

           data%Q( 1 : k2, 1 : k2 ) = zero
           DO i = 1, k2
             data%Q( i, i ) = data%P( 1, i )
           END DO
           DO i = 1, k2 - 1
             data%Q( i + 1, i ) = data%P( 2, i )
           END DO
           DO i = 1, k2 - 2
             data%Q( i + 2, i ) = data%P( 3, i )
           END DO

           IF ( printd ) THEN
             WRITE( out, "( ' P = ')" )
             DO i = 1, k2
               WRITE( out, "( 4( 2I4, ES12.4 ) )" )                            &
                 ( i, j, data%Q( i, j ), j = 1, i ) 
             END DO
           END IF

!  compute the eigenvalues (in D) and eigenvalues (overwrite Q)

           CALL SYEV( 'V','L', k2, data%Q, data%is_max, data%D,                &
                      data%WORK_syev, data%lwork_syev, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF

!  form c' = Q^T ||c|| e_1

           data%C( 1 : k2 ) = data%c_norm * data%Q( 1, 1 : k2 )

!  solve the diagonal trust-region problem 
!    min 1/2 x'^T D x' + c'^T x' : ||x'|| <= radius

           data%control%trs_control%use_initial_multiplier = .TRUE.
           data%control%trs_control%initial_multiplier = shift
           CALL TRS_solve_diagonal( k2, radius, zero,                          &
                                    data%C, data%D, data%X,                    &
                                    data%control%trs_control,                  &
                                    inform%trs_inform )
           shift = inform%trs_inform%multiplier
           inform%multiplier = shift
           inform%obj = inform%trs_inform%obj
           inform%x_norm = inform%trs_inform%x_norm

!  recover x = Q x'

           data%S1( 1 : k2 ) = MATMUL( data%Q( : k2 , : k2 ), data%X( : k2 ) )

!  exact shifts are not required

         ELSE

!  form and factorize the shifted matrix P + shift(j) I

           data%P_shift( 1, : k2 ) = data%P( 1, : k2 ) + shift
           data%P_shift( 2 : 3, : k2 ) = data%P( 2 : 3, : k2 )
           CALL PBTRF( 'L', k2, 2, data%P_shift, ldp, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF

!  solve the shifted system to find the solution s(j)

           data%S2( 1, 1 ) = data%c_norm ; data%S2( 2 : k2, 1 ) = zero
           CALL PBTRS( 'L', k2, 2, 1, data%P_shift, ldp,                       &
                       data%S2, data%is_max, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF
           data%S1( 1 : k2 ) = data%S2( 1 : k2, 1 )
         END IF

!  record ||r_2k(t)||^2 = ( alpha_k s_2k-1(j))^2 + ( alpha_-k s_2k(j))^2

         error = ABS( gamma * data%S1( k2m1 ) + delta * data%S1( k2 ) )

!  compute ||s|| and the violation

         s_norm = TWO_NORM( data%S1( 1 : k2 ) )
         violation = s_norm - radius
         IF ( printm ) WRITE( out,                                             &
           "(' k = ', I0, ' ||s||, radius, multiplier = ', 3ES11.4 )" )        &
              k, s_norm, radius, shift

!  if ||s|| > radius, update shift. First, solve L w = s ...

         IF ( .NOT. control%exact_shift ) THEN
           inform%obj                                                          &
             = half * ( data%c_norm * data%S1( 1 ) - shift * s_norm ** 2 )
!          IF ( s_norm > radius ) THEN
           IF ( .TRUE. ) THEN
             data%W( : k2 ) = data%S1( : k2 )
             CALL TBSV( 'L', 'N', 'N', k2, 2, data%P_shift, ldp,               &
                        data%W, 1_ip_)

!  ... then use the norm of w_j to improve the shift

             w_norm = TWO_NORM( data%W( 1 : k2 ) )
             shift = shift + ( s_norm - radius ) * s_norm ** 2                 &
                       / ( RADIUS * w_norm ** 2 )
           END IF
         END IF

!  record ||s|| and the current shift

         inform%x_norm = s_norm
         inform%multiplier = shift

!  debug - compare predicted and actual error

         IF ( printd ) THEN
           SOL = zero
           DO ii = 1, k2
             IF ( ii == 1 ) THEN
               i = 0
             ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
               i = - i - 1
             ELSE
               i = - i
             END IF
             SOL = SOL + data%V( : n, i ) * data%S1( ii )
           END DO
           R = shift * SOL + C
           CALL TREK_Hv( one, H, SOL, one, R, out, control%error,              &
                         symmetric = .TRUE. )
           WRITE( out, "( ' ||r||, est = ', 2ES12.4 )" ) TWO_NORM( R ), error
         END IF

         IF ( printi ) WRITE( out, 2010 ) prefix, k, k2,                       &
              s_norm, radius, shift, error, inform%obj
         error = error / data%c_norm

!  check for termination

         IF ( error < data%control%stop_residual .AND.                         &
              violation <= violation_max ) THEN
           inform%n_vec = k2

!  recover the solution

           X( : n ) = zero
           DO ii = 1, k2
             IF ( ii == 1 ) THEN
               i = 0
             ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
               i = - i - 1
             ELSE
               i = - i
             END IF
             X( : n ) = X( : n ) + data%V( : n, i ) * data%S1( ii )
           END DO

!  exit the main loop

           data%k_exit = k
           data%last_shift = shift
           inform%status = GALAHAD_ok ; EXIT
         END IF
       END IF

!  set delta_{-k-1} := ||u|| 

       delta = TWO_NORM( data%U )

!  set v_{-k-1} = u/delta_{-k-1}

       IF ( k < data%k_max ) THEN
         data%DELTA( - k - 1 ) = delta
         data%V( 1 : n, - k - 1 ) = data%U / delta
         data%n_v = data%n_v + 1
         IF ( data%n_v == n ) WRITE( out, "( ' Krylov space is full space' )" )
       END IF

!  ------------------------------------
!  end of main backward iteration loop
!  ------------------------------------

     END DO
     inform%error = error

!  debug - check the theoretical P against the computed one

!    IF  ( .true. ) THEN
     IF  ( .false. ) THEN
!      DO ii = 1, inform%n_vec
       DO ii = 1, p_dim
         IF ( ii == 1 ) THEN
           i = 0
         ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
           i = - ( i + 1 )
         ELSE
           i = - i
         END IF
         CALL TREK_Hv( one, H, data%V( 1 : n, i ), zero, data%U,               &
                       out, control%error, symmetric = .TRUE. )
         DO jj = 1, ii
           IF ( jj == 1 ) THEN
             j = 0
           ELSE IF ( MOD( jj, 2 ) == 0 ) THEN
             j = - ( j + 1 )
           ELSE
             j = - j
           END IF
           P_calc( ii, jj ) = DOT_PRODUCT( data%U, data%V( 1 : n, j ) )
         END DO
       END DO
       DO ii = 1, p_dim
         IF ( ii + 2 <= p_dim ) THEN
           WRITE( out, "( ' P(', I3, ') = ', 3ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) ), &
                 ABS( data%P( 2, ii ) - P_calc( ii + 1, ii ) ), &
                 ABS( data%P( 3, ii ) - P_calc( ii + 2, ii ) )
         ELSE IF ( ii + 1 <= p_dim ) THEN
           WRITE( out, "( ' P(', I3, ') = ', 2ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) ), &
                 ABS( data%P( 2, ii ) - P_calc( ii + 1, ii ) )
         ELSE
           WRITE( out, "( ' P(', I3, ') = ', ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) )
         END IF
       END DO
     END IF

!  successful return

 900 CONTINUE
     RETURN

!  allocation error

 910 CONTINUE
     IF ( out > 0 .AND. control%print_level > 0 )                              &
       WRITE( control%out, "( A, '   **  Allocation error return ', I0,        &
      & ' from TREK ' )" ) prefix, inform%status
     RETURN

!  other error returns

 920 CONTINUE
     IF ( out > 0 .AND. control%print_level > 0 )                              &
       WRITE( control%out, "( A, '   **  Error return ', I0,                   &
      & ' from TREK ' )" ) prefix, inform%status
     RETURN

!  non-executable statements

2000 FORMAT( A, '    k d(K)   ||x||     radius      shift      error         f')
2010 FORMAT( A, 2I5, 4ES11.4, ES12.4 )

!  end of subroutine TREK_solve

     END SUBROUTINE TREK_solve

!-*-*-*-*-*-*-*-   T R E K _ R E S O L V E    S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE TREK_resolve( n, H, C, radius, X, data, control, inform )

!  Given an n x n symmetric matrix H, an n-vector c, and a radius > 0,
!  approximately re-solve the trust-region subproblem
!
!    min 1/2 x'Hx + c'x : ||x|| <= radius
!
!  using an extended Krylov subspace method and matrices generated by a
!  a previous solve involving H and c, but with a new (smaller) radius
!
!  The method uses the "backward" estended-Krylov subspace
!
!    K_2k = { c, H^{-1} c, H c, ..., H^{-k} c },
!
!  (see module EKS)
!
!  Input:
!   n - number of unknowns
!   H - symmetric coefficient matrix, H, from the quadratic term, 
!       in any symmetric format supported by the SMT type
!   radius - scalar trust-region radius > 0
!   control - parameters structure (see preamble)
!   inform - output structure (see preamble)
!   data - prvate internal workspace
!
!   Output:
!   X - solution vector x

!  dummy arguments

     INTEGER, INTENT( IN ) :: n
     TYPE ( SMT_type ), INTENT( IN ) :: H
     REAL ( KIND = rp_), INTENT( IN ), DIMENSION( n ) :: C
     REAL ( KIND = rp_), INTENT( IN ) :: radius
     REAL ( KIND = rp_), INTENT( OUT ), DIMENSION( n  ) :: X
     TYPE ( TREK_data_type ), INTENT( INOUT ) :: data
     TYPE ( TREK_control_type ), INTENT( IN ) :: control
     TYPE ( TREK_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER :: i, j, jj, j_max, k, km1, k2, k2m1, k2p1, mkp1, lapack_info, out
     REAL ( KIND = rp_) :: alpha, beta, gamma, delta, violation, shift
     REAL ( KIND = rp_) :: e11, e12, e22, error, error_j, s_norm, w_norm
     LOGICAL :: printi, printm, printd
     LOGICAL :: termination_test = .TRUE.

!  temporary debug variables - ultimately remove

     INTEGER :: ii
     INTEGER, PARAMETER :: p_dim = 10
     REAL ( KIND = rp_) :: P_calc( p_dim, p_dim )
     REAL ( KIND = rp_) :: R( n ), SOL( n )

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  check solve has been called before resolve

     IF ( data%k_exit <= 0 ) THEN
       inform%status = GALAHAD_error_call_order ; GO TO 920
     END IF

!  check input dimensions are consistent

     IF ( n /= H%n ) THEN
       inform%status = GALAHAD_error_restrictions ; GO TO 920
     END IF

!  check the new radius value is positive and smaller than the previous one

     IF ( radius <= zero .OR. radius >= data%last_radius ) THEN
       inform%status = GALAHAD_error_restrictions ; GO TO 920
     END IF

!  record the radiius, and set the potential next one

     data%last_radius = radius
     inform%radius = radius
     inform%next_radius = radius * data%control%reduction

!  record output values

     out = control%out
     printi = control%print_level > 0 .AND. out > 0
     printm = control%print_level > 2 .AND. out > 0
     printd = control%print_level > 5 .AND. out > 0

!  initialise the shift as that from the previous solve

     shift = data%last_shift

!  use the data from the previous solve to see if that subspace is sufficiently
!  rich to contain the solution with the new (reduced) radius

     k = data%k_exit
     inform%iter = k
     k2 = 2 * k
     k2m1 = k2 - 1
     k2p1 = k2 + 1

!  ....................-------------------
!  test system of order 2k for termination
!  ....................-------------------

!  compute the projected solution ( P + shift(j) I ) s(j) = delta_0 e_1
!  for j in [1,...,m]

     IF (  termination_test ) THEN   
       j_max = 1

!  exact shifts are required

       IF ( control%exact_shift ) THEN

!  save the lower triangle of P in Q

         data%Q( 1 : k2, 1 : k2 ) = zero
         DO i = 1, k2
           data%Q( i, i ) = data%P( 1, i )
         END DO
         DO i = 1, k2 - 1
           data%Q( i + 1, i ) = data%P( 2, i )
         END DO
         DO i = 1, k2 - 2
           data%Q( i + 2, i ) = data%P( 3, i )
         END DO

         IF ( printd ) THEN
           WRITE( out, "( ' P = ')" )
           DO i = 1, k2
             WRITE( out, "( 4( 2I4, ES12.4 ) )" )                              &
               ( i, j, data%Q( i, j ), j = 1, i ) 
           END DO
         END IF

!  compute the eigenvalues (in D) and eigenvalues (overwrite Q)

         CALL SYEV( 'V','L', k2, data%Q, data%is_max, data%D,                  &
                   data%WORK_syev, data%lwork_syev, lapack_info )
         IF ( lapack_info < 0 ) THEN
           inform%status = GALAHAD_error_lapack ; GO TO 920
         END IF

!  form c' = Q^T ||c|| e_1

         data%C( 1 : k2 ) = data%c_norm * data%Q( 1, 1 : k2 )

!  solve the diagonal trust-region problem
!    min 1/2 x'^T D x' + c'^T x' : ||x'|| <= radius

         data%control%trs_control%use_initial_multiplier = .TRUE.
         data%control%trs_control%initial_multiplier = shift
         CALL TRS_solve_diagonal( k2, radius, zero,                            &
                                  data%C, data%D, data%X,                      &
                                  data%control%trs_control,                    &
                                  inform%trs_inform )
         shift = inform%trs_inform%multiplier
         inform%multiplier = shift
         inform%obj = inform%trs_inform%obj
         inform%x_norm = inform%trs_inform%x_norm

!  recover x = Q x'

         data%S1( 1 : k2 ) = MATMUL( data%Q( : k2 , : k2 ), data%X( : k2 ) )

!  exact shifts are not required

       ELSE

!  form and factorize the shifted matrix P + shift(j) I

         data%P_shift( 1, : k2 ) = data%P( 1, : k2 ) + shift
         data%P_shift( 2 : 3, : k2 ) = data%P( 2 : 3, : k2 )
         CALL PBTRF( 'L', k2, 2, data%P_shift, ldp, lapack_info )
         IF ( lapack_info < 0 ) THEN
           inform%status = GALAHAD_error_lapack ; GO TO 920
         END IF

!  solve the shifted system to find the solution s(j)

         data%S2( 1, 1 ) = data%c_norm ; data%S2( 2 : k2, 1 ) = zero
         CALL PBTRS( 'L', k2, 2, 1, data%P_shift, ldp,                         &
                     data%S2, data%is_max, lapack_info )
         IF ( lapack_info < 0 ) THEN
           inform%status = GALAHAD_error_lapack ; GO TO 920
         END IF
         data%S1( 1 : k2 ) = data%S2( 1 : k2, 1 )
       END IF

!  record ||r_2k(t)||^2 = ( alpha_k s_2k-1(j))^2 + ( alpha_-k s_2k(j))^2

       delta = data%DELTA( k )
       gamma = data%P( 3, k2m1 )
       error = ABS( gamma * data%S1( k2m1 ) + delta * data%S1( k2 ) )

!  compute ||s|| and the violation

       s_norm = TWO_NORM( data%S1( 1 : k2 ) )
       violation = s_norm - radius
       IF ( printm ) WRITE( out,                                             &
         "(' k = ', I0, ' ||s||, radius, multiplier = ', 3ES11.4 )" )         &
            k, s_norm, radius, shift

!  if ||s|| > radius, update shift. First, solve L w = s ...

       IF ( .NOT. control%exact_shift ) THEN
         inform%obj                                                            &
           = half * ( data%c_norm * data%S1( 1 ) - shift * s_norm ** 2 )
!        IF ( s_norm > radius ) THEN
         IF ( .TRUE. ) THEN
           data%W( : k2 ) = data%S1( : k2 )
           CALL TBSV( 'L', 'N', 'N', k2, 2, data%P_shift, ldp,                 &
                      data%W, 1_ip_)

!  ... then use the norm of w to improve the shift

           w_norm = TWO_NORM( data%W( 1 : k2 ) )
           shift = shift + ( s_norm - radius ) * s_norm ** 2                   &
                     / ( RADIUS * w_norm ** 2 )
         END IF
       END IF

!  record ||s|| and the current shift

       inform%multiplier = shift
       inform%x_norm = s_norm

!  debug - compare predicted and actual error

       IF ( printd ) THEN
         SOL = zero
         DO ii = 1, k2
           IF ( ii == 1 ) THEN
             i = 0
           ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
             i = - i - 1
           ELSE
             i = - i
           END IF
           SOL = SOL + data%V( : n, i ) * data%S1( ii )
         END DO
         R = shift * SOL + C
         CALL TREK_Hv( one, H, SOL, one, R, out, control%error,                &
                       symmetric = .TRUE. )
         WRITE( out, "( ' ||r||, est = ', 2ES12.4 )" ) TWO_NORM( R ), error
       END IF

       IF ( printi ) THEN
         WRITE( out, 2000 ) prefix
         WRITE( out, 2010 ) prefix, k, k2,                                     &
           s_norm, radius, shift, error, inform%obj
       END IF
       error = error / data%c_norm

!  check for termination

       IF ( error < data%control%stop_residual .AND.                           &
            violation <= violation_max ) THEN
         inform%n_vec = k2

!  recover the solution

         X( : n ) = zero
         DO ii = 1, k2
           IF ( ii == 1 ) THEN
             i = 0
           ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
             i = - i - 1
           ELSE
             i = - i
           END IF
           X( : n ) = X( : n ) + data%V( : n, i ) * data%S1( ii )
         END DO

!  exit the main loop

         data%k_exit = k
         data%last_shift = shift
         inform%error = error
         inform%status = GALAHAD_ok
         RETURN
       END IF
     END IF

!  set delta_{-k-1} := ||u|| 

     delta = TWO_NORM( data%U )

!  set v_{-k-1} = u/delta_{-k-1}

     IF ( k < data%k_max ) THEN
       data%DELTA( - k - 1 ) = delta
       data%V( 1 : n, - k - 1 ) = data%U / delta
       data%n_v = data%n_v + 1
       IF ( data%n_v == n ) WRITE( out, "( ' Krylov space is full space' )" )
     END IF

!  ****************************************************************************
!
!  build and use the backward extended Krylov space
!
!    K_2k = { c, H^{-1} c, H c, ..., H^{-k} c }
!
!  Do this by building an orthogonal basis matrix
!
!    V_2k = [ v_0, v_{-1}, v_1, ... , v_{-k} ]
!
!  and the pentadiagonal projected matrix P_2k = V_2k' A V_2k, with 
!  columns p_i, whose subdiagonal columns are 
!
!    p_2k-1 = ( ( 1 - beta_{-k+1} delta_k - alpha_k delta_{-k} ) / beta_{k-1} )
!             (                     alpha_{k-1}                               )
!             (           - delta_{-k} delta_{k+1} /beta_{k-1}                )
!
!  and
!
!    p_2k = ( alpha_{-k} )
!           (  delta_k   )
!           (     0      )
!
!  where the alpha_i, beta_i and delta_i are defined below
!
!  For details, see
!    H. Al Daas & N. I. M. Gould
!    Extended-Krylov-subspace trust-region methods
!    Working note STFC-RAL 2025
!
!  ****************************************************************************

!  ------------------------------------------------------------
!  start of main forward iteration loop (in comments, iter = k)
!  ------------------------------------------------------------

     inform%status = GALAHAD_error_max_iterations

!  start from the previous iteration data%k_exit, as data up to then has
!  already been generated

!     DO k = data%k_exit, data%k_max
      DO k = data%k_exit + 1, data%k_max
       inform%iter = k
       km1 = k - 1
       mkp1 = - k + 1
       k2 = 2 * k
       k2m1 = k2 - 1
       k2p1 = k2 + 1

!  print iteration details if required

       IF ( printm ) WRITE( out, "( A, ' iteration ', I0 )" ) prefix, k

! for testing

       IF ( printd ) THEN 
         CALL TREK_Hv( one, H, data%V( 1 : n, mkp1 ), zero, data%U,            &
                       control%out, control%error, symmetric = .TRUE. )
         CALL TREK_Hv( one, H, data%V( 1 : n, k ), zero, data%U1,              &
                       control%out, control%error, symmetric = .TRUE. )
         e11 = DOT_PRODUCT( data%V( 1 : n, mkp1 ), data%U )
         e12 = DOT_PRODUCT( data%V( 1 : n, mkp1 ), data%U1 )
         e22 = DOT_PRODUCT( data%V( 1 : n, k ), data%U1 )
       END IF

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  segment involving the product with A
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  set u = H v_{-k}

       CALL TREK_Hv( one, H, data%V( 1 : n, - k ), zero, data%U,               &
                     out, control%error, symmetric = .TRUE. )

!  set alpha_{-k} = v_{-k}' u and u = u _ alpha_{-k} v_{-k}

       alpha = DOT_PRODUCT( data%V( 1 : n, - k ), data%U )
       data%ALPHA( - k ) = alpha
       data%U = data%U - alpha * data%V( 1 : n, - k )

!  set alpha {k-1} = v_{k-1}' u and u = u - alpha_{k-1} v_{k-1}

       alpha = DOT_PRODUCT( data%V( 1 : n, km1 ), data%U )
       data%ALPHA( km1 ) = alpha
       data%U = data%U - alpha * data%V( 1 : n, km1 )

!  orthogonalise wrt the remaining vectors

!      IF ( .true. ) THEN
       IF ( control%reorthogonalize ) THEN
         DO i = mkp1, km1 - 1
           beta = DOT_PRODUCT( data%V( 1 : n, i ), data%U )
           data%U = data%U - beta * data%V( 1 : n, i )
         END DO
       END IF

!   set delta_{k} = ||u|| and v_{k} = u/delta_{k}

       delta = TWO_NORM( data%U )
       gamma = - data%DELTA( - k ) * delta / data%BETA( km1 )
       data%DELTA( k ) = delta
       data%V( 1 : n, k ) = data%U / delta

!  save column 2k-1 of P

       IF ( k > 1 ) THEN
         data%P( 1, k2m1 ) = ( one - data%DELTA( - k ) * data%ALPHA( km1 )     &
                                   - data%BETA( mkp1 ) * data%DELTA( km1 ) )   &
                               / data%BETA( km1 )
       ELSE
         data%P( 1, k2m1 ) = ( one - data%DELTA( - k ) * data%ALPHA( km1 ) )   &
                               / data%BETA( km1 )
       END IF
       data%P( 2, k2m1 ) = data%ALPHA( km1 )
       data%P( 3, k2m1 ) = gamma

!  ....................---------------------
!  test system of order 2k-1 for termination
!  ....................---------------------

!  compute the projected solution ( P + shift(j) I ) s(j) = delta_0 e_1
!  for j in [1,...,m]

!      IF ( k > 1 .AND. termination_test ) THEN   
       IF ( k > 1 .AND. .FALSE. ) THEN   

!  form and factorize the shifted matrix P + shift(j) I = LL'

         data%P_shift( 1, : k2m1 ) = data%P( 1, : k2m1 ) + shift
         data%P_shift( 2 : 3, : k2m1 ) = data%P( 2 : 3, : k2m1 )
         CALL PBTRF( 'L', k2m1, 2, data%P_shift, ldp, lapack_info )
         IF ( lapack_info < 0 ) THEN
           inform%status = GALAHAD_error_lapack ; GO TO 920
         END IF

!  solve the shifted system to find the solution s(j)

         data%S2( 1, 1 ) = data%c_norm ; data%S2( 2 : k2m1, 1 ) = zero
         CALL PBTRS( 'L', k2m1, 2, 1, data%P_shift, ldp,                       &
                     data%S2, data%is_max, lapack_info )
         IF ( lapack_info < 0 ) THEN
           inform%status = GALAHAD_error_lapack ; GO TO 920
         END IF
         data%S1( : k2m1 ) = data%S2( : k2m1, 1 )

!  record ||r_2k(t)||^2 = ( alpha_k^2 + gamma^2 ) s_2k(j)^2 

         error = SQRT( data%ALPHA( km1 ) ** 2 + gamma ** 2 )                   &
                   * ABS( data%S1( k2m1 ) )

!  compute ||s(j)||

         s_norm = TWO_NORM( data%S1( 1 : k2m1 ) )
         WRITE( 6, "('k = ', I3, ' ||s||, radius = ', 2ES12.4 )" )             &
           k, s_norm, radius

!  if ||s(j)|| > radius_j, update shift(j). First, solve L w_j = s_j ...

         violation = s_norm - radius
         IF ( s_norm > radius ) THEN
           data%W( : k2m1 ) = data%S1( : k2m1 )
           CALL TBSV( 'L', 'N', 'N', k2m1, 2, data%P_shift, ldp,               &
                      data%W, 1_ip_)

!  ... then use the norm of w_j to improve the shift

           w_norm = TWO_NORM( data%W( 1 : k2m1 ) )
           shift = shift + ( s_norm - radius ) * s_norm ** 2                   &
                     / ( radius * w_norm ** 2 )
         END IF

!  debug - compare predicted and actual error

!        IF ( .TRUE. ) THEN
         IF ( .FALSE. ) THEN
           SOL = zero
           DO ii = 1, k2m1
             IF ( ii == 1 ) THEN
               i = 0
             ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
               i = - i - 1
             ELSE
               i = - i
             END IF
             SOL = SOL + data%V( : n, i ) * data%S1( ii )
           END DO
           R = shift * SOL + C
           CALL TREK_Hv( one, H, SOL, one, R, out, control%error,              &
                         symmetric = .TRUE. )
           WRITE( out, "( ' ||r||, est = ', 2ES12.4 )" )                       &
             TWO_NORM( R ), error_j
         END IF

         IF ( printi ) WRITE( out, 2010 ) prefix, k, k2m1,                     &
              s_norm, radius, shift, error, inform%obj
         error = error / data%c_norm

!  check for termination

!        IF ( error < data%control%stop_residual ) THEN
         IF ( error < data%control%stop_residual .AND.                         &
              violation <= violation_max ) THEN
           inform%n_vec = k2m1

!  recover the solution

           X( : n ) = zero
           DO ii = 1, k2m1
             IF ( ii == 1 ) THEN
               i = 0
             ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
               i = - i - 1
             ELSE
               i = - i
             END IF
             X( : n ) = X( : n ) + data%V( : n, i ) * data%S1( ii )
           END DO

!  exit the main loop

           inform%status = GALAHAD_ok ; EXIT
         END IF
       END IF

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!  segment involving the product with A inverse
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  set u = H^{−1} v_k

       data%U = data%V( 1 : n, k )
       CALL SLS_solve( H, data%U, data%sls_data, data%control%sls_control,     &
                       inform%sls_inform )
       IF ( inform%sls_inform%status < 0 ) THEN
         inform%status = GALAHAD_error_solve ; GO TO 920
       END IF

!  set beta_{-k} := v_{-k}' u and u = u - beta_{-k} v_{-k}
 
       beta = DOT_PRODUCT( data%V( 1 : n, - k ), data%U )
       data%BETA( - k ) = beta
       data%U = data%U - beta * data%V( 1 : n, - k )

!  set beta k = v_k' u and w := w - beta_k v_k

       beta = DOT_PRODUCT( data%V( 1 : n, k ), data%U )
       data%BETA( k ) = beta
       data%U = data%U - beta * data%V( 1 : n, k )

!  orthogonalise wrt the remaining vectors

!      IF ( .true. ) THEN
       IF ( control%reorthogonalize ) THEN
         DO i = mkp1 + 1, k - 1
           beta = DOT_PRODUCT( data%V( 1 : n, i ), data%U )
           data%U = data%U - beta * data%V( 1 : n, i )
         END DO
       END IF

!  test

       IF ( printd ) THEN
!      IF ( .true. ) THEN
         DO i = - k, k
           DO j = - k, i
             alpha =  DOT_PRODUCT( data%V( 1 : n, i ), data%V( 1 : n, j ) )
             IF ( ( i == j  .AND. ABS( alpha - one ) > ten ** ( -5 ) ) .OR.    &
                  ( i /= j  .AND. ABS( alpha ) > ten ** ( -5 ) ) )  &
               WRITE( out, "( ' i, j, v_ij ', 2I3, ES12.4 )" ) i, j, alpha
           END DO
         END DO
       END IF

!  save column 2k of P

       data%P( 1, k2 ) = data%ALPHA( - k )
       data%P( 2, k2 ) = delta
       data%P( 3, k2 ) = zero

       IF ( printd ) THEN
         IF ( ABS( data%P( 1, k2m1 ) - e11 ) > ten ** ( - 5 ) .OR.             &
              ABS( data%P( 2, k2m1 ) - e12 ) > ten ** ( - 5 ) .OR.             &
              ABS( data%P( 1, k2 ) - e22 ) > ten ** ( - 5 ) ) THEN
           WRITE( out, "( ' theoretical (P) and computed (C) Hessian:', /,     &
          &               ' P_11, C_11 = ', 2ES12.4, /,                        &
          &               ' P_21, C_21 = ', 2ES12.4, /,                        &
          &               ' P_22, C_22 = ', 2ES12.4 )" )                       &
            data%P( 1, k2m1 ), e11, data%P( 2, k2m1 ), e12,                    &
            data%P( 1, k2 ), e22
          END IF
        END IF

!  ....................-------------------
!  test system of order 2k for termination
!  ....................-------------------

!  compute the projected solution ( P + shift(j) I ) s(j) = delta_0 e_1
!  for j in [1,...,m]

       IF (  termination_test ) THEN   
         j_max = 1

!  exact shifts are required

         IF ( control%exact_shift ) THEN

!  save the lower triangle of P in Q

           data%Q( 1 : k2, 1 : k2 ) = zero
           DO i = 1, k2
             data%Q( i, i ) = data%P( 1, i )
           END DO
           DO i = 1, k2 - 1
             data%Q( i + 1, i ) = data%P( 2, i )
           END DO
           DO i = 1, k2 - 2
             data%Q( i + 2, i ) = data%P( 3, i )
           END DO

           IF ( printd ) THEN
             WRITE( out, "( ' P = ')" )
             DO i = 1, k2
               WRITE( out, "( 4( 2I4, ES12.4 ) )" )                            &
                 ( i, j, data%Q( i, j ), j = 1, i ) 
             END DO
           END IF

!  compute the eigenvalues (in D) and eigenvalues (overwrite Q)

          CALL SYEV( 'V','L', k2, data%Q, data%is_max, data%D,                 &
                     data%WORK_syev, data%lwork_syev, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF

!  form c' = Q^T ||c|| e_1

           data%C( 1 : k2 ) = data%c_norm * data%Q( 1, 1 : k2 )

!  solve the diagonal trust-region problem
!    min 1/2 x'^T D x' + c'^T x' : ||x'|| <= radius

           data%control%trs_control%use_initial_multiplier = .TRUE.
           data%control%trs_control%initial_multiplier = shift
           CALL TRS_solve_diagonal( k2, radius, zero,                          &
                                    data%C, data%D, data%X,                    &
                                    data%control%trs_control,                  &
                                    inform%trs_inform )
           shift = inform%trs_inform%multiplier
           inform%multiplier = shift
           inform%obj = inform%trs_inform%obj
           inform%x_norm = inform%trs_inform%x_norm

!  recover x = Q x'

           data%S1( 1 : k2 ) = MATMUL( data%Q( : k2 , : k2 ), data%X( : k2 ) )

!  exact shifts are not required

         ELSE

!  form and factorize the shifted matrix P + shift(j) I

           data%P_shift( 1, : k2 ) = data%P( 1, : k2 ) + shift
           data%P_shift( 2 : 3, : k2 ) = data%P( 2 : 3, : k2 )
           CALL PBTRF( 'L', k2, 2, data%P_shift, ldp, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF

!  solve the shifted system to find the solution s(j)

           data%S2( 1, 1 ) = data%c_norm ; data%S2( 2 : k2, 1 ) = zero
           CALL PBTRS( 'L', k2, 2, 1, data%P_shift, ldp,                       &
                       data%S2, data%is_max, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF
           data%S1( 1 : k2 ) = data%S2( 1 : k2, 1 )
         END IF

!  record ||r_2k(t)||^2 = ( alpha_k s_2k-1(j))^2 + ( alpha_-k s_2k(j))^2

         error = ABS( gamma * data%S1( k2m1 ) + delta * data%S1( k2 ) )

!  compute ||s|| and the violation

         s_norm = TWO_NORM( data%S1( 1 : k2 ) )
         violation = s_norm - radius
         IF ( printm ) WRITE( out,                                             &
           "(' k = ', I0, ' ||s||, radius, multiplier = ', 3ES11.4 )" )        &
              k, s_norm, radius, shift

!  if ||s|| > radius, update shift. First, solve L w = s ...

         IF ( .NOT. control%exact_shift ) THEN
           inform%obj                                                          &
             = half * ( data%c_norm * data%S1( 1 ) - shift * s_norm ** 2 )
!          IF ( s_norm > radius ) THEN
           IF ( .TRUE. ) THEN
             data%W( : k2 ) = data%S1( : k2 )
             CALL TBSV( 'L', 'N', 'N', k2, 2, data%P_shift, ldp,               &
                        data%W, 1_ip_)

!  ... then use the norm of w to improve the shift

             w_norm = TWO_NORM( data%W( 1 : k2 ) )
             shift = shift + ( s_norm - radius ) * s_norm ** 2                 &
                       / ( RADIUS * w_norm ** 2 )
           END IF
         END IF

!  record ||s|| and the current shift

         inform%multiplier = shift
         inform%x_norm = s_norm

!  debug - compare predicted and actual error

!        IF ( .TRUE. ) THEN
         IF ( .FALSE. ) THEN
           SOL = zero
           DO ii = 1, k2
             IF ( ii == 1 ) THEN
               i = 0
             ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
               i = - i - 1
             ELSE
               i = - i
             END IF
             SOL = SOL + data%V( : n, i ) * data%S1( ii )
           END DO
           R = shift * SOL + C
           CALL TREK_Hv( one, H, SOL, one, R, out, control%error,              &
                         symmetric = .TRUE. )
           WRITE( out, "( ' ||r||, est = ', 2ES12.4 )" ) TWO_NORM( R ), error
         END IF

         IF ( printi ) WRITE( out, 2010 ) prefix, k, k2,                       &
              s_norm, radius, shift, error, inform%obj
         error = error / data%c_norm

!  check for termination

         IF ( error < data%control%stop_residual .AND.                         &
              violation <= violation_max ) THEN
           inform%n_vec = k2

!  recover the solution

           X( : n ) = zero
           DO ii = 1, k2
             IF ( ii == 1 ) THEN
               i = 0
             ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
               i = - i - 1
             ELSE
               i = - i
             END IF
             X( : n ) = X( : n ) + data%V( : n, i ) * data%S1( ii )
           END DO

!  exit the main loop

           data%k_exit = k
           data%last_shift = shift
           inform%status = GALAHAD_ok ; EXIT
         END IF
       END IF

!  set delta_{-k-1} := ||u|| 

       delta = TWO_NORM( data%U )

!  set v_{-k-1} = u/delta_{-k-1}

       IF ( k < data%k_max ) THEN
         data%DELTA( - k - 1 ) = delta
         data%V( 1 : n, - k - 1 ) = data%U / delta
         data%n_v = data%n_v + 1
         IF ( data%n_v == n ) WRITE( out, "( ' Krylov space is full space' )" )
       END IF

!  ------------------------------------
!  end of main backward iteration loop
!  ------------------------------------

     END DO
     inform%error = error

!  debug - check the theoretical P against the computed one

     IF  ( printd ) THEN
!      DO ii = 1, inform%n_vec
       DO ii = 1, p_dim
         IF ( ii == 1 ) THEN
           i = 0
         ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
           i = - ( i + 1 )
         ELSE
           i = - i
         END IF
         CALL TREK_Hv( one, H, data%V( 1 : n, i ), zero, data%U,               &
                       out, control%error, symmetric = .TRUE. )
         DO jj = 1, ii
           IF ( jj == 1 ) THEN
             j = 0
           ELSE IF ( MOD( jj, 2 ) == 0 ) THEN
             j = - ( j + 1 )
           ELSE
             j = - j
           END IF
           P_calc( ii, jj ) = DOT_PRODUCT( data%U, data%V( 1 : n, j ) )
         END DO
       END DO
       DO ii = 1, p_dim
         IF ( ii + 2 <= p_dim ) THEN
           WRITE( out, "( ' P(', I3, ') = ', 3ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) ), &
                 ABS( data%P( 2, ii ) - P_calc( ii + 1, ii ) ), &
                 ABS( data%P( 3, ii ) - P_calc( ii + 2, ii ) )
         ELSE IF ( ii + 1 <= p_dim ) THEN
           WRITE( out, "( ' P(', I3, ') = ', 2ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) ), &
                 ABS( data%P( 2, ii ) - P_calc( ii + 1, ii ) )
         ELSE
           WRITE( out, "( ' P(', I3, ') = ', ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) )
         END IF
       END DO
     END IF

!  error returns

 920 CONTINUE
     RETURN

!  non-executable statements

2000 FORMAT( A, '    k d(K)   ||x||     radius      shift      error         f')
2010 FORMAT( A, 2I5, 4ES11.4, ES12.4 )

!  end of subroutine TREK_resolve

     END SUBROUTINE TREK_resolve

!-*-*-*-*-*-   T R E K _ S O L V E _ A L L   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE TREK_solve_all( n, m, H, C, RADIUS, X, data, control, inform )

!  Given an n x n symmetric matrix H, an n-vector c, and an array of m 
!  decreasing positive radii, approximately solve the trust-region subproblems
!
!    min 1/2 x'Hx + c'x : ||x|| <= radius_i for i = 1,...,m
!
!  using an extended Krylov subspace method. Here radius_1 = radius > 0,
!  and radius_i = radius * reduction^{j+i} for i = 2,...,m, 
!  0 < reduction < 1, and j+2 is the smallest integer for which the
!  trust-region constraint is "active". 
!
!  The method uses the "backward" estended-Krylov subspace
!
!    K_2k = { c, H^{-1} c, H c, ..., H^{-k} c },
!
!  (see module EKS)
!
!  Input:
!   n - number of unknowns
!   m - number of radii sought
!   H - symmetric coefficient matrix, H, from the quadratic term, 
!       in any symmetric format supported by the SMT type
!   C - vector c from linear term
!   RADIUS - vector of radii; only the the initial trust-region, radius_1
!       = RADIUS(1) > 0 need be set on entry, the remaining decreasing
!       set of radii_i will be chosen as a decreasing set of values 
!       for which the trust-region constraint is active
!   control - parameters structure (see preamble)
!   inform - output structure (see preamble)
!   data - prvate internal workspace
!
!   Output:
!   X - solution with X(:,i) for problem with radius_i

!  dummy arguments

     INTEGER, INTENT( IN ) :: n, m
     TYPE ( SMT_type ), INTENT( IN ) :: H
     REAL ( KIND = rp_), INTENT( IN ), DIMENSION( n ) :: C
     REAL ( KIND = rp_), INTENT( INOUT ), DIMENSION( m ) :: RADIUS
     REAL ( KIND = rp_), INTENT( OUT ), DIMENSION( n, m  ) :: X
     TYPE ( TREK_data_type ), INTENT( INOUT ) :: data
     TYPE ( TREK_control_type ), INTENT( IN ) :: control
     TYPE ( TREK_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER :: i, j, jj, j_max, k, km1, k2, k2m1, k2p1, mkp1, k_max, nb
     INTEGER :: is_max, jr, lapack_info, n_v, out, lwork_syev
!    INTEGER :: lda, lda, lwork
     REAL ( KIND = rp_) :: alpha, beta, gamma, delta, next_radius, newton_norm
     REAL ( KIND = rp_) :: e11, e12, e22, error, error_j, c_norm, s_norm, w_norm
     REAL ( KIND = rp_) :: violation, obj
     LOGICAL :: printi, printm, printd
     LOGICAL :: termination_test = .TRUE.
     CHARACTER ( LEN = 80 ) :: array_name

!  temporary debug variables - ultimately remove

     INTEGER :: ii
     INTEGER, PARAMETER :: p_dim = 10
     REAL ( KIND = rp_) :: P_calc( p_dim, p_dim )
     REAL ( KIND = rp_) :: R( n ), SOL( n )

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  check input dimensions are consistent

     IF ( n /= H%n .OR. m <= 0 ) THEN
       inform%status = GALAHAD_error_restrictions ; GO TO 920
     END IF

!  check initial radius value is positive

     IF ( RADIUS( 1 ) <= zero ) THEN
       inform%status = GALAHAD_error_restrictions ; GO TO 920
     END IF

!  record output values

     out = control%out
     printi = control%print_level > 0 .AND. out > 0
     printm = control%print_level > 2 .AND. out > 0
     printd = control%print_level > 5 .AND. out > 0

!  record the iteration limit

     data%control = control
     IF ( control%maxit > 0 ) THEN
       k_max = control%maxit
     ELSE
       k_max = m
     END IF
     is_max = 2 * k_max

!   provide space for, and initiate, the components of the projected solution

     IF ( .NOT. data%allocated_arrays ) THEN
       data%allocated_arrays = .TRUE.

       array_name = 'trek: data%V'
       CALL SPACE_resize_array(  1_ip_, n, - k_max, k_max, data%V,             &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

!  set the projected matrix P and solution s

       array_name = 'trek: data%P'
       CALL SPACE_resize_array( ldp, is_max, data%P,                           &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%P_shift'
       CALL SPACE_resize_array( ldp, is_max, data%P_shift,                     &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%S'
       CALL SPACE_resize_array( is_max, m, data%S,                             &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%S2'
       CALL SPACE_resize_array( is_max, 1, data%S2,                            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%W'
       CALL SPACE_resize_array( is_max, data%W,                                &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

!  set workspace arrays

       array_name = 'trek: data%SHIFT'
       CALL SPACE_resize_array( m, data%SHIFT,                                 &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%ALPHA'
       CALL SPACE_resize_array( - k_max, k_max, data%ALPHA,                    &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%BETA'
       CALL SPACE_resize_array( - k_max, k_max, data%BETA,                     &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%DELTA'
       CALL SPACE_resize_array( - k_max, k_max, data%DELTA,                    &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'trek: data%U'
       CALL SPACE_resize_array( n, data%U,                                     &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       IF ( control%exact_shift ) THEN
         array_name = 'trek: data%Q'
         CALL SPACE_resize_array( is_max, is_max, data%Q,                      &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%D'
         CALL SPACE_resize_array( is_max, data%D,                              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%C'
         CALL SPACE_resize_array( is_max, data%C,                              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%X'
         CALL SPACE_resize_array( is_max, data%X,                              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

!  discover the size of, and allocate, workspace needed for the eigensolver

         nb = LAENV( 1_ip_, 'DSYTRD', 'L', is_max, - 1_ip_, - 1_ip_, - 1_ip_ )
         lwork_syev = MAX( 1_ip_, 3 * is_max - 1, ( nb + 2 ) * is_max )

         array_name = 'trek: data%WORK_syev'
         CALL SPACE_resize_array( lwork_syev, data%WORK_syev,                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910
       END IF
     END IF

     IF ( printd ) THEN
       array_name = 'trek: data%U1'
       CALL SPACE_resize_array( n, data%U1,                                    &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910
     END IF

!   initialize, analyse and factorize H

     CALL SLS_initialize( control%solver, data%sls_data,                       &
                          data%control%sls_control,                            &
                          inform%sls_inform, check = .TRUE. )

     CALL SLS_analyse( H, data%sls_data, data%control%sls_control,             &
                       inform%sls_inform )
     IF ( inform%sls_inform%status < 0 ) THEN
       inform%status = GALAHAD_error_analysis ; GO TO 920
     END IF

     CALL SLS_factorize( H, data%sls_data, data%control%sls_control,           &
                         inform%sls_inform )
     IF ( inform%sls_inform%status < 0 ) THEN
       inform%status = GALAHAD_error_factorization ; GO TO 920
     END IF

!  ****************************************************************************
!
!  build and use the backward extended Krylov space
!
!    K_2k = { c, H^{-1} c, H c, ..., H^{-k} c }
!
!  Do this by building an orthogonal basis matrix
!
!    V_2k = [ v_0, v_{-1}, v_1, ... , v_{-k} ]
!
!  and the pentadiagonal projected matrix P_2k = V_2k' A V_2k, with 
!  columns p_i, whose subdiagonal columns are 
!
!    p_2k-1 = ( ( 1 - beta_{-k+1} delta_k - alpha_k delta_{-k} ) / beta_{k-1} )
!             (                     alpha_{k-1}                               )
!             (           - delta_{-k} delta_{k+1} /beta_{k-1}                )
!
!  and
!
!    p_2k = ( alpha_{-k} )
!           (  delta_k   )
!           (     0      )
!
!  where the alpha_i, beta_i and delta_i are defined below
!
!  For details, see
!    H. Al Daas & N. I. M. Gould
!    Extended-Krylov-subspace trust-region methods
!    Working note STFC-RAL 2025
!
!  ****************************************************************************

!  record the initial error

     c_norm = TWO_NORM( C( 1 : n ) )

!  normalize the input vector, delta_0 = ||c|| and v_0 = -c/delta_0

     data%DELTA( 0 ) = c_norm
     data%V( 1 : n, 0 ) = - C( 1 : n ) / c_norm

!  set u = H^{−1} v_0

     data%U = data%V( 1 : n, 0 )
     CALL SLS_solve( H, data%U, data%sls_data, data%control%sls_control,       &
                     inform%sls_inform )
     IF ( inform%sls_inform%status < 0 ) THEN
       inform%status = GALAHAD_error_solve ; GO TO 920
     END IF

!  compute ||x_newton|| = ||H^{-1} c||

     newton_norm = c_norm * TWO_NORM( data%U ) 

!  compute the radius_i, i = 2,..,m depending on whether ||x_newton|| <= radius

     next_radius = RADIUS( 1 )
     IF ( newton_norm <= RADIUS( 1 ) ) THEN
       X( : n, 1 ) = c_norm * data%U
       IF ( m == 1 ) THEN
         inform%status = GALAHAD_ok ; GO TO 900
       END IF
       jr = 2
       DO
         next_radius = next_radius * data%control%reduction
         IF ( newton_norm > next_radius ) EXIT
       END DO        
       DO j = 2, m
         RADIUS( j ) = next_radius
         next_radius = next_radius * data%control%reduction
       END DO
     ELSE
       jr = 1
       DO j = 2, m
         next_radius = next_radius * data%control%reduction
         RADIUS( j ) = next_radius
       END DO
     END IF

!  initialise the shifts as zero

     data%SHIFT = zero

!  set beta_0 = v_0' u and u = u - beta_0 v_0
 
     beta = DOT_PRODUCT( data%V( 1 : n, 0 ), data%U )
     data%BETA( 0 ) = beta
     data%U = data%U - beta * data%V( 1 : n, 0 )

!  set delta_{-1} = ||u||; v_{-1} = u / delta_{-1}

     data%DELTA( - 1 ) = TWO_NORM( data%U )
     data%V( 1 : n, - 1 ) = data%U / data%DELTA( - 1 )
     n_v = 2

!  ------------------------------------------------------------
!  start of main forward iteration loop (in comments, iter = k)
!  ------------------------------------------------------------

     IF ( printi ) WRITE( out, 2000 ) prefix
     inform%status = GALAHAD_error_max_iterations

     DO k = 1, k_max
       inform%iter = k
       km1 = k - 1
       mkp1 = - k + 1
       k2 = 2 * k
       k2m1 = k2 - 1
       k2p1 = k2 + 1

!  print iteration details if required

       IF ( printm ) WRITE( out, "( A, ' iteration ', I0 )" ) prefix, k

! for testing

       IF ( printd ) THEN 
         CALL TREK_Hv( one, H, data%V( 1 : n, mkp1 ), zero, data%U,            &
                       control%out, control%error, symmetric = .TRUE. )
         CALL TREK_Hv( one, H, data%V( 1 : n, k ), zero, data%U1,              &
                       control%out, control%error, symmetric = .TRUE. )
         e11 = DOT_PRODUCT( data%V( 1 : n, mkp1 ), data%U )
         e12 = DOT_PRODUCT( data%V( 1 : n, mkp1 ), data%U1 )
         e22 = DOT_PRODUCT( data%V( 1 : n, k ), data%U1 )
       END IF

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  segment involving the product with A
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  set u = H v_{-k}

       CALL TREK_Hv( one, H, data%V( 1 : n, - k ), zero, data%U,               &
                     out, control%error, symmetric = .TRUE. )

!  set alpha_{-k} = v_{-k}' u and u = u _ alpha_{-k} v_{-k}

       alpha = DOT_PRODUCT( data%V( 1 : n, - k ), data%U )
       data%ALPHA( - k ) = alpha
       data%U = data%U - alpha * data%V( 1 : n, - k )

!  set alpha {k-1} = v_{k-1}' u and u = u - alpha_{k-1} v_{k-1}

       alpha = DOT_PRODUCT( data%V( 1 : n, km1 ), data%U )
       data%ALPHA( km1 ) = alpha
       data%U = data%U - alpha * data%V( 1 : n, km1 )

!  orthogonalise wrt the remaining vectors

       IF ( control%reorthogonalize ) THEN
         DO i = mkp1, km1 - 1
           beta = DOT_PRODUCT( data%V( 1 : n, i ), data%U )
           data%U = data%U - beta * data%V( 1 : n, i )
         END DO
       END IF

!   set delta_{k} = ||u|| and v_{k} = u/delta_{k}

       delta = TWO_NORM( data%U )
       gamma = - data%DELTA( - k ) * delta / data%BETA( km1 )
       data%DELTA( k ) = delta
       data%V( 1 : n, k ) = data%U / delta

!  save column 2k-1 of P

       IF ( k > 1 ) THEN
         data%P( 1, k2m1 ) = ( one - data%DELTA( - k ) * data%ALPHA( km1 )     &
                                   - data%BETA( mkp1 ) * data%DELTA( km1 ) )   &
                               / data%BETA( km1 )
       ELSE
         data%P( 1, k2m1 ) = ( one - data%DELTA( - k ) * data%ALPHA( km1 ) )   &
                               / data%BETA( km1 )
       END IF
       data%P( 2, k2m1 ) = data%ALPHA( km1 )
       data%P( 3, k2m1 ) = gamma

!  ....................---------------------
!  test system of order 2k-1 for termination
!  ....................---------------------

!  compute the projected solution ( P + shift(j) I ) s(j) = delta_0 e_1
!  for j in [1,...,m]

       IF ( k > 1 .AND. .FALSE. ) THEN   
!      IF ( k > 1 .AND. termination_test ) THEN   
         error = zero
         violation = zero
         j_max = 1
         DO j = jr, m

!  form and factorize the shifted matrix P + shift(j) I = LL'

           data%P_shift( 1, : k2m1 ) = data%P( 1, : k2m1 ) + data%SHIFT( j )
           data%P_shift( 2 : 3, : k2m1 ) = data%P( 2 : 3, : k2m1 )
           CALL PBTRF( 'L', k2m1, 2, data%P_shift, ldp, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF

!  solve the shifted system to find the solution s(j)

           data%S2( 1, 1 ) = c_norm ; data%S2( 2 : k2m1, 1 ) = zero
           CALL PBTRS( 'L', k2m1, 2, 1, data%P_shift, ldp,                     &
                       data%S2, is_max,lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF
           data%S( : k2m1, j ) = data%S2( : k2m1, 1 )

!  record ||r_2k(t)||^2 = ( alpha_k^2 + gamma^2 ) s_2k(j)^2 

           error_j = SQRT( data%ALPHA( km1 ) ** 2 + gamma ** 2 )               &
                       * ABS( data%S( k2m1, j ) )
           IF ( error_j > error ) j_max = j

!  compute ||s(j)||

           s_norm = TWO_NORM( data%S( 1 : k2m1, j ) )
           WRITE( 6, "('k = ', I3, ' ||s||, radius = ', 2ES12.4 )" )           &
             k, s_norm, RADIUS( j )

!  if ||s(j)|| > radius_j, update shift(j). First, solve L w_j = s_j ...

           violation = MAX( violation,  s_norm - RADIUS( j ) )
           IF ( s_norm > RADIUS( j ) ) THEN
             data%W( : k2m1 ) = data%S( : k2m1, j )
             CALL TBSV( 'L', 'N', 'N', k2m1, 2, data%P_shift, ldp,             &
                        data%W, 1_ip_)

!  ... then use the norm of w_j to improve the shift

             w_norm = TWO_NORM( data%W( 1 : k2m1 ) )
             data%SHIFT( j ) = data%SHIFT( j ) +                               &
               ( s_norm - RADIUS( j ) ) * s_norm ** 2                          &
                 / ( RADIUS( j ) * w_norm ** 2 )
           END IF

!  debug - compare predicted and actual error

           if ( printd ) then
             SOL = zero
             DO ii = 1, k2m1
               IF ( ii == 1 ) THEN
                 i = 0
               ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
                 i = - i - 1
               ELSE
                 i = - i
               END IF
               SOL = SOL + data%V( : n, i ) * data%S( ii, j )
             END DO
             R = data%SHIFT( j ) * SOL + C
             CALL TREK_Hv( one, H, SOL, one, R, out, control%error,            &
                          symmetric = .TRUE. )
             WRITE( out, "( ' ||r(', I0, ')||, est = ', 2ES12.4 )" ) j,        &
               TWO_NORM( R ), error_j
           end if

           error = MAX( error, error_j )
           IF ( printi ) WRITE( out, 2010 ) prefix, k, k2m1,                   &
                s_norm, RADIUS( j ), data%SHIFT( j ), error_j, inform%obj
         END DO
         error = error / c_norm

!  check for termination

         IF ( error < data%control%stop_residual .AND.                         &
              violation <= violation_max ) THEN
           inform%n_vec = k2m1

!  recover the solution

           DO j = jr, m
             X( : n, j ) = zero
             DO ii = 1, k2m1
               IF ( ii == 1 ) THEN
                 i = 0
               ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
                 i = - i - 1
               ELSE
                 i = - i
               END IF
               X( : n, j ) = X( : n, j ) + data%V( : n, i ) * data%S( ii, j )
             END DO
           END DO

!  exit the main loop

           inform%status = GALAHAD_ok ; EXIT
         END IF
       END IF

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!  segment involving the product with A inverse
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  set u = H^{−1} v_k

       data%U = data%V( 1 : n, k )
       CALL SLS_solve( H, data%U, data%sls_data, data%control%sls_control,     &
                       inform%sls_inform )
       IF ( inform%sls_inform%status < 0 ) THEN
         inform%status = GALAHAD_error_solve ; GO TO 920
       END IF

!  set beta_{-k} := v_{-k}' u and u = u - beta_{-k} v_{-k}
 
       beta = DOT_PRODUCT( data%V( 1 : n, - k ), data%U )
       data%BETA( - k ) = beta
       data%U = data%U - beta * data%V( 1 : n, - k )

!  set beta k = v_k' u and w := w - beta_k v_k

       beta = DOT_PRODUCT( data%V( 1 : n, k ), data%U )
       data%BETA( k ) = beta
       data%U = data%U - beta * data%V( 1 : n, k )

!  orthogonalise wrt the remaining vectors

!      IF ( .true. ) THEN
       IF ( control%reorthogonalize ) THEN
         DO i = mkp1 + 1, k - 1
           beta = DOT_PRODUCT( data%V( 1 : n, i ), data%U )
           data%U = data%U - beta * data%V( 1 : n, i )
         END DO
       END IF

!  test

       IF ( printd ) THEN
!      IF ( .true. ) THEN
         DO i = - k, k
           DO j = - k, i
             alpha =  DOT_PRODUCT( data%V( 1 : n, i ), data%V( 1 : n, j ) )
             IF ( ( i == j  .AND. ABS( alpha - one ) > ten ** ( -5 ) ) .OR.    &
                  ( i /= j  .AND. ABS( alpha ) > ten ** ( -5 ) ) )  &
               WRITE( out, "( ' i, j, v_ij ', 2I3, ES12.4 )" ) i, j, alpha
           END DO
         END DO
       END IF

!  save column 2k of P

       data%P( 1, k2 ) = data%ALPHA( - k )
       data%P( 2, k2 ) = delta
       data%P( 3, k2 ) = zero

       IF ( printd ) THEN
         IF ( ABS( data%P( 1, k2m1 ) - e11 ) > ten ** ( - 5 ) .OR.             &
              ABS( data%P( 2, k2m1 ) - e12 ) > ten ** ( - 5 ) .OR.             &
              ABS( data%P( 1, k2 ) - e22 ) > ten ** ( - 5 ) ) THEN
           WRITE( out, "( ' theoretical (P) and computed (C) Hessian:', /,     &
          &               ' P_11, C_11 = ', 2ES12.4, /,                        &
          &               ' P_21, C_21 = ', 2ES12.4, /,                        &
          &               ' P_22, C_22 = ', 2ES12.4 )" )                       &
            data%P( 1, k2m1 ), e11, data%P( 2, k2m1 ), e12,                    &
            data%P( 1, k2 ), e22
          END IF
        END IF

!  save the lower triangle of P in Q if exact shifts are required

        IF ( control%exact_shift ) THEN
          data%Q( 1 : k2, 1 : k2 ) = zero
          DO i = 1, k2
            data%Q( i, i ) = data%P( 1, i )
          END DO
          DO i = 1, k2 - 1
            data%Q( i + 1, i ) = data%P( 2, i )
          END DO
          DO i = 1, k2 - 2
            data%Q( i + 2, i ) = data%P( 3, i )
          END DO

          IF ( printd ) THEN
            WRITE( out, "( ' P = ')" )
            DO i = 1, k2
              WRITE( out, "( 4( 2I4, ES12.4 ) )" )                             &
                ( i, j, data%Q( i, j ), j = 1, i ) 
            END DO
          END IF

!  compute the eigenvalues (in D) and eigenvalues (overwrite Q)

          CALL SYEV( 'V','L', k2, data%Q, is_max, data%D, data%WORK_syev,      &
                     lwork_syev, lapack_info )
          IF ( lapack_info < 0 ) THEN
            inform%status = GALAHAD_error_lapack ; GO TO 920
          END IF

!  form c' = Q^T ||c|| e_1

          data%C( 1 : k2 ) = c_norm * data%Q( 1, 1 : k2 )
        END IF
!  ....................-------------------
!  test system of order 2k for termination
!  ....................-------------------

!  compute the projected solution ( P + shift(j) I ) s(j) = delta_0 e_1
!  for j in [1,...,m]

       IF (  termination_test ) THEN   
         error = zero
         violation = zero
         j_max = 1
         DO j = jr, m

!  exact shifts are required

           IF ( control%exact_shift ) THEN

!  solve the diagonal trust-region problem
!    min 1/2 x'^T D x' + c'^T x' : ||x'|| <= radius

             data%control%trs_control%use_initial_multiplier = .TRUE.
             data%control%trs_control%initial_multiplier = data%SHIFT( j )
             CALL TRS_solve_diagonal( k2, RADIUS( j ), zero,                   &
                                      data%C, data%D, data%X,                  &
                                      data%control%trs_control,                &
                                      inform%trs_inform )
             data%SHIFT( j ) = inform%trs_inform%multiplier
             obj = inform%trs_inform%obj

!  recover x = Q x'

             data%S( 1 : k2, j )                                               &
               = MATMUL( data%Q( : k2 , : k2 ), data%X( : k2 ) )

!  exact shifts are not required

           ELSE

!  form and factorize the shifted matrix P + shift(j) I

             data%P_shift( 1, : k2 ) = data%P( 1, : k2 ) + data%SHIFT( j )
             data%P_shift( 2 : 3, : k2 ) = data%P( 2 : 3, : k2 )
             CALL PBTRF( 'L', k2, 2, data%P_shift, ldp, lapack_info )
             IF ( lapack_info < 0 ) THEN
               inform%status = GALAHAD_error_lapack ; GO TO 920
             END IF

!  solve the shifted system to find the solution s(j)

             data%S2( 1, 1 ) = c_norm ; data%S2( 2 : k2, 1 ) = zero
             CALL PBTRS( 'L', k2, 2, 1, data%P_shift, ldp,                     &
                         data%S2, is_max, lapack_info )
             IF ( lapack_info < 0 ) THEN
               inform%status = GALAHAD_error_lapack ; GO TO 920
             END IF
             data%S( 1 : k2, j ) = data%S2( 1 : k2, 1 )
           END IF

!  record ||r_2k(t)||^2 = ( alpha_k s_2k-1(j))^2 + ( alpha_-k s_2k(j))^2

           error_j = ABS( gamma * data%S( k2m1, j ) + delta * data%S( k2, j ) )
           IF ( error_j > error ) j_max = j

!  compute ||s(j)||

           s_norm = TWO_NORM( data%S( 1 : k2, j ) )
           IF ( .NOT. control%exact_shift ) obj = half *                       &
             ( data%c_norm * data%S( 1, j ) - data%SHIFT( j ) * s_norm ** 2 )
           IF ( printm ) WRITE( out, "( ' k = ', I0,                           &
          & ' ||s||, radius, multiplier ', I0, ' = ', 3ES11.4 )" )             &
             k, j, s_norm, RADIUS( j ), data%SHIFT( j )
           violation = MAX( violation,  s_norm - RADIUS( j ) )

!  if ||s(j)|| > radius_j, update shift(j). First, solve L w_j = s_j ...

           IF ( .NOT. control%exact_shift ) THEN
!            IF ( s_norm > RADIUS( j ) ) THEN
             IF ( .TRUE. ) THEN
               data%W( : k2 ) = data%S( : k2, j )
               CALL TBSV( 'L', 'N', 'N', k2, 2, data%P_shift, ldp,             &
                          data%W, 1_ip_)

!  ... then use the norm of w_j to improve the shift

               w_norm = TWO_NORM( data%W( 1 : k2 ) )
               data%SHIFT( j ) = data%SHIFT( j ) +                             &
                 ( s_norm - RADIUS( j ) ) * s_norm ** 2                        &
                   / ( RADIUS( j ) * w_norm ** 2 )
             END IF
           END IF

!  debug - compare predicted and actual error

           IF ( printd ) THEN
             SOL = zero
             DO ii = 1, k2
               IF ( ii == 1 ) THEN
                 i = 0
               ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
                 i = - i - 1
               ELSE
                 i = - i
               END IF
               SOL = SOL + data%V( : n, i ) * data%S( ii, j )
             END DO
             R = data%SHIFT( j ) * SOL + C
             CALL TREK_Hv( one, H, SOL, one, R, out, control%error,            &
                           symmetric = .TRUE. )
             WRITE( out, "( ' ||r(', I0, ')||, est = ', 2ES12.4 )" ) j,        &
               TWO_NORM( R ), error_j
           END IF

           IF ( printi ) WRITE( out, 2010 ) prefix, k, k2,                     &
                s_norm, RADIUS( j ), data%SHIFT( j ), error_j, obj
           error = MAX( error, error_j )
!          IF ( error > data%control%stop_residual * c_norm ) EXIT
         END DO
         error = error / c_norm

!  check for termination

         IF ( error < data%control%stop_residual .AND.                         &
              violation <= violation_max ) THEN
           inform%n_vec = k2

!  recover the solution

           DO j = jr, m
             X( : n, j ) = zero
             DO ii = 1, k2
               IF ( ii == 1 ) THEN
                 i = 0
               ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
                 i = - i - 1
               ELSE
                 i = - i
               END IF
               X( : n, j ) = X( : n, j ) + data%V( : n, i ) * data%S( ii, j )
             END DO
           END DO

!  exit the main loop

           inform%status = GALAHAD_ok ; EXIT
         END IF
       END IF

!  set delta_{-k-1} := ||u|| 

       delta = TWO_NORM( data%U )

!  set v_{-k-1} = u/delta_{-k-1}

       IF ( k < k_max ) THEN
         data%DELTA( - k - 1 ) = delta
         data%V( 1 : n, - k - 1 ) = data%U / delta
         n_v = n_v + 1
         IF ( n_v == n ) WRITE( out, "( ' Krylov space is full space' )" )
       END IF

!  ------------------------------------
!  end of main backward iteration loop
!  ------------------------------------

     END DO
     inform%error = error

!  debug - check the theoretical P against the computed one

!    IF  ( .true. ) THEN
     IF  ( .false. ) THEN
!      DO ii = 1, inform%n_vec
       DO ii = 1, p_dim
         IF ( ii == 1 ) THEN
           i = 0
         ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
           i = - ( i + 1 )
         ELSE
           i = - i
         END IF
         CALL TREK_Hv( one, H, data%V( 1 : n, i ), zero, data%U,               &
                       out, control%error, symmetric = .TRUE. )
         DO jj = 1, ii
           IF ( jj == 1 ) THEN
             j = 0
           ELSE IF ( MOD( jj, 2 ) == 0 ) THEN
             j = - ( j + 1 )
           ELSE
             j = - j
           END IF
           P_calc( ii, jj ) = DOT_PRODUCT( data%U, data%V( 1 : n, j ) )
         END DO
       END DO
       DO ii = 1, p_dim
         IF ( ii + 2 <= p_dim ) THEN
           WRITE( out, "( ' P(', I3, ') = ', 3ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) ), &
                 ABS( data%P( 2, ii ) - P_calc( ii + 1, ii ) ), &
                 ABS( data%P( 3, ii ) - P_calc( ii + 2, ii ) )
         ELSE IF ( ii + 1 <= p_dim ) THEN
           WRITE( out, "( ' P(', I3, ') = ', 2ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) ), &
                 ABS( data%P( 2, ii ) - P_calc( ii + 1, ii ) )
         ELSE
           WRITE( out, "( ' P(', I3, ') = ', ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) )
         END IF
       END DO
     END IF

!  successful return

 900 CONTINUE
     RETURN

!  allocation error

 910 CONTINUE
     IF ( out > 0 .AND. control%print_level > 0 )                              &
       WRITE( control%out, "( A, '   **  Allocation error return ', I0,        &
      & ' from TREK ' )" ) prefix, inform%status
     RETURN

!  other error returns

 920 CONTINUE
     IF ( out > 0 .AND. control%print_level > 0 )                              &
       WRITE( control%out, "( A, '   **  Error return ', I0,                   &
      & ' from TREK ' )" ) prefix, inform%status
     RETURN

!  non-executable statements

2000 FORMAT( A, '    k d(K)   ||x||     radius      shift      error         f')
2010 FORMAT( A, 2I5, 4ES11.4, ES12.4 )

!  end of subroutine TREK_solve_all

     END SUBROUTINE TREK_solve_all

! -*-*-  G A L A H A D -  T R E K _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE TREK_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine TREK_initialize
!   control see Subroutine TREK_initialize
!   inform  see Subroutine TREK_find

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( TREK_data_type ), INTENT( INOUT ) :: data
      TYPE ( TREK_control_type ), INTENT( IN ) :: control
      TYPE ( TREK_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

      data%allocated_arrays = .FALSE.

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      inform%status = inform%SLS_inform%status
      IF ( inform%SLS_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'trek: data%SLS'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all remaing allocated arrays

      array_name = 'trek: data%P'
      CALL SPACE_dealloc_array( data%P,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%P_shift'
      CALL SPACE_dealloc_array( data%P_shift,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%V'
      CALL SPACE_dealloc_array( data%V,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%S1'
      CALL SPACE_dealloc_array( data%S1,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%S2'
      CALL SPACE_dealloc_array( data%S2,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%Q'
      CALL SPACE_dealloc_array( data%Q,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%D'
      CALL SPACE_dealloc_array( data%D,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%X'
      CALL SPACE_dealloc_array( data%X,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%WORK_syev'
      CALL SPACE_dealloc_array( data%WORK_syev,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%SHIFT'
      CALL SPACE_dealloc_array( data%SHIFT,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%ALPHA'
      CALL SPACE_dealloc_array( data%ALPHA,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%BETA'
      CALL SPACE_dealloc_array( data%BETA,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%DELTA'
      CALL SPACE_dealloc_array( data%DELTA,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%U'
      CALL SPACE_dealloc_array( data%U,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%U1'
      CALL SPACE_dealloc_array( data%U1,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine TREK_terminate

      END SUBROUTINE TREK_terminate

! -  G A L A H A D -  T R E K _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE TREK_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( TREK_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( TREK_control_type ), INTENT( IN ) :: control
     TYPE ( TREK_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

!    CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL TREK_terminate( data%trek_data, control, inform )

!  deallocate any internal problem arrays

!    array_name = 'trek: data%prob%X'
!    CALL SPACE_dealloc_array( data%prob%X,                                    &
!       inform%status, inform%alloc_status, array_name = array_name,           &
!       bad_alloc = inform%bad_alloc, out = control%error )
!    IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine TREK_full_terminate

     END SUBROUTINE TREK_full_terminate

!  end of module TREK_precision

   END MODULE GALAHAD_TREK_precision



