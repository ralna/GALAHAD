! THIS VERSION: 06/06/2011 AT 09:45:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 6 4    M O D U L E  -*-*-*-

MODULE hsl_mc64_double

   USE hsl_zd11_double

   IMPLICIT NONE
   INTEGER, PARAMETER, PRIVATE :: wp = kind(0.0d0)

   TYPE mc64_control
!     real(wp) :: relax     ! Relaxes matching
      integer :: lp     ! Unit for error messages
      integer :: wp     ! Unit for warning messages
      integer :: sp     ! Unit for statistical output
      integer :: ldiag  ! Controls level of diagnostic output
      integer :: checking  ! Control for checking input

   END TYPE mc64_control

   TYPE mc64_info
      integer :: flag   ! Flags success or failure case
      integer :: more    ! More information on failure
      integer :: strucrank ! Structural rank
      integer :: stat    ! STAT value after allocate failure
   END TYPE mc64_info

CONTAINS

   SUBROUTINE mc64_initialize(control)
      type(mc64_control), intent(out) :: control
      IF ( control%lp >= 0 ) WRITE( control%lp,                              &
      "( ' We regret that the solution options that you have ', /,             &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MC64_initialize with its  HSL namesake ', /,              &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )
    END SUBROUTINE mc64_initialize

   SUBROUTINE mc64_matching(job,matrix,control,info,perm,scale)
      USE GALAHAD_SYMBOLS
      integer, intent(in) :: job ! Control parameter for algorithm choice
      type(zd11_type), intent(in) :: matrix
      type(mc64_control), intent(in) :: control
      type(mc64_info), intent(out) :: info
      integer, intent(out) :: perm(matrix%m + matrix%n)
      real(wp), optional, intent(out) :: scale(matrix%m + matrix%n)
      IF ( control%lp >= 0 ) WRITE( control%lp,                              &
      "( ' We regret that the solution options that you have ', /,             &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MC64_matching with its HSL namesake ', /,              &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )
       info%flag = GALAHAD_unavailable_option
   END SUBROUTINE mc64_matching

END MODULE hsl_mc64_double
