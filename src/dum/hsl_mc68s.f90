! THIS VERSION: 20/01/2011 AT 12:15:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 6 8    M O D U L E  -*-*-*-

    MODULE hsl_mc68_single

      USE hsl_zb01_integer

      IMPLICIT NONE
      PRIVATE

      INTEGER, PARAMETER :: myreal_mc68 = kind(1.0)
      INTEGER, PARAMETER :: myint = kind(1)
      INTEGER, PARAMETER :: long = selected_int_kind(18)

      TYPE, PUBLIC :: mc68_control
        INTEGER :: lp = 6 ! stream number for error messages
        INTEGER :: wp = 6 ! stream number for warning messages
        INTEGER :: mp = 6 ! stream number for diagnostic messages
        INTEGER :: nemin = 1 ! stream number for diagnostic messages
        INTEGER :: print_level = 0 ! amount of informational output required
        INTEGER :: row_full_thresh = 100 ! percentage threshold for full row
        INTEGER :: row_search = 10 ! Number of rows searched 4 pivot with ord=6
      END TYPE mc68_control

      TYPE, PUBLIC :: mc68_info
        INTEGER :: flag = 0 ! error/warning flag
        INTEGER :: iostat = 0 ! holds Fortran iostat parameter
        INTEGER :: stat = 0 ! holds Fortran stat parameter
        INTEGER :: out_range = 0 ! holds number of out of range entries
        INTEGER :: duplicate = 0 ! holds number of duplicate entries
        INTEGER :: n_compressions = 0 ! holds number of compressions in order
        INTEGER :: n_zero_eigs = -1 ! holds the number of zero eigs from ma47
        INTEGER :: l_workspace = 0 ! holds length of workspace iw used in
        INTEGER :: zb01_info = 0 ! holds flag from zb01_expand1 call
        INTEGER :: n_dense_rows = 0 ! holds number of dense rows from amdd
      END TYPE mc68_info

      INTERFACE mc68_order
        MODULE PROCEDURE mc68_order_single
      END INTERFACE

      PUBLIC mc68_order

    CONTAINS

      SUBROUTINE mc68_order_single(ord,n,ptr,row,perm,control,info,            &
                                   min_l_workspace)
        USE GALAHAD_SYMBOLS
        INTEGER, INTENT (IN) :: ord
        INTEGER, INTENT (IN) :: n
        INTEGER, INTENT (IN) :: ptr(n+1)
        INTEGER, INTENT (IN) :: row(:)
        INTEGER (myint) :: perm(n)
        TYPE (mc68_control), INTENT (IN) :: control
        TYPE (mc68_info) :: info
        INTEGER, INTENT (IN), OPTIONAL :: min_l_workspace

        IF ( control%lp >= 0 ) WRITE( control%lp,                              &
      "( ' We regret that the solution options that you have ', /,             &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MC68_order HSL namesake ', /,                             &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

        info%flag = GALAHAD_unavailable_option
        info%stat = 0

      END SUBROUTINE mc68_order_single

    END MODULE hsl_mc68_single
