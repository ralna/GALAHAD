! THIS VERSION: GALAHAD 5.0 - 2024-03-17 AT 11:25 GMT.

#include "hsl_subset.h"

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 6 8    M O D U L E  -*-*-*-

    MODULE hsl_mc68_integer

      use hsl_kinds, only: ip_, long_
      use hsl_zb01_integer
#ifdef INTEGER_64
      USE GALAHAD_SYMBOLS_64, ONLY: GALAHAD_unavailable_option
#else
      USE GALAHAD_SYMBOLS, ONLY: GALAHAD_unavailable_option
#endif

      IMPLICIT NONE
      PRIVATE

      TYPE, PUBLIC :: mc68_control
        INTEGER ( KIND = ip_ ) :: lp = 6 ! stream number for error messages
        INTEGER ( KIND = ip_ ) :: wp = 6 ! stream number for warning messages
        INTEGER ( KIND = ip_ ) :: mp = 6 ! stream number for diagnostic messages
        INTEGER ( KIND = ip_ ) :: nemin = 1 ! stream number for diagnostic msgs
        INTEGER ( KIND = ip_ ) :: print_level = 0 ! amount of info output
        INTEGER ( KIND = ip_ ) :: row_full_thresh = 100 ! % thresh for full row
        INTEGER ( KIND = ip_ ) :: row_search = 10 ! # of rows searched for pivot
      END TYPE mc68_control

      TYPE, PUBLIC :: mc68_info
        INTEGER ( KIND = ip_ ) :: flag = 0 ! error/warning flag
        INTEGER ( KIND = ip_ ) :: iostat = 0 ! holds Fortran iostat parameter
        INTEGER ( KIND = ip_ ) :: stat = 0 ! holds Fortran stat parameter
        INTEGER ( KIND = ip_ ) :: out_range = 0 ! # out of range entries
        INTEGER ( KIND = ip_ ) :: duplicate = 0 ! # duplicate entries
        INTEGER ( KIND = ip_ ) :: n_compressions = 0 ! # compressions in order
        INTEGER ( KIND = ip_ ) :: n_zero_eigs = -1 ! # zero eigs from ma47
        INTEGER ( KIND = long_ ) :: l_workspace = 0 ! len workspce
        INTEGER ( KIND = ip_ ) :: zb01_info = 0 ! holds flag from zb01_expand1
        INTEGER ( KIND = ip_ ) :: n_dense_rows = 0 ! # dense rows from amdd
      END TYPE mc68_info

      INTERFACE mc68_order
        MODULE PROCEDURE mc68_order_integer
      END INTERFACE

      PUBLIC mc68_order

    CONTAINS

      SUBROUTINE mc68_order_integer( ord, n, ptr, row, perm, control, info,    &
                                     min_l_workspace )
        INTEGER ( KIND = ip_ ), INTENT (IN) :: ord
        INTEGER ( KIND = ip_ ), INTENT (IN) :: n
        INTEGER ( KIND = ip_ ), INTENT (IN) :: ptr( n + 1 )
        INTEGER ( KIND = ip_ ), INTENT (IN) :: row( : )
        INTEGER ( KIND = ip_ ), INTENT (OUT) :: perm( n )
        TYPE (mc68_control), INTENT (IN) :: control
        TYPE (mc68_info) :: info
        INTEGER ( KIND = ip_ ), INTENT (IN), OPTIONAL :: min_l_workspace
        IF ( control%lp >= 0 ) WRITE( control%lp,                              &
      "( ' We regret that the solution options that you have ', /,             &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MC68_order HSL namesake  and dependencies. See ', /,      &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

        info%flag = GALAHAD_unavailable_option
        info%stat = 0

      END SUBROUTINE mc68_order_integer

    END MODULE hsl_mc68_integer
