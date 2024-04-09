! THIS VERSION: GALAHAD 5.0 - 2024-03-17 AT 09:00 GMT.

#include "hsl_subset.h"

! COPYRIGHT (c) 2006 Council for the Central Laboratory
!                    of the Research Councils
! This package may be copied and used in any application, provided no
! changes are made to these or any other lines.
! Original date 21 February 2006. Version 1.0.0.
! 6 March 2007 Version 1.1.0. Argument stat made non-optional

MODULE hsl_zd11_real

     USE HSL_KINDS_real, ONLY: ip_, rp_
     PRIVATE :: ip_, rp_

!  ==========================
!  Sparse matrix derived type
!  ==========================

  TYPE, PUBLIC :: ZD11_type
    INTEGER ( KIND = ip_ ) :: m, n, ne
    CHARACTER, ALLOCATABLE, DIMENSION(:) :: id, type
    INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION(:) :: row, col, ptr
    REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION(:) :: val
  END TYPE

CONTAINS

   SUBROUTINE ZD11_put(array,string,stat)
     CHARACTER, allocatable :: array(:)
     CHARACTER(*), intent(in) ::  string
     INTEGER ( KIND = ip_ ), intent(OUT) ::  stat

!  Copy a string into an array.

     INTEGER ( KIND = ip_ ) :: i,l

     l = len_trim(string)
     if (allocated(array)) then
        deallocate(array,stat=stat)
        if (stat/=0) return
     end if
     allocate(array(l),stat=stat)
     if (stat/=0) return
     do i = 1, l
       array(i) = string(i:i)
     end do

   END SUBROUTINE ZD11_put

   FUNCTION ZD11_get(array)
     CHARACTER, intent(in):: array(:)
     CHARACTER(size(array)) ::  ZD11_get

! Give the value of array to string.

     INTEGER ( KIND = ip_ ) :: i
     do i = 1, size(array)
        ZD11_get(i:i) = array(i)
     end do

   END FUNCTION ZD11_get

END MODULE hsl_zd11_real
