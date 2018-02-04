! THIS VERSION: GALAHAD 2.4 - 18/06/2009 AT 16:00 GMT.

!-*-*-*-*-*-*-*-*  G A L A H A D _ S M T   M O D U L E  *-*-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released pre GALAHAD Version 1.0. December 1st 1997
!   update released with GALAHAD Version 2.0. February 16th 2005
!   replaced by the current use of the functionally-equivalent 
!    HSL package ZD11 in GALAHAD 2.4. June 18th, 2009

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SMT_double

     USE HSL_ZD11_double, SMT_type => ZD11_type, SMT_put => ZD11_put,          &
                          SMT_get => ZD11_get
!    USE HSL_ZD11_double, SMT_type => ZD11_type

!  ==========================
!  sparse matrix derived type
!  ==========================

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SMT_type, SMT_put, SMT_get

!CONTAINS

!   SUBROUTINE SMT_put(array,string,stat)
!     CHARACTER, allocatable :: array(:)
!     CHARACTER(*), intent(in) ::  string
!     INTEGER, intent(OUT) ::  stat

!     INTEGER :: i,l

!     l = len_trim(string)
!     if (allocated(array)) then
!        deallocate(array,stat=stat)
!        if (stat/=0) return
!     end if
!     allocate(array(l),stat=stat)
!     if (stat/=0) return
!     do i = 1, l
!       array(i) = string(i:i)
!     end do

!   END SUBROUTINE SMT_put

!   FUNCTION SMT_get(array)
!     CHARACTER, intent(in):: array(:)
!     CHARACTER(size(array)) ::  SMT_get
!! Give the value of array to string.

!     integer :: i
!     do i = 1, size(array)
!        SMT_get(i:i) = array(i)
!     end do

!   END FUNCTION SMT_get

!!  End of module SMT

   END MODULE GALAHAD_SMT_double


