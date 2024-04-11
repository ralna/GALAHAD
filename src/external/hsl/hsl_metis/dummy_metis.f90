#include "hsl_subset.h"

! Dummy routine that is called if MeTiS is not linked.
subroutine hsl_metis(n,iptr,irn,metftn,metopt,invprm,perm)
    implicit none
    integer, intent(in) :: n
    integer, intent(in) :: iptr(n+1)
    integer, intent(in) :: irn(*)
    integer, intent(in) :: metftn
    integer, intent(in) :: metopt(8)
    integer, intent(out) :: invprm(n)
    integer, intent(out) :: perm(n)

    ! Dummy ordering
    perm(1) = -1
end subroutine hsl_metis

subroutine hsl_metis_setopt(metopt)
    integer, intent(out) :: metopt(8)
    metopt = 0
end subroutine hsl_metis_setopt
