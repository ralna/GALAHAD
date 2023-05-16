! Routine that is called if MeTiS 4 is linked.
subroutine galahad_metis(n,iptr,irn,metftn,metopt,invprm,perm)
    implicit none
    integer, intent(in) :: n
    integer, intent(in) :: iptr(n+1)
    integer, intent(in) :: irn(*)
    integer, intent(in) :: metftn
    integer, intent(in) :: metopt(8)
    integer, intent(out) :: invprm(n)
    integer, intent(out) :: perm(n)

    ! Call MeTiS 4 to get ordering
    call metis_nodend(n,iptr,irn,metftn,metopt,invprm,perm)
end subroutine galahad_metis
