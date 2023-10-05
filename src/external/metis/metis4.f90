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

subroutine galahad_metis_setopt(metopt)
    integer, intent(out) :: metopt(8)
    metopt = (/ 0, & ! OPTION_PTYPE
                3, & ! OPTION_CTYPE
                1, & ! OPTION_ITYPE
                1, & ! OPTION_RTYPE
                0, & ! OPTION_DBGLVL (not default)
                1, & ! OPTION_OFLAGS
                -1, & ! OPTION_PFACTOR
                1 /) ! OPTION_NSEPS
end subroutine galahad_metis_setopt
