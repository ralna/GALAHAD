! Routine that is called if MeTiS 5 is linked.
subroutine galahad_metis(n,iptr,irn,metftn,metopt,invprm,perm)
    implicit none
    integer, intent(in) :: n
    integer, intent(in) :: iptr(n+1)
    integer, intent(in) :: irn(*)
    integer, intent(in) :: metftn
    integer, intent(in) :: metopt(8)
    integer, intent(out) :: invprm(n)
    integer, intent(out) :: perm(n)

    ! C interface for MeTiS 4 to 5 adapter
    interface
        subroutine metis5_adapter(nvtxs, xadj, adjncy, numflag, options, &
                                  perm, iperm) bind(C)
            use iso_c_binding
            implicit none
            integer(c_int), intent(in) :: nvtxs, numflag
            integer(c_int), dimension(*), intent(in) :: xadj, adjncy, options
            integer(c_int), dimension(*), intent(out) :: perm, iperm
        end subroutine metis5_adapter
    end interface

    ! Call MeTiS 5 to get ordering via C MeTiS 4 to 5 adapter
    call metis5_adapter(n,iptr,irn,metftn,metopt,invprm,perm)
end subroutine galahad_metis
