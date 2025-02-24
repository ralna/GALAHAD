! THIS VERSION: GALAHAD 5.2 - 2025-02-20 AT 08:25 GMT.

#ifdef INTEGER_64
#define galahad_kinds galahad_kinds_64
#define galahad_metis51 galahad_metis51_64
#define galahad_metis51_setup galahad_metis51_setup_64
#define galahad_metis51_adapter galahad_metis51_adapter_64
#endif

!  MeTiS 5.1 interface

  SUBROUTINE galahad_metis51( n, iptr, irn, metftn, metopt, invprm, perm )
    USE galahad_kinds, ONLY: ip_
    IMPLICIT NONE
    INTEGER( KIND = ip_ ), iNTENT( IN ) :: n
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: iptr
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( * ) :: irn
    INTEGER( KIND = ip_ ), INTENT( IN ) :: metftn
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( 8 ) :: metopt
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: invprm
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: perm

    ! C interface for MeTiS 5 adapter
    INTERFACE
      SUBROUTINE galahad_metis51_adapter( nvtxs, xadj, adjncy, numflag,        &
                                         options, perm, iperm ) BIND(C)
        USE galahad_kinds, ONLY: ipc_
        IMPLICIT NONE
        INTEGER( KIND = ipc_ ), INTENT( IN ) :: nvtxs, numflag
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: xadj, adjncy
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: options
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( OUT ) :: perm, iperm
      END SUBROUTINE galahad_metis51_adapter
    END INTERFACE
    ! call MeTiS 5 to get ordering via C MeTiS 4 to 5 adapter
!write(6,*) ' n ', n
!write(6,*) ' iptr ', iptr
!write(6,*) ' irn ', irn( : iptr( n + 1 ) - 1 )
!write(6,*) ' metftn ', metftn
!write(6,*) ' metopt ', metopt
    CALL galahad_metis51_adapter( n, iptr, irn, metftn, metopt, invprm, perm )
!write(6,*) ' invprm ', invprm
!write(6,*) ' perm ', perm
  RETURN
  END SUBROUTINE galahad_metis51

!  MeTiS 4 and 5 default options

  SUBROUTINE galahad_metis51_setopt( metopt )
    USE galahad_kinds, ONLY: ip_
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( 8 ) :: metopt
    metopt = (/ 1, & ! set options
                0, & ! METIS_OPTION_CTYPE
                1, & ! METIS_OPTION_IPTYPE
                0, & ! METIS_OPTION_RTYPE
                0, & ! METIS_OPTION_DBGLVL (not default)
                1, & ! METIS_OPTION_COMPRESS and _CCORDER
                0, & ! METIS_OPTION_PFACTOR
                1 /) ! METIS_OPTION_NSEPS
  RETURN
  END SUBROUTINE galahad_metis51_setopt
