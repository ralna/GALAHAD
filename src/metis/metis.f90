! THIS VERSION: GALAHAD 5.0 - 2024-03-25 AT 08:25 GMT.

#ifdef INTEGER_64
#define galahad_kinds galahad_kinds_64
#define galahad_metis galahad_metis_64
#define galahad_metis_setup galahad_metis_setup_64
#define galahad_metis5_adapter galahad_metis5_adapter_64
#endif

!  MeTiS 4 and 5 interfaces

  SUBROUTINE galahad_metis( n, iptr, irn, metftn, metopt, invprm, perm )
    USE galahad_kinds, ONLY: ip_
    IMPLICIT NONE
    INTEGER( KIND = ip_ ), iNTENT( IN ) :: n
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: iptr
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( * ) :: irn
    INTEGER( KIND = ip_ ), INTENT( IN ) :: metftn
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( 8 ) :: metopt
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: invprm
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: perm

#ifdef METIS4
    ! Call MeTiS 4 to get ordering
    CALL metis_nodend( n, iptr, irn, metftn, metopt, invprm, perm )
#else
    ! C interface for MeTiS 4 to 5 adapter
    INTERFACE
      SUBROUTINE galahad_metis5_adapter( nvtxs, xadj, adjncy, numflag,         &
                                         options, perm, iperm ) BIND(C)
        USE galahad_kinds, ONLY: ipc_
        IMPLICIT NONE
        INTEGER( KIND = ipc_ ), INTENT( IN ) :: nvtxs, numflag
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: xadj, adjncy
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: options
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( OUT ) :: perm, iperm
      END SUBROUTINE galahad_metis5_adapter
    END INTERFACE
    ! call MeTiS 5 to get ordering via C MeTiS 4 to 5 adapter
!write(6,*) ' n ', n
!write(6,*) ' iptr ', iptr
!write(6,*) ' irn ', irn( : iptr( n + 1 ) - 1 )
!write(6,*) ' metftn ', metftn
!write(6,*) ' metopt ', metopt
    CALL galahad_metis5_adapter( n, iptr, irn, metftn, metopt, invprm, perm )
!write(6,*) ' invprm ', invprm
!write(6,*) ' perm ', perm
#endif
  RETURN
  END SUBROUTINE galahad_metis

!  MeTiS 4 and 5 default options

  SUBROUTINE galahad_metis_setopt( metopt )
    USE galahad_kinds, ONLY: ip_
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( 8 ) :: metopt
#ifdef METIS4
    metopt = (/ 0,  & ! OPTION_PTYPE
                3,  & ! OPTION_CTYPE
                1,  & ! OPTION_ITYPE
                1,  & ! OPTION_RTYPE
                0,  & ! OPTION_DBGLVL (not default)
                1,  & ! OPTION_OFLAGS
                -1, & ! OPTION_PFACTOR
                1  /) ! OPTION_NSEPS
#else
    metopt = (/ 1, & ! set options
                0, & ! METIS_OPTION_CTYPE
                1, & ! METIS_OPTION_IPTYPE
                0, & ! METIS_OPTION_RTYPE
                0, & ! METIS_OPTION_DBGLVL (not default)
                1, & ! METIS_OPTION_COMPRESS and _CCORDER
                0, & ! METIS_OPTION_PFACTOR
                1 /) ! METIS_OPTION_NSEPS
#endif
  RETURN
  END SUBROUTINE galahad_metis_setopt
