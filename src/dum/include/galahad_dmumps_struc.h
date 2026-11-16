! THIS VERSION: GALAHAD 5.5 - 2026-07-12 AT 14:20 GMT.

      TYPE DMUMPS_STRUC

!  This dummy structure contains a subset of the parameters for the
!  interface to the user, plus internal information from the MUMPS solver.

        SEQUENCE
!
! This structure contains all parameters 
! for the interface to the user, plus internal
! information from the solver
!
! *****************
! INPUT PARAMETERS
! *****************
!    -----------------
!    MPI Communicator
!    -----------------
        INTEGER ( KIND = ip_ ) :: COMM
!    ------------------
!    Problem definition
!    ------------------
!    Solver (SYM=0 unsymmetric,SYM=1 symmetric Positive Definite, 
!        SYM=2 general symmetric)
!    Type of parallelism (PAR=1 host working, PAR=0 host not working)
        INTEGER ( KIND = ip_ ) ::  SYM, PAR
        INTEGER ( KIND = ip_ ) ::  JOB 
!    --------------------
!    Order of Input matrix 
!    --------------------
        INTEGER ( KIND = ip_ ) ::  N
!
!    ----------------------------------------
!    Assembled input matrix : User interface
!    ----------------------------------------
        INTEGER ( KIND = ip_ )    :: NZ  ! Standard integer input + bwd. compat.
        INTEGER ( KIND = long_ ) :: NNZ ! 64-bit integer input
        DOUBLE PRECISION, DIMENSION(:), POINTER :: A
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: IRN, JCN
!    --------------
!    Scaling arrays
!    --------------
        DOUBLE PRECISION, DIMENSION(:), POINTER :: COLSCA, ROWSCA
        DOUBLE PRECISION, DIMENSION(:), POINTER :: COLSCA_loc
        DOUBLE PRECISION, DIMENSION(:), POINTER :: ROWSCA_loc
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: ROWIND, COLIND
        DOUBLE PRECISION, DIMENSION(:), POINTER :: PIVOTS
!
!       ------------------------------------
!       Case of distributed assembled matrix
!       matrix on entry:
!       ------------------------------------
        INTEGER ( KIND = ip_ )    :: NZ_loc  ! Standard integer input + bwd. compat.
        INTEGER ( KIND = ip_ )    :: pad1
        INTEGER ( KIND = long_ ) :: NNZ_loc ! 64-bit integer input
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: IRN_loc, JCN_loc
        DOUBLE PRECISION, DIMENSION(:), POINTER :: A_loc, pad2
!
!    ----------------------------------------
!    Unassembled input matrix: User interface
!    ----------------------------------------
        INTEGER ( KIND = ip_ ) :: NELT, pad3
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: ELTPTR
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: ELTVAR
        DOUBLE PRECISION, DIMENSION(:), POINTER :: A_ELT, pad4
!
!    ---------------------------------------------
!    Symmetric permutation : 
!               PERM_IN if given by user (optional)
!    ---------------------------------------------
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: PERM_IN
!
!    ----------------
!    Format by blocks
!    ----------------
        INTEGER ( KIND = ip_ ) :: NBLK, pad5
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: BLKPTR
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: BLKVAR
!
! ******************
! INPUT/OUTPUT data 
! ******************
!    --------------------------------------------------------
!    RHS / SOL_loc
!    -------------
!       right-hand side and solution
!    -------------------------------------------------------
        DOUBLE PRECISION, DIMENSION(:), POINTER :: RHS, REDRHS
        DOUBLE PRECISION, DIMENSION(:), POINTER :: RHS_SPARSE
        DOUBLE PRECISION, DIMENSION(:), POINTER :: SOL_loc
        DOUBLE PRECISION, DIMENSION(:), POINTER :: RHS_loc
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: IRHS_SPARSE
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: IRHS_PTR
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: ISOL_loc
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: IRHS_loc
        INTEGER ( KIND = ip_ ) :: LRHS, NRHS, NZ_RHS, Nloc_RHS, LRHS_loc, LREDRHS
        INTEGER ( KIND = ip_ ) :: LSOL_loc, NSOL_loc
        INTEGER ( KIND = ip_ ) :: LD_RHSINTR, pad6
!    ----------------------------
!    Control parameters,
!    statistics and output data
!    ---------------------------
        INTEGER ( KIND = ip_ ) ::  ICNTL(60)
        INTEGER ( KIND = ip_ ) ::  INFO(80) 
        INTEGER ( KIND = ip_ ) :: INFOG(80)
        DOUBLE PRECISION ::  COST_SUBTREES
        DOUBLE PRECISION ::  CNTL(15)
        DOUBLE PRECISION ::  RINFO(40)
        DOUBLE PRECISION ::  RINFOG(40)
! The options array for metis/parmetis
        INTEGER ( KIND = ip_ ) ::  METIS_OPTIONS(40)
!    ---------------------------------------------------------
!    Permutations computed during analysis:
!       SYM_PERM: Symmetric permutation 
!       UNS_PERM: Column permutation (optional)
!    ---------------------------------------------------------
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: SYM_PERM, UNS_PERM
! 
!    -----
!    Schur
!    -----
        INTEGER ( KIND = ip_ ) ::  NPROW, NPCOL, MBLOCK, NBLOCK
        INTEGER ( KIND = ip_ ) ::  SCHUR_MLOC, SCHUR_NLOC, SCHUR_LLD
        INTEGER ( KIND = ip_ ) ::  SIZE_SCHUR
        DOUBLE PRECISION, DIMENSION(:), POINTER :: SCHUR
        DOUBLE PRECISION, DIMENSION(:), POINTER :: SCHUR_CINTERFACE
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: LISTVAR_SCHUR
!    -------------------------------------
!    Case of distributed matrix on entry:
!    DMUMPS potentially provides mapping
!    -------------------------------------
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: MAPPING
!    --------------
!    Version number
!    --------------
        CHARACTER(LEN=30) ::  VERSION_NUMBER
!    -----------
!    Out-of-core
!    -----------
        CHARACTER(LEN=1023) :: OOC_TMPDIR
        CHARACTER(LEN=255) :: OOC_PREFIX
!    ------------------------------------------
!    Name of file to dump a matrix/rhs to disk
!    ------------------------------------------
        CHARACTER(LEN=1023) ::  WRITE_PROBLEM
!    -----------
!    Save/Restore
!    -----------
        CHARACTER(LEN=1023) :: SAVE_DIR
        CHARACTER(LEN=255)  :: SAVE_PREFIX
        CHARACTER(LEN=7)   ::  pad7  
!
!
! **********************
! INTERNAL Working data
! *********************
        INTEGER ( KIND = long_ ) :: KEEP8(150), MAX_SURF_MASTER
        INTEGER ( KIND = ip_ ) ::  INST_Number
!       For MPI
        INTEGER ( KIND = ip_ ) ::  COMM_NODES, MYID_NODES, COMM_LOAD
        INTEGER ( KIND = ip_ ) ::  MYID, NPROCS, NSLAVES
        INTEGER ( KIND = ip_ ) ::  ASS_IRECV
!       IS is used for the factors + workspace for contrib. blocks
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: IS
        INTEGER ( KIND = ip_ ) ::  KEEP(500)
!       The following data/arrays are computed during the analysis
!       phase and used during the factorization and solve phases.
        INTEGER ( KIND = ip_ ) ::  LNA
        INTEGER ( KIND = ip_ ) ::  NBSA
        INTEGER ( KIND = ip_ ),POINTER,DIMENSION(:) :: STEP, NE_STEPS, ND_STEPS
        INTEGER ( KIND = ip_ ),POINTER,DIMENSION(:) :: FRERE_STEPS, DAD_STEPS
        INTEGER ( KIND = ip_ ),POINTER,DIMENSION(:) :: FILS, FRTPTR, FRTELT
        INTEGER ( KIND = long_ ),POINTER,DIMENSION(:) :: PTRAR, PTR8ARR
        INTEGER ( KIND = ip_ ),POINTER,DIMENSION(:) :: NINCOLARR,NINROWARR,PTRDEBARR
        INTEGER ( KIND = ip_ ),POINTER,DIMENSION(:) :: NA, PROCNODE_STEPS
!       Info for pruning tree 
        INTEGER ( KIND = ip_ ),POINTER,DIMENSION(:) :: Step2node
!       PTLUST_S and PTRFAC are two pointer arrays computed during
!       factorization and used by the solve
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: PTLUST_S
        INTEGER ( KIND = long_ ), DIMENSION(:), POINTER :: PTRFAC
!       main real working arrays for factorization/solve phases
        DOUBLE PRECISION, DIMENSION(:), POINTER :: S
        REAL(kind(0.E0)), DIMENSION(:), POINTER :: LPS
!       Information on mapping
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: PROCNODE
!       Input matrix ready for numerical assembly 
!           -arrowhead format in case of assembled matrix
!           -element format otherwise
!       Element entry: internal data
        INTEGER ( KIND = ip_ ) :: NELT_loc, LELTVAR
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: ELTPROC
!       Candidates and node partitionning
        INTEGER ( KIND = ip_ ), DIMENSION(:,:), POINTER :: CANDIDATES
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: ISTEP_TO_INIV2
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: FUTURE_NIV2
        INTEGER ( KIND = ip_ ), DIMENSION(:,:), POINTER :: TAB_POS_IN_PERE 
        LOGICAL, DIMENSION(:),   POINTER :: I_AM_CAND
!       For heterogeneous architecture
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: MEM_DIST
!       Compressed RHS
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: GLOB2LOC_RHS
        LOGICAL  :: GLOB2LOC_SOL_ALLOC, pad8
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: GLOB2LOC_SOL
        DOUBLE PRECISION, DIMENSION(:),   POINTER :: RHSINTR
!       Info on the subtrees to be used during factorization
        DOUBLE PRECISION, DIMENSION(:), POINTER :: MEM_SUBTREE
        DOUBLE PRECISION, DIMENSION(:), POINTER :: COST_TRAV
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: MY_ROOT_SBTR
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: MY_FIRST_LEAF
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: MY_NB_LEAF
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: DEPTH_FIRST
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: DEPTH_FIRST_SEQ
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: SBTR_ID
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: SCHED_DEP
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: SCHED_GRP
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: SCHED_SBTR
        INTEGER ( KIND = ip_ ), DIMENSION(:),   POINTER :: CROIX_MANU
        DOUBLE PRECISION, DIMENSION(:),   POINTER :: WK_USER
        INTEGER ( KIND = ip_ ) :: NBSA_LOCAL
        INTEGER ( KIND = ip_ ) :: LWK_USER
!    Internal control array
        DOUBLE PRECISION ::  DKEEP(230)
!    For simulating parallel out-of-core stack.
        DOUBLE PRECISION, DIMENSION(:),POINTER :: CB_SON_SIZE
!    Instance number used/managed by the C/F77 interface
        INTEGER ( KIND = ip_ ) ::  INSTANCE_NUMBER
!    OOC management data that must persist from factorization to solve.
        INTEGER ( KIND = ip_ ) ::  OOC_MAX_NB_NODES_FOR_ZONE
        INTEGER ( KIND = ip_ ), DIMENSION(:,:),   POINTER :: OOC_INODE_SEQUENCE
        INTEGER ( KIND = long_ ),DIMENSION(:,:), POINTER :: OOC_SIZE_OF_BLOCK
        INTEGER ( KIND = long_ ), DIMENSION(:,:),   POINTER :: OOC_VADDR
        INTEGER ( KIND = ip_ ),DIMENSION(:), POINTER :: OOC_TOTAL_NB_NODES
        INTEGER ( KIND = ip_ ),DIMENSION(:), POINTER :: OOC_NB_FILES
        INTEGER ( KIND = ip_ ) :: OOC_NB_FILE_TYPE,pad9
        INTEGER ( KIND = ip_ ),DIMENSION(:), POINTER :: OOC_FILE_NAME_LENGTH
        CHARACTER,DIMENSION(:,:), POINTER :: OOC_FILE_NAMES  
!    Indices of nul pivots
        INTEGER ( KIND = ip_ ),DIMENSION(:), POINTER :: PIVNUL_LIST
!    Array needed to manage additionnal candidate processor 
        INTEGER ( KIND = ip_ ), DIMENSION(:,:), POINTER :: SUP_PROC, pad10
!    Lists of nodes where processors work. Built/used in solve phase.
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: IPTR_WORKING, WORKING
!    Internal data structures accessor
        CHARACTER, DIMENSION(:), POINTER :: INTR_ENCODING
!    Low-rank
        INTEGER ( KIND = ip_ ), POINTER, DIMENSION(:) :: LRGROUPS
        INTEGER ( KIND = ip_ ) :: NBGRP,pad11
!    Pointer encoding for FDM_F data
        CHARACTER, DIMENSION(:), POINTER :: FDM_F_ENCODING
!    Pointer array encoding BLR factors pointers
        CHARACTER, DIMENSION(:), POINTER :: BLRARRAY_ENCODING
!    Multicore
        INTEGER ( KIND = ip_ ) :: LPOOL_A_L0_OMP, LPOOL_B_L0_OMP
        INTEGER ( KIND = ip_ ) :: L_PHYS_L0_OMP
        INTEGER ( KIND = ip_ ) :: L_VIRT_L0_OMP
        INTEGER ( KIND = ip_ ) :: LL0_OMP_MAPPING, LL0_OMP_FACTORS
        INTEGER ( KIND = long_ ) :: THREAD_LA
! Estimates before L0_OMP
        INTEGER ( KIND = ip_ ), DIMENSION(:,:), POINTER    :: I4_L0_OMP
        INTEGER ( KIND = long_ ), DIMENSION(:,:), POINTER :: I8_L0_OMP
! Pool before L0_OMP
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: IPOOL_B_L0_OMP
! Pool after L0_OMP
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: IPOOL_A_L0_OMP
! Subtrees
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: PHYS_L0_OMP
! Amalgamated subtrees
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: VIRT_L0_OMP
! Mapping of amalgamated subtrees
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: VIRT_L0_OMP_MAPPING
! From heaviest to lowest subtree
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: PERM_L0_OMP
! To get leafs in global pool
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: PTR_LEAFS_L0_OMP
! Mapping of the subtree nodes
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: L0_OMP_MAPPING
! Mpi to omp - mumps agile
        INTEGER ( KIND = ip_ ), DIMENSION(:), POINTER :: MTKO_PROCS_MAP
! for Rank-Revealing on root
        DOUBLE PRECISION, DIMENSION(:), POINTER :: SINGULAR_VALUES
        INTEGER ( KIND = ip_ ) ::  NB_SINGULAR_VALUES,pad12
! To know if OOC files are associated to a saved and so if they should be removed.
        LOGICAL :: ASSOCIATED_OOC_FILES,pad13
      END TYPE DMUMPS_STRUC
