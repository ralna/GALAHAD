! THIS VERSION: GALAHAD 5.5 - 2026-07-27 AT 17:20 GMT
!
! Pure-Fortran CPU subtree for SSIDS: an implementation of the abstract
! symbolic_subtree_base / numeric_subtree_base (see subtree.F90) that factors and
! solves entirely in Fortran, using the verified multifrontal driver in
! GALAHAD_SSIDS_factor_precision -- NO C++ / no bind(C).
!
! It is a drop-in alternative to GALAHAD_SSIDS_numeric_subtree_precision (the C++
! wrapper). ssids.F90 selects it at analyse time when the environment variable
! GALAHAD_SSIDS_FORTRAN is set to 1 (additive toggle; the C++ path remains the
! default until this is validated against ssidst).
!
! Scope: serial LDL^T with threshold partial pivoting, delayed pivots and
! foreign child_contrib (all unit-tested standalone). No OpenMP tasking and no
! small-leaf-subtree specialisation (a performance optimisation, not needed for
! correctness).

#include "galahad_modules.h"

 MODULE GALAHAD_SSIDS_numeric_subtree_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SSIDS_types_precision, ONLY: SSIDS_control_type,                &
                                            SSIDS_inform_type,                 &
                                            SSIDS_SUCCESS,                     &
                                            SSIDS_ERROR_ALLOCATION,            &
                                            SSIDS_ERROR_SINGULAR,             &
                                            SSIDS_ERROR_NOT_POS_DEF,          &
                                            PIVOT_METHOD_TPP,                  &
                                            PIVOT_METHOD_APP_AGGRESIVE,        &
                                            FAILED_PIVOT_METHOD_TPP,           &
                                            contrib_type
   USE GALAHAD_SSIDS_subtree_precision, ONLY : symbolic_subtree_base,          &
                                               numeric_subtree_base
   USE GALAHAD_SSIDS_factor_precision, ONLY : dmf_node,                &
        subtree_contrib_t, factor_subtree_delay, extract_contrib,             &
        subtree_solve_fwd_delay, subtree_solve_diag_delay,                    &
        subtree_solve_bwd_delay
   IMPLICIT none

   PRIVATE
   PUBLIC :: symbolic_subtree, construct_symbolic_subtree
   PUBLIC :: numeric_subtree, free_contrib

   !> Per-node symbolic template (built once at analyse time).
   TYPE :: node_symb_type
      INTEGER( ip_ ) :: symb_ncol = 0, symb_nrow = 0, parent = 0
      INTEGER( ip_ ), ALLOCATABLE :: rlist(:)          ! global rows
      INTEGER( long_ ), ALLOCATABLE :: a_src(:)        ! src index into aval
      INTEGER( ip_ ), ALLOCATABLE :: a_ai(:), a_aj(:)  ! symbolic front (r,c)
      INTEGER( ip_ ), ALLOCATABLE :: contribs(:)       ! local foreign contrib ids
   END TYPE node_symb_type

   TYPE, EXTENDS( symbolic_subtree_base ) :: symbolic_subtree
      INTEGER( ip_ ) :: n = 0, nnodes = 0
      TYPE( node_symb_type ), ALLOCATABLE :: nodes(:)
   CONTAINS
      PROCEDURE :: factor
      PROCEDURE :: cleanup => symbolic_cleanup
   END TYPE symbolic_subtree

   TYPE, EXTENDS( numeric_subtree_base ) :: numeric_subtree
      LOGICAL :: posdef = .FALSE.
      INTEGER( ip_ ) :: n = 0, nnodes = 0
      TYPE( symbolic_subtree ), POINTER :: symbolic => NULL()
      TYPE( dmf_node ), ALLOCATABLE :: fnode(:)
   CONTAINS
      PROCEDURE :: get_contrib
      PROCEDURE :: solve_fwd
      PROCEDURE :: solve_diag
      PROCEDURE :: solve_diag_bwd
      PROCEDURE :: solve_bwd
      PROCEDURE :: enquire_posdef
      PROCEDURE :: enquire_indef
      PROCEDURE :: alter
      PROCEDURE :: cleanup => numeric_cleanup
   END TYPE numeric_subtree

 CONTAINS

   FUNCTION construct_symbolic_subtree( n, sa, en, sptr, sparent,  &
       rptr, rlist, nptr, nlist, contrib_idx, control ) RESULT( this )
      CLASS( symbolic_subtree ), POINTER :: this
      INTEGER( ip_ ), INTENT( IN ) :: n, sa, en
      INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: sptr, sparent, rlist
      INTEGER( long_ ), DIMENSION( * ), INTENT( IN ) :: rptr, nptr
      INTEGER( long_ ), DIMENSION( 2, * ), INTENT( IN ) :: nlist
      INTEGER( ip_ ), DIMENSION( : ), INTENT( IN ) :: contrib_idx
      CLASS( SSIDS_control_type ), INTENT( IN ) :: control
      INTEGER( ip_ ) :: li, gi, ps, na, k, nrow, ci, tgt
      INTEGER( long_ ) :: p0
      NULLIFY( this )
      ALLOCATE( this )
      this%n = n
      this%nnodes = en - sa
      ALLOCATE( this%nodes( this%nnodes ) )
      DO li = 1, this%nnodes
         gi = sa + li - 1
         ASSOCIATE( nd => this%nodes( li ) )
         nd%symb_ncol = INT( sptr( gi+1 ) - sptr( gi ), ip_ )
         nrow = INT( rptr( gi+1 ) - rptr( gi ), ip_ )
         nd%symb_nrow = nrow
         ALLOCATE( nd%rlist( nrow ) )
         DO k = 1, nrow
            nd%rlist( k ) = rlist( INT( rptr( gi ), ip_ ) + k - 1 )
         END DO
         ps = sparent( gi )
         IF ( ps >= sa .AND. ps <= en - 1 ) THEN
            nd%parent = ps - sa + 1
         ELSE
            nd%parent = 0
         END IF
         na = INT( nptr( gi+1 ) - nptr( gi ), ip_ )
         ALLOCATE( nd%a_src( na ), nd%a_ai( na ), nd%a_aj( na ) )
         p0 = nptr( gi )                       ! 1-based start into nlist columns
         DO k = 1, na
            nd%a_src( k ) = nlist( 1, p0 + k - 1 )
            block
               integer( long_ ) :: dest0
               dest0 = nlist( 2, p0 + k - 1 ) - 1     ! 0-based front linear index
               nd%a_aj( k ) = INT( dest0 / nrow, ip_ ) + 1
               nd%a_ai( k ) = INT( MOD( dest0, INT( nrow, long_ ) ), ip_ ) + 1
            end block
         END DO
         END ASSOCIATE
      END DO
      ! record foreign contribution targets
      DO ci = 1, SIZE( contrib_idx )
         tgt = contrib_idx( ci ) - sa + 1
         IF ( tgt >= 1 .AND. tgt <= this%nnodes ) &
            CALL push( this%nodes( tgt )%contribs, INT( ci, ip_ ) )
      END DO
   END FUNCTION construct_symbolic_subtree

   SUBROUTINE push( a, v )
      INTEGER( ip_ ), ALLOCATABLE, INTENT( INOUT ) :: a(:)
      INTEGER( ip_ ), INTENT( IN ) :: v
      INTEGER( ip_ ), ALLOCATABLE :: tmp(:)
      IF ( .NOT. ALLOCATED( a ) ) THEN
         ALLOCATE( a( 1 ) ); a( 1 ) = v
      ELSE
         ALLOCATE( tmp( SIZE( a ) + 1 ) )
         tmp( 1:SIZE( a ) ) = a; tmp( SIZE( a ) + 1 ) = v
         CALL MOVE_ALLOC( tmp, a )
      END IF
   END SUBROUTINE push

   SUBROUTINE symbolic_cleanup( this )
      CLASS( symbolic_subtree ), INTENT( INOUT ) :: this
      IF ( ALLOCATED( this%nodes ) ) DEALLOCATE( this%nodes )
   END SUBROUTINE symbolic_cleanup

   !> Free a contribution block produced by get_contrib (Fortran allocations).
   !! Replaces the C++ free_contrib; called from contrib_iface.
   SUBROUTINE free_contrib( fcontrib )
      TYPE( contrib_type ), INTENT( INOUT ) :: fcontrib
      IF ( ASSOCIATED( fcontrib%val ) )        DEALLOCATE( fcontrib%val )
      IF ( ASSOCIATED( fcontrib%rlist ) )      DEALLOCATE( fcontrib%rlist )
      IF ( ASSOCIATED( fcontrib%delay_perm ) ) DEALLOCATE( fcontrib%delay_perm )
      IF ( ASSOCIATED( fcontrib%delay_val ) )  DEALLOCATE( fcontrib%delay_val )
      fcontrib%n = 0
   END SUBROUTINE free_contrib

   FUNCTION factor( this, posdef, aval, child_contrib, control, inform,        &
                    scaling )
      CLASS( numeric_subtree_base ), POINTER :: factor
      CLASS( symbolic_subtree ), TARGET, INTENT( INOUT ) :: this
      LOGICAL, INTENT( IN ) :: posdef
      REAL( rp_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: aval
      TYPE( contrib_type ), DIMENSION( : ), TARGET, INTENT( INOUT ) :: child_contrib
      TYPE( SSIDS_control_type ), INTENT( IN ) :: control
      TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: inform
      REAL( rp_ ), DIMENSION( * ), TARGET, OPTIONAL, INTENT( IN ) :: scaling
      TYPE( numeric_subtree ), POINTER :: fac
      TYPE( subtree_contrib_t ), ALLOCATABLE :: contribs(:)
      INTEGER( ip_ ) :: li, k, na, st, nb
      LOGICAL :: ok, aok
      NULLIFY( factor )
      ALLOCATE( fac, STAT = st ); IF ( st /= 0 ) GO TO 10
      fac%symbolic => this
      fac%n = this%n
      fac%nnodes = this%nnodes
      fac%posdef = posdef
      ALLOCATE( fac%fnode( this%nnodes ), STAT = st ); IF ( st /= 0 ) GO TO 10

      ! build numeric nodes from the symbolic template + values of A
      DO li = 1, this%nnodes
         ASSOCIATE( t => this%nodes( li ), fn => fac%fnode( li ) )
         fn%symb_ncol = t%symb_ncol
         fn%symb_nrow = t%symb_nrow
         fn%parent    = t%parent
         fn%rlist     = t%rlist
         na = SIZE( t%a_src )
         ALLOCATE( fn%ai( na ), fn%aj( na ), fn%av( na ) )
         fn%ai = t%a_ai; fn%aj = t%a_aj
         IF ( PRESENT( scaling ) ) THEN
            DO k = 1, na
               fn%av( k ) = scaling( t%rlist( t%a_ai( k ) ) ) * aval( t%a_src( k ) ) &
                          * scaling( t%rlist( t%a_aj( k ) ) )
            END DO
         ELSE
            DO k = 1, na
               fn%av( k ) = aval( t%a_src( k ) )
            END DO
         END IF
         IF ( ALLOCATED( t%contribs ) ) fn%contribs = t%contribs
         END ASSOCIATE
      END DO

      ! convert incoming child contributions
      ALLOCATE( contribs( SIZE( child_contrib ) ), STAT = st ); IF ( st /= 0 ) GO TO 10
      DO k = 1, SIZE( child_contrib )
         CALL import_contrib( child_contrib( k ), contribs( k ) )
      END DO

      ! pivot_method encoded in nb for factor_node_indef: nb=0 -> unblocked TPP;
      ! nb>0 -> APP_BLOCK at |nb| = block_size (default 256, inner 32 via
      ! recursion); nb<0 -> APP_AGGRESSIVE (optimistic unpivoted-first) at |nb|.
      if ( control%pivot_method == PIVOT_METHOD_TPP ) then
         nb = 0_ip_
      else if ( control%pivot_method == PIVOT_METHOD_APP_AGGRESIVE ) then
         nb = - MAX( 1_ip_, control%block_size )
      else
         nb = MAX( 1_ip_, control%block_size )
      end if
      CALL factor_subtree_delay( fac%fnode, this%nnodes, this%n, control%action, &
                                 control%u, control%small, nb, posdef, ok,       &
                                 contribs,                                       &
                                 small_subtree_threshold                         &
                                   = control%small_subtree_threshold,            &
                                 failed_tpp = ( control%failed_pivot_method      &
                                                == FAILED_PIVOT_METHOD_TPP ),   &
                                 alloc_ok = aok )
      IF ( .NOT. aok ) THEN
         inform%flag = SSIDS_ERROR_ALLOCATION      ! out of memory during factor
      ELSE IF ( .NOT. ok ) THEN
         IF ( posdef ) THEN
            inform%flag = SSIDS_ERROR_NOT_POS_DEF
         ELSE
            inform%flag = SSIDS_ERROR_SINGULAR
         END IF
      END IF
      CALL accumulate_stats( fac, inform )
      factor => fac
      RETURN
10    CONTINUE
      inform%flag = SSIDS_ERROR_ALLOCATION
      inform%stat = st
      IF ( ASSOCIATED( fac ) ) DEALLOCATE( fac )
   END FUNCTION factor

   !> Populate inform statistics (pivot counts, rank, factor size) from the
   !! factored nodes -- for parity with the C++ backend's ThreadStats.
   SUBROUTINE accumulate_stats( fac, inform )
      TYPE( numeric_subtree ), INTENT( IN ) :: fac
      TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: inform
      INTEGER( ip_ ) :: li, i, k, nneg, ntwo, ndel, nzero, nfst, nsnd
      REAL( rp_ ) :: d11, d21, d22, det
      LOGICAL :: is1x1
      nneg = 0; ntwo = 0; ndel = 0; nzero = 0; nfst = 0; nsnd = 0
      DO li = 1, fac%nnodes
         ASSOCIATE( fn => fac%fnode( li ) )
         ndel = ndel + fn%ndelay_out
         nfst = nfst + fn%nfirst
         nsnd = nsnd + fn%nsecond
!  L factor entries and flops: column j (0-based) of the nelim eliminated
!  columns has nrow-j sub/diagonal entries -> triangular count, as in the C++
         inform%num_factor = inform%num_factor                                 &
           + INT( fn%nelim, long_ ) * INT( fn%nrow, long_ )                     &
           - ( INT( fn%nelim, long_ ) * INT( fn%nelim - 1, long_ ) ) / 2_long_
         DO k = 0, fn%nelim - 1
            inform%num_flops = inform%num_flops + INT( fn%nrow - k, long_ ) ** 2
         END DO
         i = 0
         DO WHILE ( i < fn%nelim )
            is1x1 = ( i+1 == fn%nelim )
            IF ( .NOT. is1x1 ) is1x1 = ieee_is_finite_local( fn%d( 2*i+3 ) )
            IF ( is1x1 ) THEN
               d11 = fn%d( 2*i+1 )
               IF ( d11 == 0.0_rp_ ) THEN
                  nzero = nzero + 1
               ELSE IF ( d11 < 0.0_rp_ ) THEN
                  nneg = nneg + 1
               END IF
               i = i + 1
            ELSE
               ntwo = ntwo + 1
               d11 = fn%d( 2*i+1 ); d21 = fn%d( 2*i+2 ); d22 = fn%d( 2*i+4 )
               det = d11*d22 - d21*d21          ! det of D^{-1} (same sign as det D)
               IF ( det < 0.0_rp_ ) THEN
                  nneg = nneg + 1               ! indefinite 2x2: one negative
               ELSE IF ( d11 + d22 < 0.0_rp_ ) THEN
                  nneg = nneg + 2               ! both negative (trace<0, as C++)
               END IF
               i = i + 2
            END IF
         END DO
         END ASSOCIATE
      END DO
      ! thread_inform is fresh (0); reduce() sums these into the global inform,
      ! whose matrix_rank starts at n (set by analyse) -- so subtract num_zero.
      inform%matrix_rank = inform%matrix_rank - nzero
      inform%num_neg     = inform%num_neg   + nneg
      inform%num_two     = inform%num_two   + ntwo
      inform%num_delay   = inform%num_delay + ndel
      inform%not_first_pass  = inform%not_first_pass  + nfst
      inform%not_second_pass = inform%not_second_pass + nsnd
   END SUBROUTINE accumulate_stats

   !> contrib_type (GALAHAD interchange) -> subtree_contrib_t (driver form).
   SUBROUTINE import_contrib( c, ct )
      TYPE( contrib_type ), INTENT( IN )  :: c
      TYPE( subtree_contrib_t ), INTENT( OUT ) :: ct
      INTEGER( ip_ ) :: i, j
      ct%cn = c%n
      ct%ndelay = c%ndelay
      IF ( c%n > 0 ) THEN
         ALLOCATE( ct%rlist( c%n ) ); ct%rlist = c%rlist( 1:c%n )
         ALLOCATE( ct%val( c%n, c%n ) )
         DO i = 1, c%n
            DO j = 1, c%n
               ct%val( j, i ) = c%val( ( i-1 )*c%ldval + j )
            END DO
         END DO
      END IF
      IF ( c%ndelay > 0 ) THEN
         ALLOCATE( ct%delay_perm( c%ndelay ) )
         ct%delay_perm = c%delay_perm( 1:c%ndelay )
         ALLOCATE( ct%delay_val( c%lddelay, c%ndelay ) )
         DO i = 1, c%ndelay
            DO j = 1, c%lddelay
               ct%delay_val( j, i ) = c%delay_val( ( i-1 )*c%lddelay + j )
            END DO
         END DO
      END IF
   END SUBROUTINE import_contrib

   SUBROUTINE numeric_cleanup( this )
      CLASS( numeric_subtree ), INTENT( INOUT ) :: this
      IF ( ALLOCATED( this%fnode ) ) DEALLOCATE( this%fnode )
      NULLIFY( this%symbolic )
   END SUBROUTINE numeric_cleanup

   FUNCTION get_contrib( this )
      TYPE( contrib_type ) :: get_contrib
      CLASS( numeric_subtree ), INTENT( IN ) :: this
      TYPE( subtree_contrib_t ) :: ct
      INTEGER( ip_ ) :: root, li, i, j, cm
      root = 0
      DO li = 1, this%nnodes
         IF ( this%fnode( li )%parent == 0 ) root = li
      END DO
      CALL extract_contrib( this%fnode( root ), ct )
      cm = ct%cn
      get_contrib%n = cm
      get_contrib%ndelay = ct%ndelay
      get_contrib%posdef = this%posdef
      IF ( cm > 0 ) THEN
         get_contrib%ldval = cm
         ALLOCATE( get_contrib%val( cm*cm ) )
         DO i = 1, cm
            DO j = 1, cm
               get_contrib%val( ( i-1 )*cm + j ) = ct%val( j, i )
            END DO
         END DO
         ALLOCATE( get_contrib%rlist( cm ) ); get_contrib%rlist = ct%rlist
      END IF
      IF ( ct%ndelay > 0 ) THEN
         get_contrib%lddelay = ct%ndelay + cm
         ALLOCATE( get_contrib%delay_perm( ct%ndelay ) )
         get_contrib%delay_perm = ct%delay_perm
         ALLOCATE( get_contrib%delay_val( ( ct%ndelay + cm )*ct%ndelay ) )
         DO i = 1, ct%ndelay
            DO j = 1, ct%ndelay + cm
               get_contrib%delay_val( ( i-1 )*( ct%ndelay + cm ) + j ) &
                  = ct%delay_val( j, i )
            END DO
         END DO
      ELSE
         NULLIFY( get_contrib%delay_perm )
         NULLIFY( get_contrib%delay_val )
      END IF
   END FUNCTION get_contrib

   SUBROUTINE solve_fwd( this, nrhs, x, ldx, inform )
      CLASS( numeric_subtree ), INTENT( INOUT ) :: this
      INTEGER( ip_ ), INTENT( IN ) :: nrhs
      REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
      INTEGER( ip_ ), INTENT( IN ) :: ldx
      TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: inform
      CALL subtree_solve_fwd_delay( this%fnode, this%nnodes, nrhs, x, ldx )
   END SUBROUTINE solve_fwd

   SUBROUTINE solve_diag( this, nrhs, x, ldx, inform )
      CLASS( numeric_subtree ), INTENT( INOUT ) :: this
      INTEGER( ip_ ), INTENT( IN ) :: nrhs
      REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
      INTEGER( ip_ ), INTENT( IN ) :: ldx
      TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: inform
      CALL subtree_solve_diag_delay( this%fnode, this%nnodes, nrhs, x, ldx )
   END SUBROUTINE solve_diag

   SUBROUTINE solve_diag_bwd( this, nrhs, x, ldx, inform )
      CLASS( numeric_subtree ), INTENT( INOUT ) :: this
      INTEGER( ip_ ), INTENT( IN ) :: nrhs
      REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
      INTEGER( ip_ ), INTENT( IN ) :: ldx
      TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: inform
      CALL subtree_solve_diag_delay( this%fnode, this%nnodes, nrhs, x, ldx )
      CALL subtree_solve_bwd_delay( this%fnode, this%nnodes, nrhs, x, ldx )
   END SUBROUTINE solve_diag_bwd

   SUBROUTINE solve_bwd( this, nrhs, x, ldx, inform )
      CLASS( numeric_subtree ), INTENT( INOUT ) :: this
      INTEGER( ip_ ), INTENT( IN ) :: nrhs
      REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
      INTEGER( ip_ ), INTENT( IN ) :: ldx
      TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: inform
      CALL subtree_solve_bwd_delay( this%fnode, this%nnodes, nrhs, x, ldx )
   END SUBROUTINE solve_bwd

   SUBROUTINE enquire_posdef( this, d )
      CLASS( numeric_subtree ), INTENT( IN ) :: this
      REAL( rp_ ), DIMENSION( * ), INTENT( OUT ) :: d
      INTEGER( ip_ ) :: li, i, kk
!  return the Cholesky diagonal L_ii (as the C++ does), not the pivot L_ii^2.
!  chol_factor_node stores d(2i-1) = 1/L_ii^2, so L_ii = sqrt( 1/d(2i-1) ).
      kk = 0
      DO li = 1, this%nnodes
         ASSOCIATE( fn => this%fnode( li ) )
         DO i = 1, fn%nelim
            kk = kk + 1
            IF ( fn%d( 2*i-1 ) /= 0.0_rp_ ) THEN
               d( kk ) = SQRT( 1.0_rp_ / fn%d( 2*i-1 ) )
            ELSE
               d( kk ) = 0.0_rp_
            END IF
         END DO
         END ASSOCIATE
      END DO
   END SUBROUTINE enquire_posdef

   SUBROUTINE enquire_indef( this, piv_order, d )
      CLASS( numeric_subtree ), INTENT( IN ) :: this
      INTEGER( ip_ ), DIMENSION( * ), INTENT( OUT ), OPTIONAL :: piv_order
      REAL( rp_ ), DIMENSION( * ), INTENT( OUT ), OPTIONAL :: d
      INTEGER( ip_ ) :: li, i, piv, dk
      LOGICAL :: is1x1
      piv = 0; dk = 0
      DO li = 1, this%nnodes
         ASSOCIATE( fn => this%fnode( li ) )
         i = 0
         DO WHILE ( i < fn%nelim )
            is1x1 = ( i+1 == fn%nelim )
            IF ( .NOT. is1x1 ) is1x1 = ieee_is_finite_local( fn%d( 2*i+3 ) )
            IF ( is1x1 ) THEN
               IF ( PRESENT( piv_order ) ) THEN
                  piv_order( fn%perm( i+1 ) ) = piv; piv = piv + 1
               END IF
               IF ( PRESENT( d ) ) THEN
                  d( dk+1 ) = fn%d( 2*i+1 ); d( dk+2 ) = 0.0_rp_; dk = dk + 2
               END IF
               i = i + 1
            ELSE
               IF ( PRESENT( piv_order ) ) THEN
                  piv_order( fn%perm( i+1 ) ) = -piv; piv = piv + 1
                  piv_order( fn%perm( i+2 ) ) = -piv; piv = piv + 1
               END IF
               IF ( PRESENT( d ) ) THEN
                  d( dk+1 ) = fn%d( 2*i+1 ); d( dk+2 ) = fn%d( 2*i+2 )
                  d( dk+3 ) = fn%d( 2*i+4 ); d( dk+4 ) = 0.0_rp_; dk = dk + 4
               END IF
               i = i + 2
            END IF
         END DO
         END ASSOCIATE
      END DO
   END SUBROUTINE enquire_indef

   SUBROUTINE alter( this, d )
      CLASS( numeric_subtree ), INTENT( INOUT ) :: this
      REAL( rp_ ), DIMENSION( * ), INTENT( IN ) :: d
      INTEGER( ip_ ) :: li, i, dk
      LOGICAL :: is1x1
      dk = 0
      DO li = 1, this%nnodes
         ASSOCIATE( fn => this%fnode( li ) )
         i = 0
         DO WHILE ( i < fn%nelim )
            is1x1 = ( i+1 == fn%nelim )
            IF ( .NOT. is1x1 ) is1x1 = ieee_is_finite_local( fn%d( 2*i+3 ) )
            IF ( is1x1 ) THEN
               fn%d( 2*i+1 ) = d( dk+1 ); dk = dk + 2
               i = i + 1
            ELSE
               fn%d( 2*i+1 ) = d( dk+1 ); fn%d( 2*i+2 ) = d( dk+2 )
               fn%d( 2*i+4 ) = d( dk+3 ); dk = dk + 4
               i = i + 2
            END IF
         END DO
         END ASSOCIATE
      END DO
   END SUBROUTINE alter

   LOGICAL FUNCTION ieee_is_finite_local( v ) RESULT( f )
      USE, INTRINSIC :: ieee_arithmetic, ONLY : ieee_is_finite
      REAL( rp_ ), INTENT( IN ) :: v
      f = ieee_is_finite( v )
   END FUNCTION ieee_is_finite_local

 END MODULE GALAHAD_SSIDS_numeric_subtree_precision
