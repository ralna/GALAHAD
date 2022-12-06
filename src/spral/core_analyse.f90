! (c) STFC 2010-2013
! Author: Jonathan Hogg
!
! Given a pivot order, this package performs common tasks
! required in the analyse phase of a symmetric sparse direct solver.
! Either the entire analyse may be performed or individual tasks.
! The matrix may be held in assembled form.
!
! Routines originally based on HSL_MC78 v1.2.0
module spral_core_analyse
  implicit none

  private
  public :: basic_analyse ! Perform a full analysis for a given matrix ordering

  integer, parameter :: long = selected_int_kind(18)
  integer, parameter :: ptr_kind = long ! integer kind used for user's
    ! column pointers (rptr is always long) - integer or long

  integer, parameter :: minsz_ms = 16 ! minimum size to use merge sort

  integer, parameter :: ERROR_ALLOCATION = -1
  integer, parameter :: WARNING_SINGULAR = 1

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!            Outline analysis routine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! For assembled matrix input, this subroutine performs a full analysis.
! This is essentially a wrapper around the rest of the package.
!
! Performance might be improved by:
! * Improving the sort algorithm used in find_row_idx
!
  subroutine basic_analyse(n, ptr, row, perm, nnodes, sptr, &
       sparent, rptr, rlist, nemin, info, stat, nfact, nflops)
    implicit none
    integer, intent(in) :: n ! Dimension of system
    integer(ptr_kind), dimension(n+1), intent(in) :: ptr ! Column pointers
    integer, dimension(ptr(n+1)-1), intent(in) :: row ! Row indices
    integer, dimension(n), intent(inout) :: perm
      ! perm(i) must hold position of i in the pivot sequence.
      ! On exit, holds the pivot order to be used by factorization.
    integer, intent(out) :: nnodes ! number of supernodes found
    integer, dimension(:), allocatable, intent(out) :: sptr ! supernode pointers
    integer, dimension(:), allocatable, intent(out) :: sparent ! assembly tree
    integer(long), dimension(:), allocatable, intent(out) :: rptr
      ! pointers to rlist
    integer, dimension(:), allocatable, intent(out) :: rlist ! row lists
    integer, intent(in) :: nemin ! Node amalgamation parameter
    integer, intent(out) :: info
    integer, intent(out) :: stat
    integer(long), intent(out) :: nfact
    integer(long), intent(out) :: nflops

    integer :: i
    integer, dimension(:), allocatable :: invp ! inverse permutation of perm
    integer :: j
    integer :: realn ! number of variables with an actual entry present
    integer :: st ! stat argument in allocate calls

    integer, dimension(:), allocatable :: scc
    integer, dimension(:), allocatable :: cc ! number of entries in each column
    integer, dimension(:), allocatable :: parent ! parent of each node in etree
    integer, dimension(:), allocatable :: tperm ! temporary permutation vector

    ! Quick exit for n < 0.
    ! ERROR_ALLOCATION will be signalled, but since allocation status cannot be
    ! negative, in this way info about the invalid argument (n < 0) is conveyed.
    if (n .lt. 0) then
       st = n
    else
       st = 0
    end if
    if (st .ne. 0) goto 490

    ! Initialize return code to success
    info = 0

    ! Ensure allocatable output arguments are deallocated
    deallocate(sptr, stat=st)
    deallocate(sparent, stat=st)
    deallocate(rptr, stat=st)
    deallocate(rlist, stat=st)

    ! Initialise inverse permutation and check for duplicates
    allocate(invp(n), stat=st)
    if (st .ne. 0) goto 490
    do i = 1, n
       j = perm(i)
       invp(j) = i
    end do

    realn = n ! Assume full rank

    ! Build elimination tree
    allocate(parent(n), stat=st)
    if (st .ne. 0) goto 490
    call find_etree(n, ptr, row, perm, invp, parent, st)
    if (st .ne. 0) goto 490

    ! Postorder tree (modifies perm!)
    call find_postorder(n, realn, ptr, perm, invp, parent, st)
    if (st .ne. 0) goto 490

    if (n .ne. realn) info = WARNING_SINGULAR

    ! Determine column counts
    allocate(cc(n+1), stat=st)
    if (st .ne. 0) goto 490
    call find_col_counts(n, ptr, row, perm, invp, parent, cc, st)
    if (st .ne. 0) goto 490

    ! Identify supernodes
    allocate(tperm(n), sptr(n+1), sparent(n), scc(n), stat=st)
    if (st .ne. 0) goto 490
    call find_supernodes(n, realn, parent, cc, tperm, nnodes, sptr, sparent, &
         scc, nemin, info, st)
    if (info .lt. 0) return

    ! Apply permutation to obtain final elimination order
    call apply_perm(n, tperm, perm, invp, cc)

    ! Determine column patterns - keep%nodes(:)%index
    allocate(rptr(nnodes+1), rlist(sum(scc(1:nnodes))), stat=st)
    if (st .ne. 0) goto 490
    call find_row_lists(n, ptr, row, perm, invp, nnodes, sptr, &
         sparent, scc, rptr, rlist, info, st)
    if (st .ne. 0) goto 490

    ! Calculate info%num_factor and info%num_flops
    call calc_stats(nnodes, sptr, scc, nfact=nfact, nflops=nflops)

    ! Sort entries of row lists
    call dbl_tr_sort(n, nnodes, rptr, rlist, st)
    if (st .ne. 0) goto 490

    return

    !!!!!!!!!!!!!!!!!!
    ! Error handlers !
    !!!!!!!!!!!!!!!!!!

490 continue
    info = ERROR_ALLOCATION
    stat = st
    return
  end subroutine basic_analyse

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Elimination tree routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! This subroutine determines the elimination tree of a PAP^T where A is a
! sparse symmetric matrix stored in compressed sparse column form with
! entries both above and below the diagonal present in the argument matrix.
! P is a permutation stored in order such that order(i) gives the pivot
! position of column i. i.e. order(3) = 5 means that the fifth pivot is
! A_33.
!
! The elimination tree is returned in the array parent. parent(i) gives the
! parent in the elimination tree of pivot i.
!
! The algorithm used is that of Liu [1].
!
! [1] Liu, J. W. 1986. A compact row storage scheme for Cholesky factors using
!     elimination trees. ACM TOMS 12, 2, 127--148.
!
  subroutine find_etree(n, ptr, row, perm, invp, parent, st)
    implicit none
    integer, intent(in) :: n ! dimension of system
    integer(ptr_kind), dimension(n+1), intent(in) :: ptr ! column pointers of A
    integer, dimension(ptr(n+1)-1), intent(in) :: row ! row indices of A
    integer, dimension(n), intent(in) :: perm ! perm(i) is the pivot position
      ! of column i
    integer, dimension(n), intent(in) :: invp ! inverse of perm
    integer, dimension(n), intent(out) :: parent ! parent(i) is the
      ! parent of pivot i in the elimination tree
    integer, intent(out) :: st ! stat parmeter for allocate calls

    integer(ptr_kind) :: i ! next index into row
    integer :: j ! current entry in row
    integer :: k ! current ancestor
    integer :: l ! next ancestor
    integer :: piv ! current pivot
    integer :: rowidx ! current column of A = invp(piv)
    integer, dimension(:), allocatable :: vforest ! virtual forest, used for
      ! path compression (shortcuts to top of each tree)

    ! Allocate virtual forest and initialise it
    allocate(vforest(n), stat=st)
    if (st .ne. 0) return
    vforest(:) = n+1

    ! Loop over rows of A in pivot order
    piv = 1
    do while (piv.le.n)
       !print *, "row ", piv
       rowidx = invp(piv)
       ! Loop over entries in row in lower triangle of PAP^T
       do i = ptr(rowidx), ptr(rowidx+1)-1
          j = perm(row(i))
          if (j .ge. piv) cycle ! not in lower triangle
          !print *, "  entry ", j
          k = j
          do while (vforest(k).lt.piv)
             l = vforest(k)
             vforest(k) = piv
             k = l
          end do
          ! Check if we have already done this pivot
          if (vforest(k) .eq. piv) cycle
          parent(k) = piv
          vforest(k) = piv
       end do
       parent(piv) = n + 1 ! set to be a root if not overwritten
       piv = piv + 1 ! move on to next pivot
    end do
  end subroutine find_etree

!
! This subroutine will postorder the elimination tree. That is to say it will
! reorder the nodes of the tree such that they are in depth-first search order.
!
! This is done by performing a depth-first search to identify mapping from the
! original pivot order to the new one. This map is then applied to order, invp
! and parent to enact the relabelling.
!
  subroutine find_postorder(n, realn, ptr, perm, invp, parent, st)
    implicit none
    integer, intent(in) :: n
    integer, intent(out) :: realn
    integer(ptr_kind), dimension(n+1), intent(in) :: ptr
    integer, dimension(n), intent(inout) :: perm ! perm(i) is the pivot
      ! position of column i
    integer, dimension(n), intent(inout) :: invp ! inverse of perm
    integer, dimension(n), intent(inout) :: parent ! parent(i) is the
      ! parent of pivot i in the elimination tree
    integer, intent(out) :: st ! stat parmeter for allocate calls

    integer, dimension(:), allocatable :: chead ! chead(i) is first child of i
    integer, dimension(:), allocatable :: cnext ! cnext(i) is next child of i
    integer :: i
    integer :: id
    integer :: j
    integer, dimension(:), allocatable :: map ! mapping from original pivot
      ! order to new one
    integer :: node
    integer :: shead ! pointer to top of stack
    integer, dimension(:), allocatable :: stack ! stack for depth first search

    realn = n

    !
    ! Build linked lists of children for each node
    !
    allocate(chead(n+1), cnext(n+1), stat=st)
    if (st .ne. 0) return
    chead(:) = -1 ! no parent if necessary
    do i = n, 1, -1 ! do in reverse order so they come off in original order
       j = parent(i)
       cnext(i) = chead(j)
       chead(j) = i
    end do

    !
    ! Perform depth first search to build map
    !
    allocate(map(n+1), stack(n), stat=st)
    if (st .ne. 0) return
    ! Place virtual root on top of stack
    shead = 1
    stack(shead) = n+1
    id = n + 1 ! next node id
    do while (shead.ne.0)
       ! Get node from top of stack
       node = stack(shead)
       shead = shead - 1

       ! Number it
       map(node) = id
       id = id - 1

       ! Place all its children on the stack such that the last child is
       ! at the top of the stack and first child closest to the bottom
       if (node .eq. (n+1)) then
          ! Virtual root node, detect children with no entries at same time
          ! placing those that are empty at the top of the stack
          ! First do those which are proper roots
          i = chead(node)
          do while (i .ne. -1)
             if ((ptr(invp(i)+1)-ptr(invp(i))) .eq. 0) then
                i = cnext(i)
                cycle
             end if
             shead = shead + 1
             stack(shead) = i
             i = cnext(i)
          end do
          ! Second do those which are null roots
          i = chead(node)
          do while (i .ne. -1)
             if ((ptr(invp(i)+1)-ptr(invp(i))) .ne. 0) then
                i = cnext(i)
                cycle
             end if
             realn = realn - 1
             shead = shead + 1
             stack(shead) = i
             i = cnext(i)
          end do
       else ! A normal node
          i = chead(node)
          do while (i .ne. -1)
             shead = shead + 1
             stack(shead) = i
             i = cnext(i)
          end do
       end if
    end do

    !
    ! Apply map to perm, invp and parent
    !

    ! invp is straight forward, use stack as a temporary
    stack(1:n) = invp(1:n)
    do i = 1, n
       j = map(i)
       invp(j) = stack(i)
    end do

    ! perm can be easily done as the inverse of invp
    do i = 1, n
       perm(invp(i)) = i
    end do

    ! parent is done in two stages. The first copies it to stack and permutes
    ! parent(i), but not the locations. i.e. if 1 is a parent of 3, and
    ! map(1)=2 and map(3)=4, then the first stage sets stack(1) = 4.
    ! The second stage then permutes the entries of map back into parent
    do i = 1, n
       stack(i) = map(parent(i))
    end do
    do i = 1, n
       parent(map(i)) = stack(i)
    end do
  end subroutine find_postorder

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Column count routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! This subroutine determines column counts given the elimination tree and
! pattern of the matrix PAP^T.
!
! The algorithm is a specialisation of that given by Gilbert, Ng and Peyton [1],
! to only determine column counts. It is also described in Section 4.4 "Row
! counts" of [2].
!
! The essential technique is to determine the net number of entries introduced
! at a node (the "weight" in [1]). This is composed over the following terms:
!  wt[i] = [ - #children of node i
!            - #common indices between children
!            + #additional "new" row indices from column of A ]
!
! The clever part of this algorithm is how to determine the number of common
! indices between the children. This is accomplished by storing the last column
! at which an index was encountered, and a partial elimination tree. This
! partial elimination tree consists of all nodes processed so far, plus their
! parents. As we have a postorder on the tree, the current top of the tree
! containing node i is the least common ancestor of node i and the current node.
! We then observe that the first time an index will be double counted is at the
! least common ancestor of the current node and the last node where it was
! encountered.
!
! [1] Gilbert, Ng, Peyton, "An efficient algorithm to compute row and column
!     counts for sparse Cholesky factorization", SIMAX 15(4) 1994.
!
! [2] Tim Davis's book "Direct Methods for Sparse Linear Systems", SIAM 2006.
!
  subroutine find_col_counts(n, ptr, row, perm, invp, parent, cc, st)
    implicit none
    integer, intent(in) :: n ! dimension of system
    integer(ptr_kind), dimension(n+1), intent(in) :: ptr ! column pointers of A
    integer, dimension(ptr(n+1)-1), intent(in) :: row ! row indices of A
    integer, dimension(n), intent(in) :: perm ! perm(i) is the pivot
      ! position of column i
    integer, dimension(n), intent(in) :: invp ! inverse of perm
    integer, dimension(n), intent(in) :: parent ! parent(i) is the
      ! parent of pivot i in the elimination tree
    integer, dimension(n+1), intent(out) :: cc ! On exit, cc(i) is the
      ! number of entries in the lower triangular part of L (includes diagonal)
      ! for the column containing pivot i. For most of the routine however, it
      ! is used as a work space to track the net number of entries appearing
      ! for the first time at node i of the elimination tree (this may be
      ! negative).
    integer, intent(out) :: st ! stat parmeter for allocate calls

    integer :: col ! column of matrix associated with piv
    integer, dimension(:), allocatable :: first ! first descendants
    integer :: i
    integer(ptr_kind) :: ii
    integer, dimension(:), allocatable :: last_nbr ! previous neighbour
    integer, dimension(:), allocatable :: last_p ! previous p?
    integer :: par ! parent node of piv
    integer :: piv ! current pivot
    integer :: pp ! last pivot where u was encountered
    integer :: lca ! least common ancestor of piv and pp
    integer :: u ! current entry in column col
    integer :: uwt ! weight of u
    integer, dimension(:), allocatable :: vforest ! virtual forest

    !
    ! Determine first descendants, and set cc = 1 for leaves and cc = 0 for
    ! non-leaves.
    !
    allocate(first(n+1), stat=st)
    if (st .ne. 0) return
    do i = 1, n+1
       first(i) = i
    end do
    do i = 1, n
       par = parent(i)
       first(par) = min(first(i), first(par)) ! first descendant
       if (first(i) .eq. i) then ! is it a leaf or not?
          cc(i) = 1
       else
          cc(i) = 0
       end if
    end do
    cc(n+1) = n + 1 ! Set to non-physical value

    !
    ! We store the partial elimination trees in a virtual forest. It is
    ! initialised such that each node is in its own tree to begin with.
    !
    allocate(vforest(n+1), stat=st)
    if (st .ne. 0) return
    vforest(:) = 0

    !
    ! Initialise previous pivot and neightbour arrays to indicate no previous
    ! pivot or neightbour.
    !
    allocate(last_p(n+1), last_nbr(n+1), stat=st)
    if (st .ne. 0) return
    last_p(:) = 0
    last_nbr(:) = 0

    !
    ! Determine cc(i), the number of net new entries to pass up tree from
    ! node i.
    !
    do piv = 1, n
       ! Loop over entries in column below the diagonal
       col = invp(piv)
       do ii = ptr(col), ptr(col+1)-1
          u = perm(row(ii))
          if (u .le. piv) cycle ! not in lower triangular part

          ! Check if entry has been seen by a descendant of this pivot, if
          ! so we skip the tests that would first add one to the current
          ! pivot's weight before then subtracting it again.
          if (first(piv) .gt. last_nbr(u)) then
             ! Count new entry in current column
             uwt = 1
             cc(piv) = cc(piv) + uwt

             ! Determine least common ancestor of piv and the node at which
             ! u was last encountred
             pp = last_p(u)
             if (pp .ne. 0) then
                ! u has been seen before, find top of partial elimination
                ! tree for node pp
                lca = FIND(vforest, pp)
                ! prevent double counting of u at node lca
                cc(lca) = cc(lca) - uwt
             end if

             ! Update last as u has now been seen at piv.
             last_p(u) = piv
          end if

          ! Record last neighbour of u so we can determine if it has been
          ! seen in this subtree before
          last_nbr(u) = piv
       end do
       ! Pass uneliminated variables up to parent
       par = parent(piv)
       cc(par) = cc(par) + cc(piv) - 1

       ! place the parent of piv into the same partial elimination tree as piv
       vforest(piv) = par ! operation "UNION" from [1]
    end do
  end subroutine find_col_counts

  ! Return top most element of tree containing u.
  ! Implements path compression to speed up subsequent searches.
  integer function FIND(vforest, u)
    implicit none
    integer, dimension(:), intent(inout) :: vforest
    integer, intent(in) :: u

    integer :: current, prev

    prev = -1
    current = u
    do while (vforest(current) .ne. 0)
       prev = current
       current = vforest(current)
       if (vforest(current) .ne. 0) vforest(prev) = vforest(current)
    end do

    FIND = current
  end function FIND

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Supernode amalgamation routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! This subroutine identifies (relaxed) supernodes from the elimination tree
! and column counts.
!
! A node, u, and its parent, v, are merged if:
! (a) No new fill-in is introduced i.e. cc(v) = cc(u)-1
! (b) The number of columns in both u and v is less than nemin
!
! Note: assembly tree must be POSTORDERED on output
  subroutine find_supernodes(n, realn, parent, cc, sperm, nnodes, sptr, sparent, &
       scc, nemin, info, st)
    integer, intent(in) :: n
    integer, intent(in) :: realn
    integer, dimension(n), intent(in) :: parent ! parent(i) is the
      ! parent of supernode i in the elimination/assembly tree.
    integer, dimension(n), intent(in) :: cc ! cc(i) is the column count
      ! of supernode i, including elements eliminated at supernode i.
    integer, dimension(n), intent(out) :: sperm ! on exit contains a permutation
      ! from pivot order to a new pivot order with contigous supernodes
    integer, intent(out) :: nnodes ! number of supernodes
    integer, dimension(n+1), intent(out) :: sptr
    integer, dimension(n), intent(out) :: sparent
    integer, dimension(n), intent(out) :: scc
    integer, intent(in) :: nemin
    integer, intent(inout) :: info
    integer, intent(out) :: st ! stat paremter from allocate calls

    integer :: i, j, k
    integer, dimension(:), allocatable :: height ! used to track height of tree
    logical, dimension(:), allocatable :: mark ! flag array for nodes to finalise
    integer, dimension(:), allocatable :: map ! map vertex idx -> supernode idx
    integer, dimension(:), allocatable :: nelim ! number of eliminated variables
    integer, dimension(:), allocatable :: nvert ! number of elimd supervariables
    integer :: node
    integer, dimension(:), allocatable :: npar ! temporary array of snode pars
    integer :: par ! parent of current node
    integer :: shead ! current head of stack
    integer, dimension(:), allocatable :: stack ! used to navigate tree
    integer :: v
    integer, dimension(:), allocatable :: vhead ! heads of vertex linked lists
    integer, dimension(:), allocatable :: vnext ! next element in linked lists
    integer(long), dimension(:), allocatable :: ezero ! number of explicit zeros
    integer, dimension(:), allocatable :: chead ! chead(i) is first child of i
    integer, dimension(:), allocatable :: cnext ! cnext(i) is next child of i
    integer, dimension(:), allocatable :: child
    integer :: nchild
    integer :: start ! First pivot in block pivot
    integer :: totalwt ! sum of weights

    !
    ! Initialise supernode representation
    !
    allocate(nelim(n+1), nvert(n+1), vhead(n+1), vnext(n+1), stack(n), &
      height(n+1), mark(n), stat=st)
    if (st .ne. 0) goto 490
    vnext(:) = -1
    vhead(:) = -1
    height(:) = 1

    ! Initialise number of variables in each node
    nelim(1:n+1) = 1
    totalwt = n
    nvert(1:n+1) = 1 ! Note: Explicit bounds used to silence warning

    allocate(map(n+1), npar(n+1), ezero(n+1), stat=st)
    if (st .ne. 0) goto 490

    ezero(:) = 0 ! Initially no explicit zeros
    ezero(n+1) = huge(ezero) ! ensure root is not merged

    ! Ensure virtual root never gets amlgamated
    nelim(n+1) = totalwt+1 + nemin

    !
    ! Build child linked lists for nodes; merge block pivots if needed
    !
    allocate(chead(n+1), cnext(n+1), child(n), stat=st)
    if (st .ne. 0) goto 490
    chead(:) = -1 ! no parent if necessary
    do i = realn, 1, -1 ! do in reverse order so come off in original order
       j = parent(i)
       cnext(i) = chead(j)
       chead(j) = i
    end do

    !
    ! Merge supernodes.
    !
    v = 1
    nnodes = 0
    start=n+2
    do par = 1, n+1

       nchild = 0
       node = chead(par)
       do while (node .ne. -1)
          nchild = nchild + 1
          child(nchild) = node
          node = cnext(node)
       end do
       call sort_by_val(nchild, child, cc, st)
       if (st .ne. 0) goto 490

       do j = 1, nchild
          node = child(j)
          if (do_merge(node, par, nelim, cc, ezero, nemin)) then
             ! Merge contents of node into par. Delete node.
             call merge_nodes(node, par, nelim, nvert, vhead, vnext, height, &
                  ezero, cc)
             mark(node) = .false.
          else
             mark(node) = .true.
          end if
       end do

    end do

    do node = 1, realn
       if (.not. mark(node)) cycle
       ! Node not merged, now a complete supernode

       ! Record start of supernode
       nnodes = nnodes + 1
       sptr(nnodes) = v
       npar(nnodes) = parent(node)
       scc(nnodes) = cc(node) + nelim(node) - 1

       ! Record height in tree of parent vertices
       height(parent(node)) = max(height(parent(node)), height(node) + 1)

       ! Determine last vertex of node so we can number backwards
       v = v + nvert(node)
       k = v

       ! Loop over member vertices of node and number them
       shead = 1
       stack(shead) = node
       do while (shead .gt. 0)
          i = stack(shead)
          shead = shead - 1

          ! Order current vertex
          k = k - 1
          sperm(i) = k
          map(i) = nnodes

          ! Stack successor, if any
          if (vnext(i) .ne. -1) then
             shead = shead + 1
             stack(shead) = vnext(i)
          end if

          ! Descend into tree rooted at i
          if (vhead(i) .ne. -1) then
             shead = shead + 1
             stack(shead) = vhead(i)
          end if
       end do
    end do
    sptr(nnodes+1) = v ! Record end of final supernode
    map(n+1) = nnodes + 1 ! virtual root vertex maps to virtual root sn
    npar(nnodes+1) = n + 1

    ! Handle permutation of empty columns
    do i = realn+1, n
       sperm(i) = i
    end do

    ! Allocate arrays for return and copy data into them correctly
    do node = 1, nnodes
       par = npar(node) ! parent /vertex/ of supernode
       par = map(par)   ! parent /node/   of supernode
       sparent(node) = par ! store parent
    end do

    return

490 continue
    info = ERROR_ALLOCATION
    return
  end subroutine find_supernodes

!
! Sort n items labelled by idx into decreasing order of val(idx(i))
!
  recursive subroutine sort_by_val(n, idx, val, st)
    implicit none
    integer, intent(in) :: n
    integer, dimension(n), intent(inout) :: idx
    integer, dimension(:), intent(in) :: val
    integer, intent(out) :: st

    integer :: ice_idx, ice_val, ik_idx, ik_val
    integer :: klo,kor,k,kdummy

    st = 0

    if (n .ge. minsz_ms) then
       call sort_by_val_ms(n, idx, val, st)
    else
       klo = 2
       kor = n
       do kdummy = klo, n
          ! items kor, kor+1, .... ,nchild are in order
          ice_idx = idx(kor-1)
          ice_val = val(ice_idx)
          do k = kor, n
             ik_idx = idx(k)
             ik_val = val(ik_idx)
             if (ice_val >= ik_val) exit
             idx(k-1) = ik_idx
          end do
          idx(k-1) = ice_idx
          kor = kor - 1
       end do
    end if
  end subroutine sort_by_val

! Sort n items labelled by idx into decreasing order of val(idx(i))
!
! Merge sort version, dramatically improves performance for nodes with large
! numbers of children
! (Passes to simple sort for small numbers of entries)
  recursive subroutine sort_by_val_ms(n, idx, val, st)
    implicit none
    integer, intent(in) :: n
    integer, dimension(n), intent(inout) :: idx
    integer, dimension(:), intent(in) :: val
    integer, intent(out) :: st

    integer :: i, j, jj, jj2, k, kk, kk2
    integer :: mid
    integer, dimension(:), allocatable :: work

    if (n .le. 1) return
    if (n .lt. minsz_ms) then
       call sort_by_val(n, idx, val, st)
       return
    end if
    mid = (n-1)/2 + 1

    ! Recurse to order half lists
    call sort_by_val_ms(mid, idx(1:mid), val, st)
    if (st .ne. 0) return
    call sort_by_val_ms(n - mid, idx(mid+1:n), val, st)
    if (st .ne. 0) return

    ! Merge two half lists
    ! (Take a copy of the first half list so we don't overwrite it)
    allocate(work(mid), stat=st)
    if (st .ne. 0) return
    work(:) = idx(1:mid)
    j = 1
    k = mid+1
    jj = work(j)
    jj2 = val(jj)
    kk = idx(k)
    kk2 = val(kk)
    do i = 1, n
       if (jj2 .ge. kk2) then
          idx(i) = jj
          j = j + 1
          if(j .gt. mid) exit
          jj = work(j)
          jj2 = val(jj)
       else
          idx(i) = kk
          k = k + 1
          if (k .gt. n) exit
          kk = idx(k)
          kk2 = val(kk)
       end if
    end do
    if (j .le. mid) idx(i+1:n) = work(j:mid)
  end subroutine sort_by_val_ms

!
! Return .true. if we should merge node and par, .false. if we should not
!
  logical function do_merge(node, par, nelim, cc, ezero, nemin)
    implicit none
    integer, intent(in) :: node ! node to merge and delete
    integer, intent(in) :: par ! parent to merge into
    integer, dimension(:), intent(in) :: nelim
    integer, dimension(:), intent(in) :: cc
    integer(long), dimension(:), intent(in) :: ezero
    integer, intent(in) :: nemin

    if (ezero(par) .eq. huge(ezero)) then
       do_merge = .false.
       return
    end if

    do_merge = ((cc(par) .eq. (cc(node)-1)) .and. (nelim(par) .eq. 1)) .or. &
         ((nelim(par) .lt. nemin) .and. (nelim(node) .lt. nemin))
  end function do_merge

!
! This subroutine merges node with its parent, deleting node in the process.
!
  subroutine merge_nodes(node, par, nelim, nvert, vhead, vnext, height, ezero, cc)
    implicit none
    integer, intent(in) :: node ! node to merge and delete
    integer, intent(in) :: par ! parent to merge into
    integer, dimension(:), intent(inout) :: nelim
    integer, dimension(:), intent(inout) :: nvert
    integer, dimension(:), intent(inout) :: vhead
    integer, dimension(:), intent(inout) :: vnext
    integer, dimension(:), intent(inout) :: height
    integer(long), dimension(:), intent(inout) :: ezero
    integer, dimension(:), intent(in) :: cc

    ! Add node to list of children merged into par
    vnext(node) = vhead(par)
    vhead(par) = node

    ! Work out number of explicit zeros in new node
    ezero(par) = ezero(par) + ezero(node) + &
         (cc(par)-1+nelim(par) - cc(node) + 1_long) * nelim(par)

    ! Add together eliminated variables
    nelim(par) = nelim(par) + nelim(node)
    nvert(par) = nvert(par) + nvert(node)

    ! nodes have same height
    height(par) = max(height(par), height(node))
  end subroutine merge_nodes

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Statistics routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! This subroutine merely calculates interesting statistics
!
  subroutine calc_stats(nnodes, sptr, scc, nfact, nflops)
    implicit none
    integer, intent(in) :: nnodes
    integer, dimension(nnodes+1), intent(in) :: sptr
    integer, dimension(nnodes), intent(in) :: scc
    integer(long), optional, intent(out) :: nfact
    integer(long), optional, intent(out) :: nflops

    integer :: j
    integer :: m ! number of entries in retangular part of ndoe
    integer :: nelim ! width of node
    integer :: node ! current node of assembly tree
    integer(long) :: r_nfact, r_nflops

    if ((.not. present(nfact)) .and. (.not. present(nflops))) return ! nothing to do

    r_nfact = 0
    r_nflops = 0
    do node = 1, nnodes
       nelim = sptr(node+1) - sptr(node)
       m = scc(node) - nelim

       ! number of entries
       r_nfact = r_nfact + (nelim * (nelim+1)) / 2 ! triangular block
       r_nfact = r_nfact + nelim * m ! below triangular block

       ! flops
       do j = 1, nelim
          r_nflops = r_nflops + (m+j)**2
       end do
    end do

    if (present(nfact)) nfact = r_nfact
    if (present(nflops)) nflops = r_nflops

    !print *, "n = ", n
    !print *, "nnodes = ", nnodes
    !print *, "nfact = ", nfact
    !print *, "sum cc=", sum(cc(1:n))
    !print *, "nflops = ", nflops
  end subroutine calc_stats

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Row list routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! This subroutine determines the row indices for each supernode
!
  subroutine find_row_lists(n, ptr, row, perm, invp, nnodes, &
       sptr, sparent, scc, rptr, rlist, info, st)
    implicit none
    integer, intent(in) :: n
    integer(ptr_kind), dimension(n+1), intent(in) :: ptr
    integer, dimension(ptr(n+1)-1), intent(in) :: row
    integer, dimension(n), intent(in) :: perm
    integer, dimension(n), intent(in) :: invp
    integer, intent(in) :: nnodes
    integer, dimension(nnodes+1), intent(in) :: sptr
    integer, dimension(nnodes), intent(in) :: sparent
    integer, dimension(nnodes), intent(in) :: scc
    integer(long), dimension(nnodes+1), intent(out) :: rptr
    integer, dimension(sum(scc(1:nnodes))), intent(out) :: rlist
    integer, intent(inout) :: info
    integer, intent(out) :: st

    integer :: child ! current child of node
    integer :: col ! current column of matrix corresponding to piv
    integer(long) :: i
    integer(long) :: idx ! current insert position into nodes(node)%index
    integer :: j
    integer :: node ! current node of assembly tree
    integer :: piv ! current pivot position
    integer, dimension(:), allocatable :: seen ! tag last time index was seen
    integer, dimension(:), allocatable :: chead ! head of child linked lists
    integer, dimension(:), allocatable :: cnext ! pointer to next child

    ! Allocate and initialise memory
    allocate(seen(n), chead(nnodes+1), cnext(nnodes+1), stat=st)
    if (st .ne. 0) then
       info = ERROR_ALLOCATION
       return
    end if
    seen(:) = 0
    chead(:) = -1

    ! Build child linked lists (backwards so pop off in good order)
    do node = nnodes, 1, -1
       i = sparent(node)
       cnext(node) = chead(i)
       chead(i) = node
    end do

    ! Loop over nodes from bottom up building row lists.
    rptr(1) = 1
    do node = 1, nnodes

       ! Allocate space for row indices
       rptr(node+1) = rptr(node) + scc(node)
       idx = rptr(node) ! insert position

       ! Add entries eliminated at this node
       do piv = sptr(node), sptr(node+1)-1
          seen(piv) = node
          rlist(idx) = piv
          idx = idx + 1
       end do

       ! Find indices inherited from children
       child = chead(node)
       do while (child .ne. -1)
          do i = rptr(child), rptr(child+1)-1
             j = rlist(i)
             if (j .lt. sptr(node)) cycle ! eliminated
             if (seen(j) .eq. node) cycle ! already seen
             seen(j) = node
             rlist(idx) = j
             idx = idx + 1
          end do
          child = cnext(child)
       end do

       ! Find new indices from A
       do piv = sptr(node), sptr(node+1)-1
          col = invp(piv)
          do i = ptr(col), ptr(col+1)-1
             j = perm(row(i))
             if (j .lt. piv) cycle ! in upper triangle
             if (seen(j) .eq. node) cycle ! already seen in this snode
             ! Otherwise, this is a new entry
             seen(j) = node
             rlist(idx) = j
             idx = idx + 1
          end do
       end do
    end do
  end subroutine find_row_lists

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Assorted auxilary routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! This subroutine performs a double transpose sort on the row indices of sn
!
  subroutine dbl_tr_sort(n, nnodes, rptr, rlist, st)
    implicit none
    integer, intent(in) :: n
    integer, intent(in) :: nnodes
    integer(long), dimension(nnodes+1), intent(in) :: rptr
    integer, dimension(rptr(nnodes+1)-1), intent(inout) :: rlist
    integer, intent(out) :: st

    integer :: node
    integer :: i, j
    integer(long) :: ii, jj
    integer(long), dimension(:), allocatable :: ptr
    integer(long), dimension(:), allocatable :: nptr
    integer, dimension(:), allocatable :: col

    allocate(ptr(n+2), stat=st)
    if (st .ne. 0) return
    ptr(:) = 0

    ! Count number of entries in each row. ptr(i+2) = #entries in row i
    do node = 1, nnodes
       do ii = rptr(node), rptr(node+1)-1
          j = rlist(ii) ! row entry
          ptr(j+2) = ptr(j+2) + 1
       end do
    end do

    ! Determine row starts. ptr(i+1) = start of row i
    ptr(1:2) = 1
    do i = 1, n
       ptr(i+2) = ptr(i+1) + ptr(i+2)
    end do

    jj = ptr(n+2)-1 ! total number of entries
    allocate(col(jj), stat=st)
    if (st .ne. 0) return

    ! Now fill in col array
    do node = 1, nnodes
       do ii = rptr(node), rptr(node+1)-1
          j = rlist(ii) ! row entry
          col( ptr(j+1) ) = node
          ptr(j+1) = ptr(j+1) + 1
       end do
    end do

    ! Finally transpose back into nodes
    allocate(nptr(nnodes), stat=st)
    if (st .ne. 0) return
    nptr(:) = rptr(1:nnodes)
    do i = 1, n
       do jj = ptr(i), ptr(i+1)-1
          node = col(jj)
          rlist(nptr(node)) = i
          nptr(node) = nptr(node) + 1
       end do
    end do
  end subroutine dbl_tr_sort

!
! This subroutine applies the permutation perm to order, invp and cc
!
  subroutine apply_perm(n, perm, order, invp, cc)
    implicit none
    integer, intent(in) :: n
    integer, dimension(n), intent(in) :: perm
    integer, dimension(n), intent(inout) :: order
    integer, dimension(n), intent(inout) :: invp
    integer, dimension(n), intent(inout) :: cc

    integer :: i
    integer :: j

    ! Use order as a temporary variable to permute cc. Don't care about cc(n+1)
    order(1:n) = cc(1:n)
    do i = 1, n
       j = perm(i)
       cc(j) = order(i)
    end do

    ! Use order as a temporary variable to permute invp.
    order(1:n) = invp(1:n)
    do i = 1, n
       j = perm(i)
       invp(j) = order(i)
    end do

    ! Recover order as inverse of invp
    do i = 1, n
       order(invp(i)) = i
    end do
  end subroutine apply_perm

end module spral_core_analyse
