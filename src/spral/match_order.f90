! COPYRIGHT (c) 2012-3 Science and Technology Facilities Council
! Authors: Jonathan Hogg and Jennifer Scott
! Note: This code is a heavily modified version of HSL_MC80

! Given a sparse symmetric  matrix A, this module provides routines to
! use a matching algorithm to compute an elimination
! order that is suitable for use with a sparse direct solver.
! It optionally computes scaling factors.

! FIXME: At some stage replace call to mo_match() with call to
! a higher level routine from spral_scaling instead (NB: have to cope with
! fact we are currently expecting a full matrix, even if it means 2x more log
! operations)

module spral_match_order

  use spral_metis_wrapper, only : metis_order
  use spral_scaling, only : hungarian_match
  implicit none

  private
  public :: match_order_metis ! Find a matching-based ordering using the
    ! Hungarian algorithm for matching and METIS for ordering.

  integer, parameter :: wp = kind(0d0)
  integer, parameter :: long = selected_int_kind(18)

  ! Error flags
  integer, parameter :: SUCCESS               = 0
  integer, parameter :: ERROR_ALLOCATION      = -1
  integer, parameter :: ERROR_A_N_OOR         = -2
  integer, parameter :: ERROR_UNKNOWN         = -99

  ! warning flags
  integer, parameter :: WARNING_SINGULAR      = 1

  interface match_order_metis
     module procedure match_order_metis_ptr32, match_order_metis_ptr64
  end interface match_order_metis

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! On input ptr, row , val hold the ** lower AND upper ** triangular
! parts of the matrix.
! this reduces amount of copies of matrix required (so slightly
! more efficient on memory and does not need to expand supplied matrix)
!
  subroutine match_order_metis_ptr32(n, ptr, row, val, order, scale, &
       flag, stat)
    implicit none
    integer, intent(in) :: n
    integer, dimension(:), intent(in) :: ptr
    integer, dimension(:), intent(in) :: row
    real(wp), dimension(:), intent(in) :: val
    integer, dimension(:), intent(out) :: order ! order(i)  holds the position
      ! of variable i in the elimination order (pivot sequence).
    real(wp), dimension(n), intent(out) :: scale ! returns the mc64 symmetric
      ! scaling
    integer, intent(out) :: flag ! return value
    integer, intent(out) :: stat ! stat value returned on failed allocation

    integer, dimension(:), allocatable :: cperm ! used to hold matching
    integer(long), dimension(:), allocatable :: ptr2 ! column pointers for
      ! expanded matrix.
    integer, dimension(:), allocatable :: row2 ! row indices for expanded matrix
    real(wp), dimension(:), allocatable :: val2 ! entries of expanded matrix.

    integer :: i, j, ne
    integer(long) :: k

    flag = 0
    stat = 0

    ! check n has valid value
    if (n .lt. 0) then
       flag = ERROR_A_N_OOR
       return
    end if

    ! just return with no action if n = 0
    if (n .eq. 0) return

    !
    ! Remove any explicit zeroes and take absolute values
    !
    ne = ptr(n+1) - 1
    allocate(ptr2(n+1), row2(ne), val2(ne), cperm(n), stat=stat)
    if (stat .ne. 0) then
       flag = ERROR_ALLOCATION
       return
    end if

    k = 1
    do i = 1, n
       ptr2(i) = k
       do j = ptr(i), ptr(i+1)-1
          if (val(j) .eq. 0.0) cycle
          row2(k) = row(j)
          val2(k) = abs(val(j))
          k = k + 1
       end do
    end do
    ptr2(n+1) = k

    ! Compute matching and scaling

    call mo_scale(n,ptr2,row2,val2,scale,flag,stat,perm=cperm)
    deallocate(val2, stat=stat)

    if (flag .lt. 0) return

    ! Note: row j is matched with column cperm(j)
    ! write (*,'(a,15i4)') 'cperm',cperm(1:min(15,n))
    !
    ! Split matching into 1- and 2-cycles only and then
    ! compress matrix and order.

    call mo_split(n,row2,ptr2,order,cperm,flag,stat)

    scale(1:n) = exp( scale(1:n) )

  end subroutine match_order_metis_ptr32

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! On input ptr, row , val hold the ** lower AND upper ** triangular
! parts of the matrix.
! this reduces amount of copies of matrix required (so slightly
! more efficient on memory and does not need to expand supplied matrix)
!
  subroutine match_order_metis_ptr64(n, ptr, row, val, order, scale, &
       flag, stat)
    implicit none
    integer, intent(in) :: n
    integer(long), dimension(:), intent(in) :: ptr
    integer, dimension(:), intent(in) :: row
    real(wp), dimension(:), intent(in) :: val
    integer, dimension(:), intent(out) :: order ! order(i)  holds the position
      ! of variable i in the elimination order (pivot sequence).
    real(wp), dimension(n), intent(out) :: scale ! returns the mc64 symmetric
      ! scaling
    integer, intent(out) :: flag ! return value
    integer, intent(out) :: stat ! stat value returned on failed allocation

    integer, dimension(:), allocatable :: cperm ! used to hold matching
    integer(long), dimension(:), allocatable :: ptr2 ! column pointers for
      ! expanded matrix.
    integer, dimension(:), allocatable :: row2 ! row indices for expanded matrix
    real(wp), dimension(:), allocatable :: val2 ! entries of expanded matrix.

    integer :: i
    integer(long) :: j, k, ne

    flag = 0
    stat = 0

    ! check n has valid value
    if (n .lt. 0) then
       flag = ERROR_A_N_OOR
       return
    end if

    ! just return with no action if n = 0
    if (n .eq. 0) return

    !
    ! Remove any explicit zeroes and take absolute values
    !
    ne = ptr(n+1) - 1
    allocate(ptr2(n+1), row2(ne), val2(ne), cperm(n), stat=stat)
    if (stat .ne. 0) then
       flag = ERROR_ALLOCATION
       return
    end if

    k = 1
    do i = 1, n
       ptr2(i) = k
       do j = ptr(i), ptr(i+1)-1
          if (val(j) .eq. 0.0) cycle
          row2(k) = row(j)
          val2(k) = abs(val(j))
          k = k + 1
       end do
    end do
    ptr2(n+1) = k

    ! Compute matching and scaling

    call mo_scale(n,ptr2,row2,val2,scale,flag,stat,perm=cperm)
    deallocate(val2, stat=stat)

    if (flag .lt. 0) return

    ! Note: row j is matched with column cperm(j)
    ! write (*,'(a,15i4)') 'cperm',cperm(1:min(15,n))
    !
    ! Split matching into 1- and 2-cycles only and then
    ! compress matrix and order.

    call mo_split(n,row2,ptr2,order,cperm,flag,stat)

    scale(1:n) = exp( scale(1:n) )
  end subroutine match_order_metis_ptr64

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! Split matching into 1- and 2-cycles only and then
! compress matrix and order using mc68.
!
! Input (ptr2, row2 , val2) holds the ** lower and upper triangles **
! of the matrix (with explicit zeros removed).
! Overwritten in the singular case
!
  subroutine mo_split(n,row2,ptr2,order,cperm,flag,stat)
    implicit none
    integer, intent(in) :: n
    integer(long), dimension(:), intent(in) :: ptr2
    integer, dimension(:), intent(in) :: row2
    integer, dimension(n), intent(out) :: order ! used to hold ordering
    integer, dimension(n), intent(inout) :: cperm ! used to hold matching
    integer, intent(inout) :: flag
    integer, intent(inout) :: stat

    integer, dimension(:), allocatable :: iwork ! work array
    integer, dimension(:), allocatable :: old_to_new, new_to_old
      ! holds mapping between original matrix indices and those in condensed
      ! matrix.
    integer, dimension(:), allocatable :: ptr3 ! column pointers for condensed
      ! matrix.
    integer, dimension(:), allocatable :: row3 ! row indices for condensed
      ! matrix.

    integer :: csz ! current cycle length
    integer :: i, j, j1, j2, jj, k, krow, metis_flag
    integer(long) :: klong
    integer :: max_csz ! maximum cycle length
    integer :: ncomp ! order of compressed matrix
    integer :: ncomp_matched ! order of compressed matrix (matched entries only)
    integer(long) :: ne ! number of non zeros
    integer, dimension(:), allocatable :: invp

   ! Use iwork to track what has been matched:
   ! -2 unmatched
   ! -1 matched as singleton
   !  0 not yet seen
   ! >0 matched with specified node

    ne = ptr2(n+1) - 1
    allocate(ptr3(n+1), row3(ne), old_to_new(n), new_to_old(n), iwork(n), &
         stat=stat)
    if (stat .ne. 0) return

    iwork(1:n) = 0
    max_csz = 0
    do i = 1, n
       if (iwork(i) .ne. 0) cycle
       j = i
       csz = 0
       do
          if (cperm(j) .eq. -1) then
             ! unmatched by MC64
             iwork(j) = -2
             csz = csz + 1
             exit
          else if (cperm(j) .eq. i) then
             ! match as singleton, unmatched or finished
             iwork(j) = -1
             csz = csz + 1
             exit
          end if
          ! match j and cperm(j)
          jj = cperm(j)
          iwork(j) = jj
          iwork(jj) = j
          csz = csz + 2
          ! move onto next start of pair
          j = cperm(jj)
          if (j .eq. i) exit
       end do
       max_csz = max(max_csz, csz)
    end do

    ! Overwrite cperm with new matching
    cperm(1:n) = iwork(1:n)

    !
    ! Build maps for new numbering schemes
    !
    k = 1
    do i = 1,n
       j = cperm(i)
       if ((j .lt. i) .and. (j .gt. 0)) cycle
       old_to_new(i) = k
       new_to_old(k) = i ! note: new_to_old only maps to first of a pair
       if (j .gt. 0) old_to_new(j) = k
       k = k + 1
    end do
    ncomp_matched = k-1

    !
    ! Produce a condensed version of the matrix for ordering.
    ! Hold pattern using ptr3 and row3.
    !
    ptr3(1) = 1
    iwork(:) = 0 ! Use to indicate if entry is in a paired column
    ncomp = 1
    jj = 1
    do i = 1, n
       j = cperm(i)
       if ((j .lt. i) .and. (j .gt. 0)) cycle ! already seen
       do klong = ptr2(i), ptr2(i+1)-1
          krow = old_to_new(row2(klong))
          if (iwork(krow) .eq. i) cycle ! already added to column
          if (krow .gt. ncomp_matched) cycle ! unmatched row not participating
          row3(jj) = krow
          jj = jj + 1
          iwork(krow) = i
       end do
       if (j .gt. 0) then
          ! Also check column cperm(i)
          do klong = ptr2(j), ptr2(j+1)-1
             krow = old_to_new(row2(klong))
             if (iwork(krow) .eq. i) cycle ! already added to column
             if (krow .gt. ncomp_matched) cycle ! unmatched row not participating
             row3(jj) = krow
             jj = jj + 1
             iwork(krow) = i
          end do
       end if
       ptr3(ncomp+1) = jj
       ncomp = ncomp + 1
    end do
    ncomp = ncomp - 1

    ! store just lower triangular part for input to hsl_mc68
    ptr3(1) = 1
    jj = 1
    j1 = 1
    do i = 1, ncomp
       j2 = ptr3(i+1)
       do k = j1, j2-1
          krow = row3(k)
          if (krow .lt. i) cycle ! already added to column
          row3(jj) = krow
          jj = jj + 1
       end do
       ptr3(i+1) = jj
       j1 = j2
    end do

    allocate(invp(ncomp), stat=stat)
    if (stat .ne. 0) return

    ! reorder the compressed matrix using metis.
    ! switch off metis printing
    call metis_order(ncomp,ptr3,row3,order,invp,metis_flag,stat)
    select case(metis_flag)
    case(0)
       ! OK, do nothing
    case(-1)
       ! Allocation error
       flag = ERROR_ALLOCATION
       return
    case default
       ! Unknown error, should never happen
       print *, "metis_order() returned unknown error ", metis_flag
       flag = ERROR_UNKNOWN
    end select

    do i = 1, ncomp
       j = order(i)
       iwork(j) = i
    end do

    !
    ! Translate inverse permutation in iwork back to
    ! permutation for original variables.
    !
    k = 1
    do i = 1, ncomp
       j = new_to_old( iwork(i) )
       order(j) = k
       k = k + 1
       if (cperm(j) .gt. 0) then
          j = cperm(j)
          order(j) = k
          k = k + 1
       end if
    end do
  end subroutine mo_split

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! Scale the matrix using MC64, accounting for singular matrices using the
! approach of Duff and Pralet
!
! Expects a full matrix as input
!
  subroutine mo_scale(n, ptr, row, val, scale, flag, stat, perm)
    implicit none
    integer, intent(in) :: n
    integer(long), dimension(:), intent(in) :: ptr
    integer, dimension(:), intent(in) :: row
    real(wp), dimension(:), intent(in) :: val
    real(wp), dimension(n), intent(out) :: scale ! returns the symmetric scaling
    integer, intent(inout) :: flag
    integer, intent(inout) :: stat
    integer, dimension(n), intent(out), optional :: perm ! if present, returns
      ! the matching

    integer(long), dimension(:), allocatable :: ptr2 ! column pointers after
      ! zeros removed.
    integer, dimension(:), allocatable :: row2 ! row indices after zeros
      !  removed.
    real(wp), dimension(:), allocatable :: val2 ! matrix of absolute values
      ! (zeros removed).
    real(wp), dimension(:), allocatable :: cscale ! temporary copy of scaling
      ! factors. Only needed if A rank deficient. allocated to have size n.

    integer :: struct_rank
    integer :: i
    integer(long) :: j, k, ne

    struct_rank = n

    !
    ! Remove any explicit zeroes and take absolute values
    !
    ne = ptr(n+1) - 1
    allocate(ptr2(n+1), row2(ne), val2(ne),stat=stat)
    if (stat .ne. 0) then
       flag = ERROR_ALLOCATION
       return
    end if

    k = 1
    do i = 1, n
       ptr2(i) = k
       do j = ptr(i), ptr(i+1)-1
          if (val(j) .eq. 0.0) cycle
          row2(k) = row(j)
          val2(k) = val(j)
          k = k + 1
       end do
    end do
    ptr2(n+1) = k

    call mo_match(n,row2,ptr2,val2,scale,flag,stat,perm=perm)
    if (flag .lt. 0) return

    if (struct_rank .ne. n) then
       ! structurally singular case. At this point, scaling factors
       ! for rows in corresponding to rank deficient part are set to
       ! zero. The following is to set them according to Duff and Pralet.
       deallocate(ptr2, stat=stat)
       deallocate(row2, stat=stat)
       deallocate(val2, stat=stat)
       allocate(cscale(n),stat=stat)
       if (stat .ne. 0) then
          flag = ERROR_ALLOCATION
          return
       end if
       cscale(1:n) = scale(1:n)
       do i = 1, n
          if (cscale(i) .ne. -huge(scale)) cycle
          do j = ptr(i), ptr(i+1)-1
             k = row(j)
             if (cscale(k) .eq. -huge(scale)) cycle
             scale(i) = max(scale(i), val(j)+scale(k))
          end do
          if (scale(i) .eq. -huge(scale)) then
             scale(i) = 0.0
          else
             scale(i) = -scale(i)
          end if
       end do
    end if
  end subroutine mo_scale

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! Input (ptr2, row2 , val2) holds the ** lower and upper triangles **
! of the matrix (with explicit zeros removed).
! val2 holds absolute values of matrix entries.
! Overwritten in the singular case
!
  subroutine mo_match(n,row2,ptr2,val2,scale,flag,stat,perm)
    implicit none
    integer, intent(in) :: n
    integer(long), dimension(:), intent(inout) :: ptr2 ! In singular case,
      ! overwritten by column pointers for non singular part of matrix.
    integer, dimension(:), intent(inout) :: row2 ! In singular case, overwritten
      ! by row indices for non singular part of matrix.
    real(wp), dimension(:), intent(inout) :: val2 ! In singular case, overwritten
      ! by entries for non singular part of matrix.
    real(wp), dimension(n), intent(out) :: scale ! returns the symmetric scaling
    integer, intent(inout) :: flag
    integer, intent(inout) :: stat
    integer, dimension(n), intent(out), optional :: perm ! if present, returns
      ! the matching

    integer, dimension(:), allocatable :: cperm ! used to hold matching
    integer, dimension(:), allocatable :: old_to_new, new_to_old
      ! holds mapping between original matrix indices and those in reduced
      ! non singular matrix.
    real(wp), dimension(:), allocatable :: cmax ! (log) column maximum
    real(wp), dimension(:), allocatable :: dw ! array used by mc64

    integer :: i, j, jj, k
    integer(long) :: jlong, j1, j2
    integer(long) :: ne ! number of non zeros
    integer :: nn ! Holds number of rows/cols in non singular part of matrix
    integer :: nne ! Only used in singular case. Holds number of non zeros
      ! in non-singular part of matrix.
    integer :: rank ! returned by mc64
    real(wp) :: colmax ! max. entry in col. of expanded matrix

    allocate(cperm(n), dw(2*n), cmax(n), stat=stat)
    if (stat .ne. 0) then
       flag = ERROR_ALLOCATION
       return
    end if

    ! Compute column maximums
    do i = 1, n
       colmax = max(0.0_wp,maxval(val2(ptr2(i):ptr2(i+1)-1)))
       if (colmax .ne. 0.0) colmax = log(colmax)
       cmax(i) = colmax
    end do

    do i = 1, n
       val2(ptr2(i):ptr2(i+1)-1) = cmax(i) - log(val2(ptr2(i):ptr2(i+1)-1))
    end do

    ne = ptr2(n+1)-1
    call hungarian_match(n,n,ptr2,row2,val2,cperm,rank,dw(1),dw(n+1),stat)
    if (stat .ne. 0) then
       flag = ERROR_ALLOCATION
       return
    end if

    if (rank .eq. n) then
       do i = 1, n
          scale(i) = (dw(i)+dw(n+i)-cmax(i))/2
       end do
       if (present(perm)) perm(1:n) = cperm(1:n)
       return
    end if

    !!!! we have to handle the singular case. Either immediate exit
    ! or set warning, squeeze out the unmatched entries and recall mc64wd.

    flag = WARNING_SINGULAR

    allocate(old_to_new(n), new_to_old(n),stat=stat)
    if (stat .ne. 0) then
       flag = ERROR_ALLOCATION
       return
    end if

    k = 0
    do i = 1, n
       if (cperm(i) .lt. 0) then
          ! row i and col j are not part of the matching
          old_to_new(i) = -1
       else
          k = k + 1
          ! old_to_new(i) holds the new index for variable i after
          ! removal of singular part and new_to_old(k) is the
          ! original index for k
          old_to_new(i) = k
          new_to_old(k) = i
       end if
    end do

    ! Overwrite ptr2, row2 and val2
    nne = 0
    k = 0
    ptr2(1) = 1
    j2 = 1
    do i = 1, n
       j1 = j2
       j2 = ptr2(i+1)
       ! skip over unmatched entries
       if (cperm(i) .lt. 0) cycle
       k = k + 1
       do jlong = j1, j2-1
          jj = row2(jlong)
          if (cperm(jj) .lt. 0) cycle
          nne = nne + 1
          row2(nne) = old_to_new(jj)
          val2(nne) = val2(jlong)
       end do
       ptr2(k+1) = nne + 1
    end do
    ! nn is order of non-singular part.
    nn = k
    call hungarian_match(nn,nn,ptr2,row2,val2,cperm,rank,dw(1),dw(nn+1),stat)
    if (stat .ne. 0) then
       flag = ERROR_ALLOCATION
       return
    end if

    do i = 1, n
       j = old_to_new(i)
       if (j .lt. 0) then
          scale(i) = -huge(scale)
       else
          ! Note: we need to subtract col max using old matrix numbering
          scale(i) = (dw(j)+dw(nn+j)-cmax(i))/2
       end if
    end do

    if (present(perm)) then
       perm(1:n) = -1
       do i = 1, nn
          j = cperm(i)
          perm(new_to_old(i)) = new_to_old(j)
       end do
    end if
  end subroutine mo_match

!**********************************************************************
end module spral_match_order
