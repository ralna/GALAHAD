! THIS VERSION: GALAHAD 4.3 - 2024-01-16 AT 10:20 GMT.

! COPYRIGHT (c) 2007-2013 Science & Technology Facilities Council
! Authors: Sue Thorne and Jonathan Hogg
! Origin: Heavily modified version of hsl_mc68
!

#ifdef INTEGER_64
#define METIS_NodeND_4 METIS_NodeND_4_64
#define spral_metis_wrapper spral_metis_wrapper_64
#define spral_kinds spral_kinds_64
#else
#define METIS_NodeND_4 METIS_NodeND_4
#endif

module spral_metis_wrapper

  use, intrinsic :: iso_c_binding
  use spral_kinds
  implicit none

  private
  public :: metis_order ! Calls metis on a symmetric matrix

  integer(ip_), parameter :: ERROR_ALLOC = -1
  integer(ip_), parameter :: ERROR_N_OOR = -2
  integer(ip_), parameter :: ERROR_NE_OOR = -3
! nimg added 2021-03-24
  integer(ip_), parameter :: ERROR_NO_METIS = -4

  interface metis_order
     module procedure metis_order32, metis_order64
  end interface metis_order

  INTERFACE
    SUBROUTINE METIS_NodeND( n, PTR, ROW, options, perm, iperm ) BIND( C )
      USE SPRAL_KINDS, ONLY: ipc_
      IMPLICIT NONE
      INTEGER( KIND = ipc_ ), INTENT( IN ) :: n
      INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: PTR, ROW
      INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: options
      INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( OUT ) :: perm, iperm
    END SUBROUTINE METIS_NodeND
  END INTERFACE

contains

!
! Fortran wrapper around metis
!
  subroutine metis_order32(n,ptr,row,perm,invp,flag,stat)
    implicit none
    integer(ip_), intent(in) :: n ! Must hold the number of rows in A
    integer(i4_), intent(in) :: ptr(n+1) ! ptr(j) holds position in row of start
      ! of row indices for column j. ptr(n)+1 must equal the number of entries
      ! stored + 1. Only the lower triangular entries are stored with no
      ! duplicates or out-of-range entries
    integer(ip_), intent(in) :: row(ptr(n+1)-1) ! size at least ptr(n+1)-1
    integer(ip_), intent(out) :: perm(n) ! Holds elimination order on output
    integer(ip_), intent(out) :: invp(n) ! Holds inverse of elimination order
      ! on exit
    integer(ip_), intent(out) :: flag ! Return value
    integer(ip_), intent(out) :: stat ! Stat value on allocation failure

    ! ---------------------------------------------
    ! Local variables
    ! ---------------------------------------------
    integer(ip_), allocatable :: ptr2(:) ! copy of pointers which is later
      ! modified
    integer(ip_), allocatable :: row2(:) ! copy of row indices
    integer(ip_) :: metis_opts(8) ! metis options array
    integer(i4_) :: iwlen ! length of iw

    ! Initialise flag and stat
    flag = 0
    stat = 0

    !
    ! Check that restrictions are adhered to
    !
    if (n .lt. 1) then
       flag = ERROR_N_OOR
       return
    end if

    if (n .eq. 1) then
       ! Matrix is of order 1 so no need to call ordering subroutines
       perm(1) = 1
       return
    end if

    ! Set length of iw
    iwlen = 2*ptr(n+1) - 2

    ! Allocate arrays
    allocate(ptr2(n+1),row2(iwlen),stat=stat)
    if (stat .ne. 0) then
       flag = ERROR_ALLOC
       return
    end if

    ! Expand matrix, dropping diagonal entries
    call half_to_full_drop_diag32_32(n, ptr, row, ptr2, row2)

   ! Carry out ordering
    metis_opts(1) = 0 ! MeTiS defaults
    call METIS_NodeND(n,ptr2,row2,1_ip_,metis_opts,invp,perm)
! nimg added 2021-03-24
    if (perm(1)<0) then
      flag = ERROR_NO_METIS
      return
    end if
  end subroutine metis_order32

  !
  ! Fortran wrapper around metis
  !
  subroutine metis_order64(n,ptr,row,perm,invp,flag,stat)
    implicit none
    integer(ip_), intent(in) :: n ! Must hold the number of rows in A
    integer(i8_), intent(in) :: ptr(n+1) ! ptr(j) holds position in row of
      ! start of row indices for column j. ptr(n)+1 must equal the number of
      ! entries stored + 1. Only the lower triangular entries are stored with
      ! no duplicates or out-of-range entries
    integer(ip_), intent(in) :: row(ptr(n+1)-1) ! size at least ptr(n+1)-1
    integer(ip_), intent(out) :: perm(n) ! Holds elimination order on output
    integer(ip_), intent(out) :: invp(n) ! Holds inverse of elimination order
      ! on exit
    integer(ip_), intent(out) :: flag ! Return value
    integer(ip_), intent(out) :: stat ! Stat value on allocation failure

    ! ---------------------------------------------
    ! Local variables
    ! ---------------------------------------------
    integer(ip_), allocatable :: ptr2(:) ! copy of pointers which is later
      ! modified
    integer(ip_), allocatable :: row2(:) ! copy of row indices
    integer(ip_) :: metis_opts(8) ! metis options array
    integer(i8_) :: iwlen ! length of iw

    ! Initialise flag and stat
    flag = 0
    stat = 0

    !
    ! Check that restrictions are adhered to
    !
    if (n .lt. 1) then
       flag = ERROR_N_OOR
       return
    end if

    if (n .eq. 1) then
       ! Matrix is of order 1 so no need to call ordering subroutines
       perm(1) = 1
       return
    endif

    ! Set length of iw
    iwlen = 2*ptr(n+1) - 2
    if (iwlen .gt. huge(ptr2)) then
       ! Can't accomodate this many entries with 32-bit interface to metis
       flag = ERROR_NE_OOR
       return
    end if

    ! Allocate arrays
    allocate(ptr2(n+1),row2(iwlen),stat=stat)
    if (stat .ne. 0) then
       flag = ERROR_ALLOC
       return
    end if

    ! Expand matrix, dropping diagonal entries
    call half_to_full_drop_diag64_32(n, ptr, row, ptr2, row2)

    ! Carry out ordering
    metis_opts(1) = 0 ! MeTiS defaults
    call METIS_NodeND(n,ptr2,row2,1_ip_,metis_opts,invp,perm)
! nimg added 2021-03-24
    if (perm(1)<0) then
      flag = ERROR_NO_METIS
      return
    end if
  end subroutine metis_order64

  ! Convert a matrix in half storage to one in full storage.
  ! Drops any diagonal entries.
  subroutine half_to_full_drop_diag32_32(n, ptr, row, ptr2, row2)
    implicit none
    integer(ip_), intent(in) :: n
    integer(i4_), dimension(n+1), intent(in) :: ptr
    integer(ip_), dimension(ptr(n+1)-1), intent(in) :: row
    integer(ip_), dimension(*), intent(out) :: ptr2
    integer(ip_), dimension(*), intent(out) :: row2

    integer(ip_) :: i, j
    integer(i4_) :: k

    ! Set ptr2(j) to hold no. nonzeros in column j
    ptr2(1:n+1) = 0
    do j = 1, n
       do k = ptr(j), ptr(j+1) - 1
          i = row(k)
          if (j .ne. i) then
             ptr2(i) = ptr2(i) + 1
             ptr2(j) = ptr2(j) + 1
          end if
       end do
    end do

    ! Set ptr2(j) to point to where row indices will end in row2
    do j = 2, n
       ptr2(j) = ptr2(j-1) + ptr2(j)
    end do
    ptr2(n+1) = ptr2(n) + 1

    ! Fill ptr2 and row2
    do j = 1, n
       do k = ptr(j), ptr(j+1) - 1
          i = row(k)
          if (j .ne. i) then
             row2(ptr2(i)) = j
             row2(ptr2(j)) = i
             ptr2(i) = ptr2(i) - 1
             ptr2(j) = ptr2(j) - 1
          end if
       end do
    end do
    do j = 1, n
       ptr2(j) = ptr2(j) + 1
    end do
  end subroutine half_to_full_drop_diag32_32

  ! Convert a matrix in half storage to one in full storage.
  ! Drops any diagonal entries.
  ! 64-bit to 32-bit ptr version. User must ensure no oor entries prior to call.
  subroutine half_to_full_drop_diag64_32(n, ptr, row, ptr2, row2)
    implicit none
    integer(ip_), intent(in) :: n
    integer(i8_), dimension(n+1), intent(in) :: ptr
    integer(ip_), dimension(ptr(n+1)-1), intent(in) :: row
    integer(ip_), dimension(*), intent(out) :: ptr2
    integer(ip_), dimension(*), intent(out) :: row2

    integer(ip_) :: i, j
    integer(i8_) :: kk

    ! Set ptr2(j) to hold no. nonzeros in column j
    ptr2(1:n+1) = 0
    do j = 1, n
       do kk = ptr(j), ptr(j+1) - 1
          i = row(kk)
          if (j .ne. i) then
             ptr2(i) = ptr2(i) + 1
             ptr2(j) = ptr2(j) + 1
          end if
       end do
    end do

    ! Set ptr2(j) to point to where row indices will end in row2
    do j = 2, n
       ptr2(j) = ptr2(j-1) + ptr2(j)
    end do
    ptr2(n+1) = ptr2(n) + 1

    ! Fill ptr2 and row2
    do j = 1, n
       do kk = ptr(j), ptr(j+1) - 1
          i = row(kk)
          if (j .ne. i) then
             row2(ptr2(i)) = j
             row2(ptr2(j)) = i
             ptr2(i) = ptr2(i) - 1
             ptr2(j) = ptr2(j) - 1
          end if
       end do
    end do
    do j = 1, n
       ptr2(j) = ptr2(j) + 1
    end do
  end subroutine half_to_full_drop_diag64_32

end module spral_metis_wrapper
