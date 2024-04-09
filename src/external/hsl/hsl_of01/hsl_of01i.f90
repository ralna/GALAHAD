! THIS VERSION: GALAHAD 5.0 - 2024-03-17 AT 09:00 GMT.

#include "hsl_subset.h"

    MODULE hsl_of01_integer
      use hsl_kinds, only: ip_, long_, lp_
      private :: ip_, long_, lp_
      integer(ip_), parameter  :: maxpath = 400
      integer(ip_), parameter  :: maxname = 400
      type of01_data_private
        integer(ip_), allocatable :: buffer(:,:)
        logical(lp_),  allocatable :: differs(:)
        character(maxname), allocatable :: filename(:)
        integer(long_), allocatable :: first(:)
        integer(ip_) :: free
        integer(long_), allocatable :: left(:),right(:)
        integer(long_), allocatable :: highest(:)
        integer(ip_), allocatable :: index(:)
        integer(ip_) :: iolength
        integer(ip_) :: maxfiles = 0
        integer(ip_), allocatable :: name(:)
        integer(long_), allocatable :: next(:)
        integer(ip_) :: nfiles
        integer(long_) :: nrec
        integer(ip_), allocatable :: nextfile(:)
        integer(long_), allocatable :: older(:)
        integer(long_), allocatable :: page(:)
        integer(long_), allocatable :: page_list(:)
        character(maxpath), allocatable :: path(:)
        integer(long_), allocatable :: prev(:)
        integer(ip_), allocatable :: unit(:)
        integer(long_), allocatable :: younger(:)
        integer(ip_) :: youngest
      end type of01_data_private

      type of01_data
        integer(ip_) :: entry = 0
        integer(ip_) :: iostat
        integer(ip_) :: lpage
        integer(long_) :: ncall_read
        integer(long_) :: ncall_write
        integer(long_) :: nio_read
        integer(long_) :: nio_write
        integer(long_) :: npage
        integer(long_) :: file_size
        integer(long_) :: nwd_read
        integer(long_) :: nwd_write
        type (of01_data_private) :: private
        integer(ip_) :: stat ! Fortran stat parameter.
      end type of01_data
    CONTAINS
      SUBROUTINE of01i( )
      END SUBROUTINE of01i
    END MODULE hsl_of01_integer
