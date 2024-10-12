! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 15:00 GMT.

#include "hsl_subset.h"

    MODULE hsl_mc34_real
      implicit none
      private
      public mc34_expand
      LOGICAL, PUBLIC, PARAMETER :: mc34_available = .FALSE.
      interface mc34_expand
         module procedure mc34_expand_real
      end interface
    CONTAINS
      SUBROUTINE mc34_expand_real( )
      END SUBROUTINE mc34_expand_real
    END MODULE hsl_mc34_real
