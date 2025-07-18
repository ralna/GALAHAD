! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 10:30 GMT.

#include "hsl_subset.h"

    MODULE hsl_kb22_long_integer
      PRIVATE
      PUBLIC :: KB22_build_heap, KB22_get_smallest
      PUBLIC :: kb22_available
    CONTAINS
      logical function kb22_available()
        kb22_available = .FALSE.
      end function

      SUBROUTINE KB22_build_heap( )
      END SUBROUTINE KB22_build_heap

      SUBROUTINE KB22_get_smallest( )
      END SUBROUTINE KB22_get_smallest
    END MODULE hsl_kb22_long_integer
