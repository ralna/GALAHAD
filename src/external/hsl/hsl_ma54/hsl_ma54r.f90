! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 15:00 GMT.

#include "hsl_subset.h"

    MODULE hsl_ma54_real
      PUBLIC :: ma54_available
    CONTAINS
      logical function ma54_available()
        ma54_available = .FALSE.
      end function

      SUBROUTINE ma54r( )
      END SUBROUTINE ma54r
    END MODULE hsl_ma54_real
