/* versiont.c */
/* Full test for the VERSION C interface */

#include <stdio.h>
#include "galahad_version.h"

int main(void) {

    int major;
    int minor;
    int patch;

    version_galahad( &major, &minor, &patch );
    printf( " GALAHAD version: %d.%d.%d\n", major, minor, patch );
}
