#ifndef HSL_MI28D_H
#define HSL_MI28D_H

#ifndef mi28_control
#define mi28_control mi28_control_s
#define mi28_info mi28_info_s
#endif

typedef float mi28pkgtype_s_;

/* Derived type to hold control parameters for hsl_mi28 */
struct mi28_control_s {
   mi28pkgtype_s_ alpha;
   bool check ;
   int iorder;
   int iscale;
   mi28pkgtype_s_ lowalpha;
   int maxshift;
   bool rrt;
   mi28pkgtype_s_ shift_factor;
   mi28pkgtype_s_ shift_factor2;
   mi28pkgtype_s_ small;
   mi28pkgtype_s_ tau1;
   mi28pkgtype_s_ tau2;
   int unit_error;
   int unit_warning;
};

/* Communucates errors and information to the user. */
struct mi28_info_s {
   int band_after;
   int band_before;
   int dup;
   int flag;
   int flag61;
   int flag64;
   int flag68;
   int flag77;
   int nrestart;
   int nshift;
   int oor;
   mi28pkgtype_s_ profile_before;
   mi28pkgtype_s_ profile_after;
   long size_r;
   int stat;
   mi28pkgtype_s_ alpha;
};

