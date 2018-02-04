
/*
 * ==============================================
 * Main header file for the interface between
 * the Galahad solvers and the Ampl language.
 *
 * D. Orban@ECE, CUTEr version, Chicago 2002-2003
 * Nick Gould, CUTEst evolution,  February 6 2013 
 * ==============================================
 */

#include <math.h>

/* AIX does not append underscore to Fortran subroutine names */
#ifdef _AIX
#define FUNDERSCORE(a)   a
#else
#define FUNDERSCORE(a)   a##_
#endif

#ifdef  Fujitsu_frt
#define MAINENTRY MAIN__
#elif  Lahey_lf95
#define MAINENTRY MAIN__
#else
#define MAINENTRY main
#endif

#include "galahad_ampl.h"

#define MAX(m,n)  ((m)>(n)?(m):(n))
#define MIN(m,n)  ((m)<(n)?(m):(n))

/* Some real constants -- GalahadReal is defined in galahad.h */
#define ZERO             (GalahadReal)0.0
#define ONE              (GalahadReal)1.0
#define TWO              (GalahadReal)2.0
#define THREE            (GalahadReal)3.0
#define FIVE             (GalahadReal)5.0
#define FORTRAN_INFINITY (GalahadReal)pow( 10, 20 )

/*
 * =================================
 *  T Y P E   D E F I N I T I O N S
 * =================================
 */

/*
 * Define Fortran types for integer and double precision
 * The following choices are from f2c.h
 */

typedef long int integer;
typedef long int logical;
#define FORTRAN_FALSE (0)     /* Fortran FALSE */
#define FORTRAN_TRUE  (1)     /* Fortran  TRUE */

/*
 * Define shortcuts for the CUTEst library functions,
 * and try to avoid the trailing underscore.
 *
 */

#define CUTEST_usetup   FUNDERSCORE(cutest_usetup)
#define CUTEST_csetup   FUNDERSCORE(cutest_csetup)

#define CUTEST_udimen   FUNDERSCORE(cutest_udimen)
#define CUTEST_udimsh   FUNDERSCORE(cutest_udimsh)
#define CUTEST_udimse   FUNDERSCORE(cutest_udimse)
#define CUTEST_uvartype FUNDERSCORE(cutest_uvartype)
#define CUTEST_unames   FUNDERSCORE(cutest_unames)
#define CUTEST_ureport  FUNDERSCORE(cutest_ureport)

#define CUTEST_cdimen   FUNDERSCORE(cutest_cdimen)
#define CUTEST_cdimsj   FUNDERSCORE(cutest_cdimsj)
#define CUTEST_cdimsh   FUNDERSCORE(cutest_cdimsh)
#define CUTEST_cdimse   FUNDERSCORE(cutest_cdimse)
#define CUTEST_cdstats  FUNDERSCORE(cutest_cstats)
#define CUTEST_cvartype FUNDERSCORE(cutest_cvartype)
#define CUTEST_cnames   FUNDERSCORE(cutest_cnames)
#define CUTEST_creport  FUNDERSCORE(cutest_creport)

#define CUTEST_connames FUNDERSCORE(cutest_connames)
#define CUTEST_probname FUNDERSCORE(cutest_probname)
#define CUTEST_varnames FUNDERSCORE(cutest_varnames)

#define CUTEST_ufn      FUNDERSCORE(cutest_ufn)
#define CUTEST_ugr      FUNDERSCORE(cutest_ugr)
#define CUTEST_uofg     FUNDERSCORE(cutest_uofg)
#define CUTEST_ubandh   FUNDERSCORE(cutest_ubandh)
#define CUTEST_udh      FUNDERSCORE(cutest_udh)
#define CUTEST_ush      FUNDERSCORE(cutest_ush)
#define CUTEST_ueh      FUNDERSCORE(cutest_ueh)
#define CUTEST_ugrdh    FUNDERSCORE(cutest_ugrdh)
#define CUTEST_ugrsh    FUNDERSCORE(cutest_ugrsh)
#define CUTEST_ugreh    FUNDERSCORE(cutest_ugreh)
#define CUTEST_uhprod   FUNDERSCORE(cutest_uhprod)

#define CUTEST_cfn      FUNDERSCORE(cutest_cfn)
#define CUTEST_cofg     FUNDERSCORE(cutest_cofg)
#define CUTEST_ccfg     FUNDERSCORE(cutest_ccfg)
#define CUTEST_cgr      FUNDERSCORE(cutest_cgr)
#define CUTEST_csgr     FUNDERSCORE(cutest_csgr)
#define CUTEST_ccfsg    FUNDERSCORE(cutest_ccfsg)
#define CUTEST_ccifg    FUNDERSCORE(cutest_ccifg)
#define CUTEST_ccifsg   FUNDERSCORE(cutest_ccifsg)
#define CUTEST_cgrdh    FUNDERSCORE(cutest_cgrdh)
#define CUTEST_cdh      FUNDERSCORE(cutest_cdh)
#define CUTEST_csh      FUNDERSCORE(cutest_csh)
#define CUTEST_cshc     FUNDERSCORE(cutest_cshc)
#define CUTEST_ceh      FUNDERSCORE(cutest_ceh)
#define CUTEST_cidh     FUNDERSCORE(cutest_cidh)
#define CUTEST_cish     FUNDERSCORE(cutest_cish)
#define CUTEST_csgrsh   FUNDERSCORE(cutest_csgrsh)
#define CUTEST_csgreh   FUNDERSCORE(cutest_csgreh)
#define CUTEST_chprod   FUNDERSCORE(cutest_chprod)
#define CUTEST_chcprod  FUNDERSCORE(cutest_chcprod)
#define CUTEST_cjprod   FUNDERSCORE(cutest_cjprod)

#define CUTEST_uterminate FUNDERSCORE(cutest_uterminate)
#define CUTEST_cterminate FUNDERSCORE(cutest_cterminate)

#define ELFUN    FUNDERSCORE(elfun)
#define RANGE    FUNDERSCORE(range)
#define GROUP    FUNDERSCORE(group)

#define FORTRAN_OPEN  FUNDERSCORE(fortran_open)
#define FORTRAN_CLOSE FUNDERSCORE(fortran_close)

/*
 * Prototypes for CUTEst FORTRAN routines found in libcutest.a
 * See http://ccpforge.cse.rl.ac.uk/gf/project/cutest/
 */

/* Setup routines */
void CUTEST_usetup( integer *status, integer *funit, integer *iout, 
              integer *io_buffer, integer *n, GalahadReal *x,
	      GalahadReal *bl, GalahadReal *bu );
void CUTEST_csetup( integer *status, integer *funit, integer *iout, 
             integer *io_buffer, integer *n, integer *m,
	      GalahadReal *x, GalahadReal *bl, GalahadReal *bu, 
              GalahadReal *v, GalahadReal *cl, GalahadReal *cu, 
	      logical *equatn, logical *linear, 
              integer *e_order, integer *l_order, integer *v_order );

/* Unconstrained dimensioning and report routines */
void CUTEST_udimen( integer *status, integer *funit, integer *n );
void CUTEST_udimsh( integer *status, integer *nnzh );
void CUTEST_udimse( integer *status, integer *ne, integer *nzh,
                    integer *nzirnh );
void CUTEST_uvartype( integer *status, integer *n, integer *ivarty );
void CUTEST_unames( integer *status, integer *n, char *pname, char *vnames );
void CUTEST_ureport( integer *status, GalahadReal *calls, GalahadReal *time );

/* Constrained dimensioning and report routines */
void CUTEST_cdimen( integer *status, integer *funit, integer *n, integer *m );
void CUTEST_cdimsj( integer *status, integer *nnzj );
void CUTEST_cdimsh( integer *status, integer *nnzh );
void CUTEST_cdimse( integer *status, integer *ne, integer *nzh, 
                    integer *nzirnh );
void CUTEST_cstats( integer *status, integer *nonlinear_variables_objective, 
                    integer *nonlinear_variables_constraints,
                    integer *equality_constraints, 
                    integer *linear_constraints );
void CUTEST_cvartype( integer *status, integer *n, integer *ivarty );
void CUTEST_cnames( integer *status, integer *n, integer *m, char *pname, 
                    char *vnames, char *gnames );
void CUTEST_creport( integer *status, GalahadReal *calls, GalahadReal *time );

void CUTEST_connames( integer *status, integer *m, char *gname );
void CUTEST_probname( integer *status, char *pname );
void CUTEST_varnames( integer *status, integer *n, char *vname );

/* Unconstrained optimization routines */
void CUTEST_ufn( integer *status, integer *n, GalahadReal *x, GalahadReal *f );
void CUTEST_ugr( integer *status, integer *n, GalahadReal *x, GalahadReal *g );
void CUTEST_uofg( integer *status, integer *n, GalahadReal *x, GalahadReal *f, 
                  GalahadReal *g, logical *grad );
void CUTEST_ubandh( integer *status, integer *n, GalahadReal *x, 
                    integer *nsemib, GalahadReal *bandh, integer *lbandh, 
                    integer *maxsbw );
void CUTEST_udh( integer *status, integer *n, GalahadReal *x, integer *lh1, 
          GalahadReal *h );
void CUTEST_ush( integer *status, integer *n, GalahadReal *x, integer *nnzh, 
           integer *lh, GalahadReal *h, integer *irnh, integer *icnh );
void CUTEST_ueh( integer *status, integer *n, GalahadReal *x, integer *ne, 
          integer *le, integer *iprnhi, integer *iprhi, integer *lirnhi, 
          integer *irnhi, integer *lhi, GalahadReal *hi, logical *byrows );

void CUTEST_ugrdh( integer *status, integer *n, GalahadReal *x, GalahadReal *g, 
            integer *lh1, GalahadReal *h);
void CUTEST_ugrsh( integer *status, integer *n, GalahadReal *x, GalahadReal *g, 
             integer *nnzh,
	     integer *lh, GalahadReal *h, integer *irnh, integer *icnh );
void CUTEST_ugreh( integer *status, integer *n, GalahadReal *x, GalahadReal *g, 
             integer *ne,
             integer *le, integer *iprnhi, integer *iprhi, integer *lirnhi, 
             integer *irnhi, integer *lhi, GalahadReal *hi, logical *byrows );
void CUTEST_uhprod( integer *status, integer *n, logical *goth, GalahadReal *x, 
             GalahadReal *p, GalahadReal *q );

/* Constrained optimization routines */
void CUTEST_cfn( integer *status,  integer *n, integer *m, GalahadReal *x, 
          GalahadReal *f, GalahadReal *c );
void CUTEST_cofg( integer *status, integer *n, GalahadReal *x, GalahadReal *f, 
           GalahadReal *g, logical *grad );
void CUTEST_ccfg( integer *status, integer *n, integer *m, GalahadReal *x, 
	    GalahadReal *c, logical *jtrans, integer *lcjac1, integer *lcjac2,
	    GalahadReal *cjac, logical *grad );
void CUTEST_cgr( integer *status,  integer *n, integer *m, GalahadReal *x, 
            GalahadReal *v, logical *grlagf, GalahadReal *g, logical *jtrans,
	    integer *lcjac1, integer *lcjac2, GalahadReal *cjac );
void CUTEST_csgr( integer *status, integer *n, integer *m, 
            GalahadReal *x, GalahadReal *v, logical *grlagf, 
            integer *nnzj, integer *lcjac,
	    GalahadReal *cjac, integer *indvar, integer *indfun );
void CUTEST_ccfsg( integer *status,  integer *n, integer *m, GalahadReal *x, 
              GalahadReal *c, integer *nnzj, integer *lcjac,
	      GalahadReal *cjac, integer *indvar, integer *indfun,
	      logical *grad );
void CUTEST_ccifg( integer *status,  integer *n, integer *i, GalahadReal *x, 
              GalahadReal *ci, GalahadReal *gci, logical *grad );
void CUTEST_ccifsg( integer *status, integer *n, integer *i, GalahadReal *x, 
              GalahadReal *ci, integer *nnzsgc, integer *lsgci, 
              GalahadReal *sgci, integer *ivsgci, logical *grad );
void CUTEST_cgrdh( integer *status, integer *n, integer *m, GalahadReal *x, 
             GalahadReal *v, logical *grlagf, GalahadReal *g, logical *jtrans,
	     integer *lcjac1, integer *lcjac2, GalahadReal *cjac,
	     integer *lh1, GalahadReal *h );
void CUTEST_cdh( integer *status, integer *n, integer *m, GalahadReal *x, 
           GalahadReal *v, integer *lh1, GalahadReal *h );
void CUTEST_csh( integer *status, integer *n, integer *m, GalahadReal *x, 
           GalahadReal *v, integer *nnzh, integer *lh, GalahadReal *h, 
           integer *irnh, integer *icnh );
void CUTEST_cshc( integer *status, integer *n, integer *m, GalahadReal *x, 
           GalahadReal *v, integer *nnzh, integer *lh, GalahadReal *h, 
           integer *irnh, integer *icnh );
void CUTEST_ceh( integer *status, integer *n, integer *m, GalahadReal *x, 
           integer *lv, GalahadReal *v, integer *ne, 
           integer *le, integer *iprnhi, integer *iprhi, integer *lirnhi, 
           integer *irnhi, integer *lhi, GalahadReal *hi, logical *byrows );
void CUTEST_cidh( integer *status, integer *n, GalahadReal *x, integer *iprob, 
            integer *lh1, GalahadReal *h );
void CUTEST_cish( integer *status, integer *n, GalahadReal *x, integer *iprob, 
            integer *nnzh,
	    integer *lh, GalahadReal *h, integer *irnh, integer *icnh );
void CUTEST_csgrsh( integer *status, integer *n, integer *m, GalahadReal *x, 
	      GalahadReal *v, logical *grlagf, integer *nnzj, integer *lcjac,
	      GalahadReal *cjac, integer *indvar, integer *indfun,
	      integer *nnzh, integer *lh, GalahadReal *h, integer *irnh,
	      integer *icnh );
void CUTEST_csgreh( integer *status, integer *n, integer *m, GalahadReal *x, 
	      GalahadReal *v, logical *grlagf, integer *nnzj, integer *lcjac,
	      GalahadReal *cjac, integer *indvar, integer *indfun,
	      integer *ne, 
              integer *le, integer *iprnhi, integer *iprhi, integer *lirnhi, 
              integer *irnhi, integer *lhi, GalahadReal *hi, logical *byrows );
void CUTEST_chprod( integer *status, integer *n, integer *m, logical *goth, 
              GalahadReal *x, GalahadReal *v, GalahadReal *p, GalahadReal *q );
void CUTEST_chcprod( integer *status, integer *n, integer *m, logical *goth, 
               GalahadReal *x, GalahadReal *v, GalahadReal *p, GalahadReal *q );
void CUTEST_cjprod( integer *status, integer *n, integer *m, logical *gotj, 
              logical *jtrans, GalahadReal *x, GalahadReal *p, integer *lp, 
              GalahadReal *r, integer *lr );

/* Termination routines */
void CUTEST_uterminate( integer *status );
void CUTEST_cterminate( integer *status );

/* FORTRAN auxiliary subroutines to retrieve stream unit numbers */
void FORTRAN_OPEN(  integer *funit, char *fname, integer *ierr );
void FORTRAN_CLOSE( integer *funit, integer *ierr );

/*  Low-level SIF functions required by Lancelot-B  */
/* Arrays appear in uppercase, scalars in lowercase */

/*
void ELFUN( GalahadReal *FUVALS, GalahadReal *XVALUE, GalahadReal *EPVALU,
	    integer *ncalcf, integer *ITYPEE, integer *ISTAEV, integer *IELVAR,
	    integer *INTVAR, integer *ISTADH, integer *ISTEPA, integer *ICALCF,
	    integer *ltypee, integer *lstaev, integer *lelvar, integer *lntvar,
	    integer *lstadh, integer *lstepa, integer *lcalcf, integer *lfvalu,
	    integer *lxvalu, integer *lepvlu, integer *ifflag, integer *ifstat );

void RANGE( integer *ielemn, logical *transp, GalahadReal *W1, GalahadReal *W2, integer *nelvar,
	    integer *ninvar, integer *itype, integer *lw1, integer *lw2 );

void GROUP( GalahadReal *GVALUE, integer *lgvalu, GalahadReal *FVALUE, GalahadReal *GPVALU,
	    integer *ncalcg, integer *ITYPEG, integer *ISTGPA, integer *ICALCG,
	    integer *ltypeg, integer *lstgpa, integer *lcalcg, integer *lfvalu,
	    integer *lgpvlu, logical *derivs, integer *igstat );
*/

/* Functions declared in galahad_ps.c */
/*
void ps_initialize( void );
void ranges( fint *IELEM, fint *TRANSP, GalahadReal *W1, GalahadReal *W2,
	     fint *NELV, fint *NINV);
void derivs(fint j, GalahadReal *g, GalahadReal *h);
void funcvals(fint *ICALCF, int n, GalahadReal *FUVALS, GalahadReal *XT);
void funcgrads(fint *ICALCF, int n, GalahadReal *FUVALS,
	       fint *INTVAR, fint *ISTADH);
void groupvals(fint *ICALCG, int n, GalahadReal *ft, GalahadReal *gv);
void groupgrads(fint *ICALCG, int n, GalahadReal *g1, GalahadReal *g2);
*/
