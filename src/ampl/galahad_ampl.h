
/*
 * GALAHAD-specific common header file
 */

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {   /* To prevent C++ compilers from mangling symbols */
#endif

/*
 * Type definitions
 */

#if defined(SinglePrecision) || defined(REAL_32)
    typedef float  GalahadReal;    /* Single precision real numbers */
#define     AmplCast    (double)   /* Used to cast single to double */
#define     RealCast    (float)    /* Used to cast back to single   */
#else
    typedef double GalahadReal;    /* Double precision real numbers */
#define     RealCast               /* No cast needed as Ampl uses doubles */
#define     AmplCast               /* No cast needed as Ampl uses doubles */
#endif

    typedef struct Calls {
        unsigned feval;            /* # evaluations of f(x) */
        unsigned geval;            /* # evaluations of g(x)  = gradient of f(x) */
        unsigned Heval;            /* # evaluations of H(x)  = Hessian  of f(x) */
        unsigned Hprod;            /* # evaluations of H*p   = matrix-vector products */
        float    ceval;            /* # evaluations of ci(x) = constraints */
        float    Jeval;            /* # evaluations of Ji(x) = gradient of ci(x) */
        float    cHess;            /* # evaluations of Hi(x) = Hessian of ci(x) */
    } Calls;

    typedef struct WallClock {
        clock_t setup;             /* Total setup time */
        clock_t solve;             /* Total solve time */
        double  setup_time;
        double  solve_time;
    } WallClock;

    typedef struct {
        int n_var_fixed;           /* # fixed variables                   */
        int n_var_below;           /* # variables bounded below           */
        int n_var_above;           /* # variables bounded above           */
        int n_var_range;           /* # variables bounded below and above */
        int n_var_free;            /* # free variables                    */
    } VariableTypes;

    typedef struct {
        int n_con_range;           /* # range constraints      */
        int n_con_eq;              /* # equality constraints   */
        int n_con_ineq;            /* # inequality constraints */
        int n_con_compl;           /* # complementarity constraints */
        int n_con_free;            /* # dummy constraints      */
    } ConstraintTypes;


/*
 * Prototypes for Ampl-specific functions
 */

    char *itoa( int i );
    void Ampl_Init( void );
    void Ampl_Terminate( void );
    void Debug_Print_Values( void );
    void Detailed_Stats( void );
    void GetVarTypes( const ASL_pfgh *asl, VariableTypes *vartypes );
    void GetConTypes( const ASL_pfgh *asl, ConstraintTypes *constypes );
    real dummy_objective( int nobj, real *x, fint *nerror );
    void dummy_gradient( int nobj, real *x, real *G, fint *nerror );

/*
 * Prototypes for wrappers around Galahad subroutines
 */

#define WRAP_USE_QPA      FUNDERSCORE(wrap_use_qpa)
#define WRAP_USE_QPB      FUNDERSCORE(wrap_use_qpb)
#define WRAP_USE_QPC      FUNDERSCORE(wrap_use_qpc)
#define WRAP_USE_CQP      FUNDERSCORE(wrap_use_cqp)
#define WRAP_USE_QP       FUNDERSCORE(wrap_use_qp)
#define WRAP_USE_PRESOLVE FUNDERSCORE(wrap_use_presolve)
#define WRAP_USE_LANCELOT FUNDERSCORE(wrap_use_lancelot)
#define WRAP_USE_FILTRANE FUNDERSCORE(wrap_use_filtrane)

#if defined(QPA) || defined(GALAHAD_AMPL)
    Cextern void WRAP_USE_QPA( void (*fn)() );
#endif
#if defined(QPB) || defined(GALAHAD_AMPL)
    Cextern void WRAP_USE_QPB( void (*fn)() );
#endif
#if defined(QPC) || defined(GALAHAD_AMPL)
    Cextern void WRAP_USE_QPC( void (*fn)() );
#endif
#if defined(CQP) || defined(GALAHAD_AMPL)
    Cextern void WRAP_USE_CQP( void (*fn)() );
#endif
#if defined(QP) || defined(GALAHAD_AMPL)
    Cextern void WRAP_USE_QP( void (*fn)() );
#endif
#if defined(PRESOLVE) || defined(GALAHAD_AMPL)
    Cextern void WRAP_USE_PRESOLVE( void (*fn)() );
#endif
#if defined(LANCELOT) || defined(GALAHAD_AMPL)
    Cextern void WRAP_USE_LANCELOT( void (*fn)() );
#endif
#if defined(FILTRANE) || defined(GALAHAD_AMPL)
    Cextern void WRAP_USE_FILTRANE( void (*fn)() );
#endif

/*
 * Prototypes for Galahad-specific subroutines.
 * As these functions appear in Fortran 90 modules,
 * we have to obtain pointers to their actual
 * address in memory. This is done by means of the
 * wrappers above and gateways in galahad.c
 */

#if defined(QPA) || defined(GALAHAD_AMPL)
    void (*USE_QPA)();       /* Function pointer to USE_QPA( )  */
#endif
#if defined(QPB) || defined(GALAHAD_AMPL)
    void (*USE_QPB)();       /* Function pointer to USE_QPB( )  */
#endif
#if defined(QPC) || defined(GALAHAD_AMPL)
    void (*USE_QPC)();       /* Function pointer to USE_QPC( )  */
#endif
#if defined(CQP) || defined(GALAHAD_AMPL)
    void (*USE_CQP)();       /* Function pointer to USE_CQP( )  */
#endif
#if defined(QP) || defined(GALAHAD_AMPL)
    void (*USE_QP)();        /* Function pointer to USE_QP( )  */
#endif
#if defined(PRESOLVE) || defined(GALAHAD_AMPL)
    void (*USE_PRESOLVE)();  /* Function pointer to USE_PRESOLVE( )  */
#endif
#if defined(LANCELOT) || defined(GALAHAD_AMPL)
    void (*USE_LANCELOT)();  /* Function pointer to USE_LANCELOT( ) */
#endif
#if defined(FILTRANE) || defined(GALAHAD_AMPL)
    void (*USE_FILTRANE)();  /* Function pointer to USE_FILTRANE( ) */
#endif

/*
 * Treatment of error
 */

#define GALAHAD_ERR_ARG_BADPTR   -1
#define NOT_YET_IMPLEMENTED      -2
#define DIMENSION_MISMATCH       -3
#define INPUT_OUTPUT_ERROR       -4
#define AMBIGUOUS_SOLVER_NAME    -5
#define ELFUN_UNDEFINED          -6
#define GROUP_UNDEFINED          -7

#define SETERRQ(n,s) {                                     \
  fprintf( stderr, "  Galahad Error::      Code : %d\n", n );    \
  fprintf( stderr, "                       Msg :: %s\n", s );    \
  fprintf( stderr, "  Error  occured in  function %s\n", __FUNCT__ ); \
  fprintf( stderr, "                        file: %s\n", __FILE__ );  \
  fprintf( stderr, "                        line: %d\n", __LINE__ );  \
  fprintf( stderr, " -------------------------------\n" );            \
  exit( n );                                                        \
}

#define SETERRQi(n,s,i) {                                  \
  fprintf( stderr, "  Galahad Error::     Code  : %d\n", n );    \
  fprintf( stderr, "                      Msg  :: %s\n", s );    \
  fprintf( stderr, "                      Value : %d\n", i );    \
  fprintf( stderr, "  Error  occured in  function %s\n", __FUNCT__ ); \
  fprintf( stderr, "                        file: %s\n", __FILE__ );  \
  fprintf( stderr, "                        line: %d\n", __LINE__ );  \
  fprintf( stderr, " -------------------------------\n" );            \
  exit( n );                                                        \
}


#define SETWARNQ(n,s) {                                      \
  fprintf( stderr, "  Galahad Warning::    Code : %d\n", n );    \
  fprintf( stderr, "                       Msg :: %s\n", s );    \
  fprintf( stderr, "  Warning occured in function %s\n", __FUNCT__ ); \
  fprintf( stderr, "                        file: %s\n", __FILE__ );  \
  fprintf( stderr, "                        line: %d\n", __LINE__ );  \
  fprintf( stderr, " -------------------------------\n" );            \
}

#define SETWARNQi(n,s,i) {                                   \
  fprintf( stderr, "  Galahad Warning::   Code  : %d\n", n );    \
  fprintf( stderr, "                      Msg  :: %s\n", s );    \
  fprintf( stderr, "                      Value : %d\n", i );    \
  fprintf( stderr, "  Warning occured in function %s\n", __FUNCT__ ); \
  fprintf( stderr, "                        file: %s\n", __FILE__ );  \
  fprintf( stderr, "                        line: %d\n", __LINE__ );  \
  fprintf( stderr, " -------------------------------\n" );            \
}

#define SETWARNQs(n,s1,s2) {                                 \
  fprintf( stderr, "  Galahad Warning::   Code  : %d\n", n );    \
  fprintf( stderr, "                      Msg  :: %s\n", s1 );   \
  fprintf( stderr, "                      Value : %s\n", s2 );   \
  fprintf( stderr, "  Warning occured in function %s\n", __FUNCT__ ); \
  fprintf( stderr, "                        file: %s\n", __FILE__ );  \
  fprintf( stderr, "                        line: %d\n", __LINE__ );  \
  fprintf( stderr, " -------------------------------\n" );            \
}

#define GalahadValidPointer(h)                                        \
  {if (!h) {SETERRQ(GALAHAD_ERR_ARG_BADPTR,"Null Pointer");}          \
  if ((unsigned long)h & (unsigned long)3){                           \
    SETERRQ(GALAHAD_ERR_ARG_BADPTR,"Invalid Pointer");                \
  }}

#define GalahadValidCharPointer(h)                                    \
  {if (!h) {SETERRQ(GALAHAD_ERR_ARG_BADPTR,"Null Pointer");}          \
  }

#define GalahadValidIntPointer(h)                                     \
  {if (!h) {SETERRQ(GALAHAD_ERR_ARG_BADPTR,"Null Pointer");}          \
  if ((unsigned long)h & (unsigned long)3){                           \
    SETERRQ(GALAHAD_ERR_ARG_BADPTR,"Invalid Pointer to Int");         \
  }}

#define GalahadValidScalarPointer(h)                                  \
  {if (!h) {SETERRQ(GALAHAD_ERR_ARG_BADPTR,"Null Pointer");}          \
  if ((unsigned long)h & (unsigned long)3) {                          \
    SETERRQ(GALAHAD_ERR_ARG_BADPTR,"Invalid Pointer to Scalar");      \
  }}


#ifdef __cplusplus
}    /* To prevent C++ compilers from mangling symbols */
#endif
