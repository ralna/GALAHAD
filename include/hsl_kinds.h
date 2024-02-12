#ifdef NAG_KINDS

#define lp_ 3
#ifdef INTEGER_64
#define ip_ 4
#else
#define ip_ 3
#endif
#define sp_ 1
#define dp_ 2
#ifdef HSL_DOUBLE
#define rp_ 1
#else
#define rp_ 2
#endif

#else

#define lp_ 4
#ifdef INTEGER_64
#define ip_ 8
#else
#define ip_ 4
#endif
#define sp_ 4
#define dp_ 8
#ifdef HSL_DOUBLE
#define rp_ 4
#else
#define rp_ 8
#endif

#endif
