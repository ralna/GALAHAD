
/* ====================================================
 * Generic Ampl interface to GALAHAD Solvers :
 *
 * - QP: Generic Quadratic Programming solver,
 * - CQP: Interir-point Convex Quadratic Programming solver,
 * - QPA: Active-Set Quadratic Programming solver,
 * - QPB: Interior-Point Quadratic Programming solver,
 * - QPC: Crossover Interior-Point/Working-Set Quadratic Programming solver,
 * - Lancelot-B: Augmented Lagrangian nonlinear solver.
 * - Filtrane: Smooth Feasibility Problem solver.
 *
 * This is version v0.4 (11/Dec/2009)
 *
 *   Nick Gould, Dominique Orban and Philippe Toint
 *               for GALAHAD productions.
 * All rights reserved.            Chicago, March 2003.
 * ====================================================
 */

/* Includes */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "asl_pfgh.h"
#include "getstub.h"
#include "jacpdim.h"            /* For partially-separable structure */

#ifdef __cplusplus
extern "C" {                /* To prevent C++ compilers from mangling symbols */
#endif

#include "amplinter.h"       /* Includes base GALAHAD and pointer definitions */

/* ========================================================================== */

    /* Gateways used to access Galahad functions in their module */

#ifdef QPA
    /* Attribute memory address of USE_QPA( ) when available */
    void setup_use_qpa( void ( *fn ) (  ) ) {
        USE_QPA = fn;
    }
    /* Gateway to USE_QPA */ void Init_Galahad_Qpa( void ) {

        /* Request address of USE_QPA( ) */
        /* WRAP_USE_QPA( (void *)setup_use_qpa ); */
        WRAP_USE_QPA( setup_use_qpa );

        return;
    }
#endif


#ifdef QPB
    /* Attribute memory address of USE_QPB( ) when available */
    void setup_use_qpb( void ( *fn ) (  ) ) {
        USE_QPB = fn;
    }

    /* Gateway to USE_QPB */
    void Init_Galahad_Qpb( void ) {

        /* Request address of USE_QPB( ) */
        /*WRAP_USE_QPB( (void *)setup_use_qpb ); */
        WRAP_USE_QPB( setup_use_qpb );
        return;
    }
#endif

#ifdef QPC
    /* Attribute memory address of USE_QPC( ) when available */
    void setup_use_qpc( void ( *fn ) (  ) ) {
        USE_QPC = fn;
    }

    /* Gateway to USE_QPC */
    void Init_Galahad_Qpc( void ) {

        /* Request address of USE_QPC( ) */
        /*WRAP_USE_QPC( (void *)setup_use_qpc ); */
        WRAP_USE_QPC( setup_use_qpc );
        return;
    }
#endif

#ifdef CQP
    /* Attribute memory address of USE_CQP( ) when available */
    void setup_use_cqp( void ( *fn ) (  ) ) {
        USE_CQP = fn;
    }

    /* Gateway to USE_CQP */
    void Init_Galahad_Cqp( void ) {

        /* Request address of USE_CQP( ) */
        /*WRAP_USE_CQP( (void *)setup_use_cqp ); */
        WRAP_USE_CQP( setup_use_cqp );
        return;
    }
#endif

#ifdef QP
    /* Attribute memory address of USE_QP( ) when available */
    void setup_use_qp( void ( *fn ) (  ) ) {
        USE_QP = fn;
    }

    /* Gateway to USE_QP */
    void Init_Galahad_Qp( void ) {

        /* Request address of USE_QP( ) */
        /*WRAP_USE_QP( (void *)setup_use_qp ); */
        WRAP_USE_QP( setup_use_qp );
        return;
    }
#endif

#ifdef PRESOLVE
    /* Attribute memory address of USE_PRESOLVE( ) when available */
    void setup_use_presolve( void ( *fn ) (  ) ) {
        USE_PRESOLVE = fn;
    }

    /* Gateway to USE_PRESOLVE */
    void Init_Galahad_Presolve( void ) {

        /* Request address of USE_PRESOLVE( ) */
        /*WRAP_USE_PRESOLVE( (void *)setup_use_presolve ); */
        WRAP_USE_PRESOLVE( setup_use_presolve );
        return;
    }
#endif

#ifdef LANCELOT
    /* Attribute memory address of USE_LANCELOT( ) when available */
    void setup_use_lancelot( void ( *fn ) (  ) ) {
        USE_LANCELOT = fn;
    }

    /* Gateway to USE_LANCELOT */
    void Init_Galahad_Lancelot( void ) {

        /* Request address of USE_LANCELOT( ) */
        /*WRAP_USE_LANCELOT( (void *)setup_use_lancelot ); */
        WRAP_USE_LANCELOT( setup_use_lancelot );
        return;
    }
#endif

#ifdef FILTRANE
    /* Attribute memory address of USE_FILTRANE( ) when available */
    void setup_use_filtrane( void ( *fn ) (  ) ) {
        USE_FILTRANE = fn;
    }

    /* Gateway to USE_FILTRANE */
    void Init_Galahad_Filtrane( void ) {

        /* Request address of USE_FILTRANE( ) */
        /*WRAP_USE_FILTRANE( (void *)setup_use_filtrane ); */
        WRAP_USE_FILTRANE( setup_use_filtrane );
        return;
    }
#endif

/* ========================================================================== */

    /* Ampl driver specific declarations */

#define R_OPS  ((ASL_work*)asl)->I.r_ops_
#define OBJ_DE ((ASL_work*)asl)->I.obj_de_
#define CON_DE ((ASL_work*)asl)->I.con_de_
#define CHR (char*)

    /* Local declarations */

    ASL_pfgh *asl;              /* Main ASL structure */

    fint filtrane = ( fint ) 0; /* Run Filtrane */
    fint lancelot = ( fint ) 0; /* Run Lancelot-B */
    fint qpa = ( fint ) 0;      /* Run QPA  */
    fint qpb = ( fint ) 0;      /* Run QPB  */
    fint qpc = ( fint ) 0;      /* Run QPC  */
    fint cqp = ( fint ) 0;      /* Run CQP  */
    fint qp = ( fint ) 0;       /* Run QP  */
    fint presolve = ( fint ) 0;      /* Run PRESOLVE  */
    fint print = ( fint ) 0;    /* Print problem stats only */
    fint stats = ( fint ) 0;    /* Rprint statistics after solve */

    char code_name[13] = "Undefined";
    char prob_name[11] = "Undefined";

    Calls ncalls = { 0, 0, 0, 0, 0.0, 0.0, 0.0 };
    WallClock w_clock = { ( clock_t ) 0.0, ( clock_t ) 0.0 };
    VariableTypes *vartypes;
    ConstraintTypes *contypes;

    /* Keywords here select the Galahad solver to use.
     * They do not refer to solver-specific options,
     * as these will be read from the spec files
     * RUNQPA.SPC,  RUNQPB.SPC,  RUNQPC.SPC, RUNCQP.SPC,
     * RUNQP.SPC, RUNLANCELOT.SPC, RUNPRESOLVE.SPC   and RUNFILTRANE.SPC.
     *
     * Keywords must appear in alphabetical order.
     */
    static keyword keywds[] = {
        KW( CHR "filtrane", L_val, &filtrane, CHR "Use Feasibility solver Filtrane" ),
        KW( CHR "lancelot", L_val, &lancelot,
            CHR "Use Augmented Lagrangian Nonlinear solver Lancelot-B" ),
        KW( CHR "presolve", L_val, &presolve, CHR "Quadratic Program Presolver PRESOLVE" ),
        KW( CHR "print", L_val, &print, CHR "Print problem statistics only" ),
        KW( CHR "qpa", L_val, &qpa, CHR "Use Active Set QP solver QPA" ),
        KW( CHR "qpb", L_val, &qpb, CHR "Use Interior-Point QP solver QPB" ),
        KW( CHR "qpc", L_val, &qpc,
            CHR "Use Interior-Point/Active-Set crossover QP solver QPC" ),
        KW( CHR "cqp", L_val, &cqp,
            CHR "Use Interior-point convex QP solver CQP" ),
        KW( CHR "qp", L_val, &cqp,
            CHR "Use generic QP solver QP" ),
        KW( CHR "stats", L_val, &stats,
            CHR "Print detailed statistics after solve" )
    };

    Option_Info Oinfo = { CHR "galahad", CHR "GALAHAD", CHR "galahad_options",
        keywds, nkeywds, 0, CHR "0.2"
    };

/*
 * ============================================================================
 * Main Program
 * ============================================================================
 */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "main"

    int MAINENTRY( int argc, char **argv ) {

        char *stub, *kusage;
        GalahadReal dummy = ZERO;
        FILE *nl;
        clock_t time_tmp;
        int input = 0;
        int i;

        /* Usage */
        if( argc < 2 ) {
            fprintf( stderr, "Usage: %s stub\n", argv[0] );
            fprintf( stderr,
                     " use   %s -=  for a list of keyword=value options.\n",
                     argv[0] );
            return 1;
        }

        /* Start monitoring setup and solve time */
        w_clock.setup = clock();

        /* Allocate and initialize structures to hold problem data */
        asl = ( ASL_pfgh * ) ASL_alloc( ASL_read_pfgh );
        stub = getstub( &argv, &Oinfo );
        nl = jac0dim( stub, ( fint ) strlen( stub ) );

        /* Get command-line options */
        if( getopts( argv, &Oinfo ) )
            exit( 1 );

        Ampl_Init();

        if( pfgh_read( nl, 0 ) )
            SETERRQ( INPUT_OUTPUT_ERROR, "Error reading nl file." );

#ifdef DEBUG_GALAHAD
        Debug_Print_Values(  );
#endif

        /* Allocate space for variables and constraints types */
        vartypes = ( VariableTypes * ) calloc( 1, sizeof( VariableTypes ) );
        contypes = ( ConstraintTypes * ) calloc( 1, sizeof( ConstraintTypes ) );
        GalahadValidPointer( vartypes );
        GalahadValidPointer( contypes );
        GetVarTypes( asl, vartypes );
        GetConTypes( asl, contypes );

        /* Reset possibly infinite values */
        for( i = 0; i < n_var; i++ ) {
            LUv[i] = MAX( -FORTRAN_INFINITY, MIN( LUv[i], FORTRAN_INFINITY ) );
            Uvx[i] = MAX( -FORTRAN_INFINITY, MIN( Uvx[i], FORTRAN_INFINITY ) );
        }
        for( i = 0; i < n_con; i++ ) {
            LUrhs[i] =
                MAX( -FORTRAN_INFINITY, MIN( LUrhs[i], FORTRAN_INFINITY ) );
            Urhsx[i] =
                MAX( -FORTRAN_INFINITY, MIN( Urhsx[i], FORTRAN_INFINITY ) );
        }

        /* Problem set-up complete. Call Fortran driver */
        if( qpa ) {

#ifdef QPA
            strcpy( code_name, "QPA" );
            Init_Galahad_Qpa(  );
            w_clock.setup = clock() - w_clock.setup;
            w_clock.solve = clock();
            USE_QPA( &input );  /* QPA */
            w_clock.solve = clock() - w_clock.solve;
#else
            SETERRQ( 0, "QPA/Ampl was not installed" );
#endif

        } else if( qpb ) {

#ifdef QPB
            strcpy( code_name, "QPB" );
            Init_Galahad_Qpb(  );
            w_clock.setup = clock(  ) - w_clock.setup;
            w_clock.solve = clock();
            USE_QPB( &input );  /* QPB */
            w_clock.solve = clock() - w_clock.solve;
#else
            SETERRQ( 0, "QPB/Ampl was not installed" );
#endif

        } else if( qpc ) {

#ifdef QPC
            strcpy( code_name, "QPC" );
            Init_Galahad_Qpc(  );
            w_clock.setup = clock(  ) - w_clock.setup;
            w_clock.solve = clock();
            USE_QPC( &input );  /* QPC */
            w_clock.solve = clock() - w_clock.solve;
#else
            SETERRQ( 0, "QPC/Ampl was not installed" );
#endif

        } else if( cqp ) {

#ifdef CQP
            strcpy( code_name, "CQP" );
            Init_Galahad_Cqp(  );
            w_clock.setup = clock(  ) - w_clock.setup;
            w_clock.solve = clock();
            USE_CQP( &input );  /* CQP */
            w_clock.solve = clock() - w_clock.solve;
#else
            SETERRQ( 0, "CQP/Ampl was not installed" );
#endif

        } else if( cqp ) {

#ifdef QP
            strcpy( code_name, "QP" );
            Init_Galahad_Qp(  );
            w_clock.setup = clock(  ) - w_clock.setup;
            w_clock.solve = clock();
            USE_QP( &input );  /* QP */
            w_clock.solve = clock() - w_clock.solve;
#else
            SETERRQ( 0, "QP/Ampl was not installed" );
#endif

        } else if( lancelot ) {
            strcpy( code_name, "Lancelot-B" );
            /*Init_Galahad_Lancelot( ); */
            /*w_clock.setup = clock(  ) - w_clock.setup;*/
            /*w_clock.solve = clock();*/
            /*USE_LANCELOT( &input ); *//* Lancelot-B */
            /*w_clock.solve = clock() - w_clock.solve;*/
            SETERRQ( 0, "Lancelot-B/Ampl is not yet available." );
        } else if( presolve ) {

#ifdef PRESOLVE
            strcpy( code_name, "Presolve" );
            Init_Galahad_Presolve(  );
            w_clock.setup = clock(  ) - w_clock.setup;
            w_clock.solve = clock();
            USE_PRESOLVE( &input );  /* PRESOLVE */
            w_clock.solve = clock() - w_clock.solve;
#else
            SETERRQ( 0, "PRESOLVE/Ampl was not installed" );
#endif

        } else if( filtrane ) {

#ifdef FILTRANE
            strcpy( code_name, "Filtrane" );
            Init_Galahad_Filtrane(  );
            w_clock.setup = clock(  ) - w_clock.setup;
            w_clock.solve = clock();
            USE_FILTRANE( &input );     /* Filtrane */
            w_clock.solve = clock() - w_clock.solve;
#else
            SETERRQ( 0, "Filtrane/Ampl was not installed" );
#endif

        } else if( print ) {

            /* Do nothing; statistics will be printed below */
            w_clock.setup = clock(  ) - w_clock.setup;
            w_clock.solve = clock();
            strcpy( code_name, "Print" );
            for( i = 0; i < strlen( stub ); i++ )
                prob_name[i] = stub[i];
            for( i = strlen( stub ); i < 10; i++ )
                prob_name[i] = 0;
            stats = ( fint ) 1;
            w_clock.solve = clock() - w_clock.solve;

        } else {
            w_clock.setup = clock(  ) - w_clock.setup;
            w_clock.solve = clock();
            SETERRQ( AMBIGUOUS_SOLVER_NAME, "Solver name not recognized." );
            w_clock.solve = clock() - w_clock.solve;
        }

        /* Execution time */

        w_clock.setup_time = ( double )w_clock.setup / ( double )CLOCKS_PER_SEC;
        w_clock.solve_time = ( double )w_clock.solve / ( double )CLOCKS_PER_SEC;

        if( stats )
            Detailed_Stats(  );

        /* Optimization phase finished. Flush */
        Ampl_Terminate(  );

        /* Output a dummy solution */
        /* write_sol( CHR "\n End of execution", 0, 0, &Oinfo ); */

        /* End of execution */
        return 0;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "Ampl_Init"

    void Ampl_Init( void ) {

        int n_badvals = 0;

        /* Process command-line options -- trap errors */
        switch ( qpa ) {
            case 0:
                break;
            case 1:
                Printf( "Using solver QPA...\n" );
                break;
            default:
                fprintf( stderr, "qpa: invalid value: %d. Aborting\n", qpa );
                n_badvals++;
        }
        switch ( qpb ) {
            case 0:
                break;
            case 1:
                Printf( "Using solver QPB...\n" );
                break;
            default:
                fprintf( stderr, "qpb: invalid value: %d. Aborting\n", qpb );
                n_badvals++;
        }
        switch ( qpc ) {
            case 0:
                break;
            case 1:
                Printf( "Using solver QPC...\n" );
                break;
            default:
                fprintf( stderr, "qpc: invalid value: %d. Aborting\n", qpc );
                n_badvals++;
        }
        switch ( cqp ) {
            case 0:
                break;
            case 1:
                Printf( "Using solver CQP...\n" );
                break;
            default:
                fprintf( stderr, "cqp: invalid value: %d. Aborting\n", cqp );
                n_badvals++;
        }
        switch ( qp ) {
            case 0:
                break;
            case 1:
                Printf( "Using solver QP...\n" );
                break;
            default:
                fprintf( stderr, "qp: invalid value: %d. Aborting\n", qp );
                n_badvals++;
        }
        switch ( lancelot ) {
            case 0:
                break;
            case 1:
                Printf( "Using solver Lancelot-B...\n" );
                break;
            default:
                fprintf( stderr, "lancelot: invalid value: %d. Aborting\n", lancelot );
                n_badvals++;
        }
        switch ( presolve ) {
            case 0:
                break;
            case 1:
                Printf( "Using presolver PRESOLVE...\n" );
                break;
            default:
                fprintf( stderr, "presolve: invalid value: %d. Aborting\n", presolve );
                n_badvals++;
        }
        switch ( filtrane ) {
            case 0:
                break;
            case 1:
                Printf( "Using solver Filtrane...\n" );
                break;
            default:
                fprintf( stderr, "filtrane: invalid value: %d. Aborting\n", filtrane );
                n_badvals++;
        }
        switch ( stats ) {
            case 0:
                break;
            case 1:
                Printf( "Collecting statistics...\n" );
                break;
            default:
                fprintf( stderr, "stats: invalid value: %d. Using 0\n", stats );
                stats = ( fint ) 0;
        }
        if( n_badvals )
            exit( 1 );

        /* Only one solver can be specified */
        if( qpa + qpb + qpc + cqp + qp + lancelot + presolve + print + filtrane > 1 )
            SETERRQ( AMBIGUOUS_SOLVER_NAME, "Conflicting options specified." );

        if( !( qp || qpa || qpb || qpc || cqp || lancelot || presolve || print || filtrane ) ) {
            /* The default here should be Lancelot-B */
            SETWARNQ( AMBIGUOUS_SOLVER_NAME,
                      "No solver specified. Defaults to PRINT." );
            qp = qpa = qpb = qpc = cqp = lancelot = presolve = filtrane = ( fint ) 0;
            print = ( fint ) 1;
        }

        /* Allocate room to store problem data */
        X0 = ( real * ) Malloc( n_var * sizeof( real ) );
        LUv = ( real * ) Malloc( n_var * sizeof( real ) );
        Uvx = ( real * ) Malloc( n_var * sizeof( real ) );
        pi0 = ( real * ) Malloc( n_con * sizeof( real ) );
        LUrhs = ( real * ) Malloc( n_con * sizeof( real ) );
        Urhsx = ( real * ) Malloc( n_con * sizeof( real ) );

        /* Set Ampl reading options */
        want_xpi0 = 3;          /* Read primal and dual estimates */

        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "Ampl_Terminate"

    void Ampl_Terminate( void ) {

        /* Free Ampl data structures */
        free( X0 );
        free( LUrhs );
        free( Uvx );
        free( pi0 );
        free( Urhsx );
        free( LUv );
        free( vartypes );
        free( contypes );
        ASL_free( (ASL**)&asl );
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "Debug_Print_Values"

    void Debug_Print_Values( void ) {

        FILE *dbf;
        char *debugfile = "gampl_debug_log";
        int i;

        if( !( dbf = fopen( debugfile, "w" ) ) )
            SETERRQ( INPUT_OUTPUT_ERROR, "Unable to open log file." );

        fprintf( stderr, "Writing debug log file %s\n", debugfile );
        fprintf( dbf, "Galahad_Ampl ::\n" );
        fprintf( dbf, " n = %-8d\tm = %-8d\n", n_var, n_con );
        fprintf( dbf, " Initial X0: [ " );
        for( i = 0; i < n_var; i++ ) fprintf( dbf, "%g ", X0[i] );
        fprintf( dbf, "]\n" );
        fprintf( dbf, " Initial pi0: [ " );
        for( i = 0; i < n_con; i++ ) fprintf( dbf, "%g ", pi0[i] );
        fprintf( dbf, "]\n" );
        fprintf( dbf, " Initial Xl: [ " );
        for( i = 0; i < n_var; i++ ) fprintf( dbf, "%g ", LUv[i] );
        fprintf( dbf, "]\n" );
        fprintf( dbf, " Initial Xu: [ " );
        for( i = 0; i < n_var; i++ ) fprintf( dbf, "%g ", Uvx[i] );
        fprintf( dbf, "]\n" );
        fprintf( dbf, " Initial Cl: [ " );
        for( i = 0; i < n_con; i++ ) fprintf( dbf, "%g ", LUrhs[i] );
        fprintf( dbf, "]\n" );
        fprintf( dbf, " Initial Cu: [ " );
        for( i = 0; i < n_con; i++ ) fprintf( dbf, "%g ", Urhsx[i] );
        fprintf( dbf, "]\n" );

        fclose( dbf );

        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "udimen"

    void UDIMEN( integer * funit, integer * n ) {

        /* Argument 'funit' is ignored */
        *n = ( integer ) n_var;
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "cdimen"

    void CDIMEN( integer * funit, integer * n, integer * m ) {

        /* Argument 'funit' is ignored */
        *n = ( integer ) n_var;
        *m = ( integer ) n_con;
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "usetup"

    void USETUP( integer * funit, integer * iout, integer * n,
                 GalahadReal * x, GalahadReal * bl, GalahadReal * bu,
                 integer * nmax ){

        int i;

        GalahadValidPointer( x );
        GalahadValidPointer( bl );
        GalahadValidPointer( bu );

        /* Argument 'funit' is ignored */
        *n = ( integer ) n_var;
        for( i = 0; i < n_var; i++ ) {
            *( x + i ) = RealCast X0[i];
            *( bl + i ) = RealCast LUv[i];
            *( bu + i ) = RealCast Uvx[i];
        }

        /* Setup time */
        w_clock.setup = clock(  ) - w_clock.setup;
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "csetup"

    void CSETUP( integer * funit, integer * iout, integer * n, integer * m,
                 GalahadReal * x, GalahadReal * bl, GalahadReal * bu,
                 integer * nmax, logical * equatn, logical * linear,
                 GalahadReal * y, GalahadReal * cl, GalahadReal * cu,
                 integer * mmax, logical * efirst, logical * lfirst,
                 logical * nvfrst ) {

        int i;

        GalahadValidPointer( x );
        GalahadValidPointer( bl );
        GalahadValidPointer( bu );
        GalahadValidPointer( equatn );
        GalahadValidPointer( linear );
        GalahadValidPointer( y );
        GalahadValidPointer( cl );
        GalahadValidPointer( cu );

        /* Argument 'funit' is ignored */
        *n = ( integer ) n_var;
        *m = ( integer ) n_con;
        for( i = 0; i < n_var; i++ ) {
            *( x + i ) = RealCast X0[i];
            *( bl + i ) = RealCast LUv[i];
            *( bu + i ) = RealCast Uvx[i];
        }

        if( *efirst || *lfirst || *nvfrst )
            SETERRQ( NOT_YET_IMPLEMENTED,
                     "Ordering of constraints not yet implemented." );

        for( i = 0; i < n_con; i++ ) {
            *( y + i ) = RealCast pi0[i];
            *( cl + i ) = RealCast LUrhs[i];
            *( cu + i ) = RealCast Urhsx[i];
            *( equatn + i ) =
                ( LUrhs[i] == Urhsx[i] ) ? FORTRAN_TRUE : FORTRAN_FALSE;
            *( linear + i ) = ( i >= nlc ) ? FORTRAN_TRUE : FORTRAN_FALSE;
        }

        /* Setup time */
        w_clock.setup = clock(  ) - w_clock.setup;
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "unames"
    /* pname  is a Fortran string of length 10 holding the problem name.
     * vnames is an array (nvar x 10) of Fortran strings of length 10,
     *              holding the variables names.
     */

  void UNAMES( integer * n, char *pname, char *vnames ){

        char *names, id[4], *varid = "var", *snum, *p;
        long nv, ndigits = 0;
        int i, j, l;

        GalahadValidCharPointer( pname );

        /* Assign problem name, right-justified */
        l = MIN( strlen( filename ), 10 );
        if( strstr( filename, ".nl" ) ) l -= 3;
        for( i = 0; i < 10 - l; i++ )
            pname[i] = ' ';
        for( i = 10 - l; i < 10; i++ )
            pname[i] = filename[i - 10 + l];
        /* strcpy( prob_name, pname ); */

        /* Assign bogus variable names */
        /* Fortran strings end with blanks instead of \0, */
        /* thus we pad as in: var0000001 on 10 characters */
        strcpy( id, varid );
        for( nv = n_var + 1; nv; nv /= 10 )
            ndigits++;          /* #digits in n_var */
        snum = ( char * )Malloc( (ndigits+1) * sizeof( char ) );
        for( i = 0; i < n_var; i++ ) {
            p = itoa( i+1 );      /* Fortran indexing */
            strcpy( snum, p );
            free( p );
            l = strlen( snum );
            for( j = 0; j < 3; j++ )
                vnames[10 * i + j] = id[j];
            for( j = 3; j < 10 - l; j++ )
                vnames[10 * i + j] = '0';
            for( j = 0; j < l; j++ )
                vnames[10 * i + 10 - l + j] = snum[j];
        }
        free( snum );

        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "cnames"
    /* pname  is a Fortran string of length 10 holding the problem name.
     * vnames is an array (nvar x 10) of Fortran strings of length 10,
     *              holding the variables names.
     * gnames is an array (ncon x 10) of Fortran strings of length 10,
     *              holding the constraints names.
     */

    void CNAMES( integer * n, integer * m, char *pname, char *vnames,
                 char *gnames ) {

        char *names, id[4], *varid = "var", *conid = "con", *snum, *p;
        long nv, ndigits = 0;
        int i, j, l;

        GalahadValidCharPointer( pname );

        /* Assign problem name, right-justified */
        l = MIN( strlen( filename ), 10 );
        if( strstr( filename, ".nl" ) ) l -= 3;
        for( i = 0; i < 10 - l; i++ )
            pname[i] = ' ';
        for( i = 10 - l; i < 10; i++ )
            pname[i] = filename[i - 10 + l];
        /* strcpy( prob_name, pname ); */

        /* Assign bogus variable names */
        /* Fortran strings end with blanks instead of \0, */
        /* thus we pad as in: var0000001 on 10 characters */
        strcpy( id, varid );
        for( nv = n_var + 1; nv; nv /= 10 )
            ndigits++;          /* #digits in n_var */
        snum = ( char * )Malloc( (ndigits+1) * sizeof( char ) );
        for( i = 0; i < n_var; i++ ) {
            p = itoa( i+1 );      /* Fortran indexing */
            strcpy( snum, p );
            free( p );
            l = strlen( snum );
            for( j = 0; j < 3; j++ )
                vnames[10 * i + j] = id[j];
            for( j = 3; j < 10 - l; j++ )
                vnames[10 * i + j] = '0';
            for( j = 0; j < l; j++ )
                vnames[10 * i + 10 - l + j] = snum[j];
        }
        free( snum );

        /* Assign bogus constraint names */
        /*  con0000001 on 10 characters  */
        strcpy( id, conid );
        for( nv = n_con + 1; nv; nv /= 10 )
            ndigits++;          /* #digits in n_con */
        snum = ( char * )Malloc( ndigits * sizeof( char ) );
        for( i = 0; i < n_con; i++ ) {
            p = itoa( i+1 );      /* Fortran indexing */
            strcpy( snum, p );
            free( p );
            l = strlen( snum );
            for( j = 0; j < 3; j++ )
                gnames[10 * i + j] = id[j];
            for( j = 3; j < 10 - l; j++ )
                gnames[10 * i + j] = '0';
            for( j = 0; j < l; j++ )
                gnames[10 * i + 10 - l + j] = snum[j];
        }
        free( snum );

        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "ufn"

  void UFN( integer * n, GalahadReal * x, GalahadReal * f ){

        int i;
        fint nerror = ( fint ) 0;
        real *Xtmp, f0;

#ifdef SinglePrecision
        real *c_double;         /* For conval( ) */
#endif

        GalahadValidPointer( x );

        /* Evaluate objective at given point */
        Xtmp = ( real * ) Malloc( n_var * sizeof( real ) );
        for( i = 0; i < n_var; i++ )
            Xtmp[i] = AmplCast ZERO;

        /* There might be no objective in feasibility problems */
        if( filtrane && !n_obj ) {
            f0 = RealCast dummy_objective( 0, Xtmp, &nerror );
        } else {
            f0 = RealCast objval( 0, Xtmp, &nerror );
        }

#ifdef SinglePrecision
        for( i = 0; i < n_var; i++ )
            Xtmp[i] = AmplCast * ( x + i );
#endif

        if( filtrane && !n_obj ) {
            *f = RealCast dummy_objective( 0, Xtmp, &nerror ) + f0;
        } else {
            *f = RealCast objval( 0, Xtmp, &nerror ) + f0;
        }

        ncalls.feval++;
#ifdef DEBUG_GALAHAD
        printf( " Ampl:: f(x) = %g\n", *f );
#endif

        free( Xtmp );
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "cfn"

    void CFN( integer * n, integer * m, GalahadReal * x, GalahadReal * f,
              integer * lc, GalahadReal * c ) {

        int i;
        fint nerror = ( fint ) 0;
        real *Xtmp, f0;

#ifdef SinglePrecision
        real *c_double;         /* For conval( ) */
#endif

        GalahadValidPointer( x );
        GalahadValidPointer( c );

        /* Evaluate objective at given point */
        Xtmp = ( real * ) Malloc( n_var * sizeof( real ) );
        for( i = 0; i < n_var; i++ )
            Xtmp[i] = AmplCast ZERO;

        /* There might be no objective in feasibility problems */
        if( filtrane && !n_obj ) {
            f0 = RealCast dummy_objective( 0, Xtmp, &nerror );
        } else {
            f0 = RealCast objval( 0, Xtmp, &nerror );
        }

#ifdef SinglePrecision
        for( i = 0; i < n_var; i++ )
            Xtmp[i] = AmplCast * ( x + i );
#endif

        if( filtrane && !n_obj ) {
            *f = RealCast dummy_objective( 0, Xtmp, &nerror ) + f0;
        } else {
            *f = RealCast objval( 0, Xtmp, &nerror ) + f0;
        }

        ncalls.feval++;
#ifdef DEBUG_GALAHAD
        printf( " Ampl:: f(x) = %g\n", *f );
#endif

        /* Evaluate constraints at given point.
         * Must use intermediate array in single precision
         */
#ifdef SinglePrecision
        c_double = ( real * ) Malloc( n_con * sizeof( real ) );
        conval( Xtmp, c_double, &nerror );
        for( i = 0; i < n_con; i++ )
            *( c + i ) = RealCast c_double[i];
        free( c_double );
#else
        conval( x, c, &nerror );
#endif
        ncalls.ceval += n_con;

#ifdef DEBUG_GALAHAD
        for( i = 0; i < n_con; i++ ) {
            printf( " Ampl:: c[%d] = %g\n", i, *( c + i ) );
        }
#endif
        free( Xtmp );
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "cdimsj"

    void CDIMSJ( integer * nnzj ) {

        /* nnzj = nzc + n_var to also store dense gradient of f */
        *nnzj = ( n_con ? nzc : 1 ) + n_var;
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "udimsh"

    void UDIMSH( integer * nnzh ) {

        *nnzh = ( integer ) sphsetup( -1, 1, 1, 1 );
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "cdimsh"

    void CDIMSH( integer * nnzh ) {

        *nnzh = ( integer ) sphsetup( -1, 1, 1, 1 );
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "ugr"

  void UGR( integer * n, GalahadReal * x, GalahadReal * g ){

        fint nerror = ( fint ) 0;       /* Error code for jacval */
        real *gradL;            /* Gradient of the Lagrangian */
#ifdef SinglePrecision
        real *Xtmp;             /* Temporary X */
#endif
        cgrad *cg;              /* Jacobian in the DAG */
        int i, j;               /* Loops indices */

        GalahadValidPointer( x );
        GalahadValidPointer( g );

#ifdef SinglePrecision
        Xtmp = ( real * ) Malloc( n_var * sizeof( real ) );
        for( i = 0; i < n_var; i++ )
            Xtmp[i] = AmplCast *( x + i );

        /* There might be no objective defined in feasibility problems */
        if( filtrane && !n_obj )
            dummy_gradient( 0, Xtmp, g, &nerror );
        else
            objgrd( 0, Xtmp, g, &nerror );
#else
        if( filtrane && !n_obj )
            dummy_gradient( 0, x, g, &nerror );
        else
            objgrd( 0, x, g, &nerror );
#endif
        if( !filtrane )
            ncalls.geval++;     /* Filtrane does not really evaluate g(x) */

        /* gradL now contains the gradient of f(x) */

#ifdef DEBUG_GALAHAD
        Printf( "Galahad_Ampl :: gradient\n" );
        for( i = 0; i < *n; i++ )
            Printf( " g(%d) = %g\n", i, *( g + i ) );
#endif

        return;
    }

/* ========================================================================== */
/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "csgr"

    void CSGR( integer * n, integer * m, logical * grlagf, integer * lv,
               GalahadReal * v, GalahadReal * x, integer * nnzj,
               integer * lcjac, GalahadReal * cjac, integer * indvar,
               integer * indfun ) {

        fint nerror = ( fint ) 0;       /* Error code for jacval */
        real *gradL;            /* Gradient of the Lagrangian */
        real *Jtmp;             /* Temporary Jacobian */
#ifdef SinglePrecision
        real *Xtmp;             /* Temporary X */
#endif
        cgrad *cg;              /* Jacobian in the DAG */
        int i, j;               /* Loops indices */

        GalahadValidPointer( x );
        GalahadValidPointer( v );
        GalahadValidPointer( cjac );
        GalahadValidIntPointer( indvar );
        GalahadValidIntPointer( indfun );

        /* Store number of nonzeros in Jacobian */
        *nnzj = n_con ? nzc : 1;
        *nnzj += n_var;            /* Gradient of objective/Lagrangian */

        /* Make room to store the gradient of the Lagrangian */
        gradL = ( real * ) Malloc( n_var * sizeof( real ) );

#ifdef SinglePrecision
        Xtmp = ( real * ) Malloc( n_var * sizeof( real ) );
        for( i = 0; i < n_var; i++ )
            Xtmp[i] = AmplCast *( x + i );

        /* There might be no objective defined in feasibility problems */
        if( filtrane && !n_obj )
            dummy_gradient( 0, Xtmp, gradL, &nerror );
        else
            objgrd( 0, Xtmp, gradL, &nerror );
#else
        if( filtrane && !n_obj )
            dummy_gradient( 0, x, gradL, &nerror );
        else
            objgrd( 0, x, gradL, &nerror );
#endif
        if( !filtrane )
            ncalls.geval++;     /* Filtrane does not really evaluate g(x) */

        /* gradL now contains the gradient of f(x) */

        /* Evaluate and store the sparse Jacobian */
        Jtmp = ( real * ) Malloc( *nnzj * sizeof( real ) );
        nerror = ( fint ) 0;
#ifdef SinglePrecision
        jacval( Xtmp, Jtmp, &nerror );
        free( Xtmp );
#else
        jacval( x, Jtmp, &nerror );
#endif
        ncalls.Jeval += n_con;

        /* Counter j indicates slot for next element in cjac */
        j = 0;

        for( i = 0; i < n_con; i++ ) {
            for( cg = Cgrad[i]; cg; cg = cg->next ) {
                *( cjac + j ) = RealCast Jtmp[cg->goff];
                *( indfun + j ) = i + 1;
                *( indvar + j ) = cg->varno + 1;

                /* Update contribution to gradL */
                if( *grlagf )
                    gradL[cg->varno] +=
                        ( AmplCast * ( v + i ) ) * Jtmp[cg->goff];
                j++;
            }
        }
        free( Jtmp );

        /* Incorporate gradL in cjac; indfun = 0 indicates gradL */
        for( i = 0; i < n_var; i++ ) {
            *( cjac   + j + i ) = RealCast gradL[i];
            *( indfun + j + i ) = 0;
            *( indvar + j + i ) = i + 1;
        }

#ifdef DEBUG_GALAHAD
        Printf( "Galahad_Ampl :: Jacobian\n" );
        for( i = 0; i < *nnzj; i++ )
            Printf( " cjac(%d) = %g\n", i, *( cjac + i ) );
#endif

        free( gradL );
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "ush"

    void USH( integer * n, GalahadReal * x,
              integer * nnzh, integer * lh, GalahadReal * h,
              integer * irnh, integer * icnh ) {

        real *OW;               /* For Hessian computations */
        real *v_double;      /* For sphes( ) */
#ifdef SinglePrecision
        real *h_double;      /* For sphes( ) */
#endif
        int i, j, k;            /* Loops indices */

        GalahadValidPointer( x );
        GalahadValidPointer( h );
        GalahadValidIntPointer( irnh );
        GalahadValidIntPointer( icnh );

        /* Make room for Hessian of Lagrangian (hence the -1) */
        *nnzh = ( integer ) sphsetup( -1, 1, 1, 1 );
#ifdef SinglePrecision
        h_double = ( real * ) Malloc( *nnzh * sizeof( real ) );
#endif
        v_double = ( real * ) Malloc( n_con * sizeof( real ) );
        for( i = 0; i < n_con; i++ )
            v_double[i] = 0.0;

        /* Evaluate Hessian.
         * sphes( ) evaluates H and fills sputinfo->hrownos and
         * sputinfo->hcolstarts to describe the upper triangle of H in
         * Harwell-Boeing format.
         * QP, QPA, QPB, QPC, CQP, PRE and LANCELOT do not use H_col.
         */
        OW = ( real * ) Malloc( n_obj * sizeof( real ) );
        for( i = 0; i < n_obj; i++ )
            OW[i] = objtype[i] ? -ONE : ONE;   /* Indicates min/max-imization */

#ifdef SinglePrecision
        sphes( h_double, -1, OW, v_double );
        for( i = 0; i < *nnzh; i++ )
            *( h + i ) = RealCast h_double[i];
#else
        sphes( h, -1, OW, v_double );
#endif
        ncalls.Heval++;

        /* Obtain irnh and icnh from hcolstarts and hrownos */
        k = 0;
        for( i = 0; i < n_var + 1; i++ ) {
            for( j = sputinfo->hcolstarts[i]; j < sputinfo->hcolstarts[i + 1];
                 j++ ) {
                /* Add 1 to indices to get Fortran indexing */
                *( icnh + k ) = sputinfo->hrownos[j] + 1;
                *( irnh + k ) = i + 1;
                k++;
            }
        }

#ifdef DEBUG_GALAHAD
        /* Display Hessian for debugging */
        Printf( "Galahad_Ampl :: Hessian: nnzh = %d\n ", *nnzh );
        for( i = 0; i < *nnzh; i++ ) Printf( "%d ", *( irnh + i ) );
        Printf( "\n " );
        for( i = 0; i < *nnzh; i++ ) Printf( "%d ", *( icnh + i ) );
        Printf( "\n " );
        for( i = 0; i < *nnzh; i++ ) Printf( "%g ", *( h + i ) );
        Printf( "\n" );
#endif

        free( OW );
#ifdef SinglePrecision
        free( h_double );
#endif
        free( v_double );
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "csh"

    void CSH( integer * n, integer * m, GalahadReal * x, integer * lv,
              GalahadReal * v, integer * nnzh, integer * lh, GalahadReal * h,
              integer * irnh, integer * icnh ) {

        real *OW;               /* For Hessian computations */
#ifdef SinglePrecision
        real *h_double, *v_double;      /* For sphes( ) */
#endif
        int i, j, k;            /* Loops indices */

        GalahadValidPointer( x );
        GalahadValidPointer( v );
        GalahadValidPointer( h );
        GalahadValidIntPointer( irnh );
        GalahadValidIntPointer( icnh );

        /* Make room for Hessian of Lagrangian (hence the -1) */
        *nnzh = ( integer ) sphsetup( -1, 1, 1, 1 );
#ifdef SinglePrecision
        h_double = ( real * ) Malloc( *nnzh * sizeof( real ) );
        v_double = ( real * ) Malloc( n_con * sizeof( real ) );
        for( i = 0; i < n_con; i++ )
            v_double[i] = AmplCast * ( v + i );
#endif

        /* Evaluate Hessian.
         * sphes( ) evaluates H and fills sputinfo->hrownos and
         * sputinfo->hcolstarts to describe the upper triangle of H in
         * Harwell-Boeing format.
         * QP, QPA, QPB, QPC, CQP, PRESOLVE and LANCELOT do not use H_col.
         */
        OW = ( real * ) Malloc( n_obj * sizeof( real ) );
        for( i = 0; i < n_obj; i++ )
            OW[i] = objtype[i] ? -ONE : ONE;   /* Indicates min/max-imization */

#ifdef SinglePrecision
        sphes( h_double, -1, OW, v_double );
        for( i = 0; i < *nnzh; i++ )
            *( h + i ) = RealCast h_double[i];
#else
        sphes( h, -1, OW, v );
#endif
        ncalls.Heval++;

        /* Obtain irnh and icnh from hcolstarts and hrownos */
        k = 0;
        for( i = 0; i < n_var + 1; i++ ) {
            for( j = sputinfo->hcolstarts[i]; j < sputinfo->hcolstarts[i + 1];
                 j++ ) {
                /* Add 1 to indices to get Fortran indexing */
                *( icnh + k ) = sputinfo->hrownos[j] + 1;
                *( irnh + k ) = i + 1;
                k++;
            }
        }

#ifdef DEBUG_GALAHAD
        /* Display Hessian for debugging */
        Printf( "Galahad_Ampl :: Hessian: nnzh = %d\n ", *nnzh );
        for( i = 0; i < *nnzh; i++ ) Printf( "%d ", *( irnh + i ) );
        Printf( "\n " );
        for( i = 0; i < *nnzh; i++ ) Printf( "%d ", *( icnh + i ) );
        Printf( "\n " );
        for( i = 0; i < *nnzh; i++ ) Printf( "%g ", *( h + i ) );
        Printf( "\n" );
#endif

        free( OW );
#ifdef SinglePrecision
        free( h_double );
        free( v_double );
#endif
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "cprod"

    void CPROD( integer * n, integer * m, logical * goth, GalahadReal * x,
                integer * lv, GalahadReal * v, GalahadReal * p,
                GalahadReal * q ) {

        /* Does not yet use goth = true */

        real *OW;               /* For Hessian computations */
        real *h;                /* For sphes( ) */
        real sum;
        int nnzh;
        int i, j, k;            /* Loops indices */

        GalahadValidPointer( x );
        GalahadValidPointer( v );
        GalahadValidPointer( p );
        GalahadValidPointer( q );

        for( i = 0; i < n_var; i++ )
            *( q + i ) = RealCast ZERO;

        /* Make room for Hessian of Lagrangian (hence the -1) */
        nnzh = ( integer ) sphsetup( -1, 1, 1, 1 );
        h = ( real * ) Malloc( nnzh * sizeof( real ) );

        /* Evaluate Hessian.
         * sphes( ) evaluates H and fills sputinfo->hrownos and
         * sputinfo->hcolstarts to describe the upper triangle of H in
         * Harwell-Boeing format.
         * QP, QPA, QPB, QPC, CQP, PRESOLVE and LANCELOT do not use H_col.
         */
        OW = ( real * ) Malloc( n_obj * sizeof( real ) );
        for( i = 0; i < n_obj; i++ )
            OW[i] = objtype[i] ? -ONE : ONE;   /* Indicates min/max-imization */

        sphes( h, -1, OW, v );
        ncalls.Heval++;

        /* Perform product q = Hp */
        for( i = 0; i < n_var; i++ )
            *( q + i ) = ZERO;
        for( i = 0; i < n_var; i++ ) {
            for( j = sputinfo->hcolstarts[i]; j < sputinfo->hcolstarts[i + 1];
                 j++ ) {
                k = sputinfo->hrownos[j];
                *( q + k ) += RealCast( *( p + i ) * h[j] );
            }
        }

        free( OW );
        free( h );
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "ccfsg"

    void CCFSG( integer * n, integer * m, GalahadReal * x, integer * lc,
                GalahadReal * c, integer * nnzj, integer * lcjac,
                GalahadReal * cjac, integer * indvar, integer * indfun,
                logical * grad ) {

        fint nerror = ( fint ) 0;       /* Error code for jacval */
        real *Jtmp;             /* Temporary Jacobian */
#ifdef SinglePrecision
        real *Xtmp;             /* Temporary X */
        real *c_double;         /* Temporary C(X) */
#endif
        cgrad *cg;              /* Jacobian in the DAG */
        int i, j;               /* Loops indices */

        GalahadValidPointer( x );
        GalahadValidPointer( c );
        GalahadValidPointer( cjac );
        GalahadValidIntPointer( indvar );
        GalahadValidIntPointer( indfun );

        /* Store number of nonzeros in Jacobian */
        *nnzj = n_con ? nzc : 1;

#ifdef SinglePrecision
        Xtmp = ( real * ) Malloc( n_var * sizeof( real ) );
        c_double = ( real * ) Malloc( n_con * sizeof( real ) );
        for( i = 0; i < n_var; i++ )
            Xtmp[i] = AmplCast * ( x + i );
        conval( Xtmp, c_double, &nerror );
        for( i = 0; i < n_con; i++ )
            *( c + i ) = RealCast c_double[i];
        free( c_double );
#else
        conval( x, c, &nerror );
#endif
        ncalls.ceval += n_con;

        if( *grad ) {
            /* Evaluate and store the sparse Jacobian */
            Jtmp = ( real * ) Malloc( *nnzj * sizeof( real ) );
            nerror = ( fint ) 0;
#ifdef SinglePrecsion
            jacval( Xtmp, Jtmp, &nerror );
            free( Xtmp );
#else
            jacval( x, Jtmp, &nerror );
#endif
            ncalls.Jeval += n_con;

            /* Counter j indicates slot for next element in cjac */
            j = 0;

            for( i = 0; i < n_con; i++ ) {
                for( cg = Cgrad[i]; cg; cg = cg->next ) {
                    *( cjac + j ) = RealCast Jtmp[cg->goff];
                    *( indfun + j ) = i + 1;
                    *( indvar + j ) = cg->varno + 1;
                    j++;
                }
            }
            free( Jtmp );
        }
#ifdef DEBUG_GALAHAD
        Printf( "Galahad_Ampl :: Jacobian\n" );
        for( i = 0; i < *nnzj; i++ )
            Printf( " cjac(%d) = %g\n", i, *( cjac + i ) );
#endif
        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "GetVarTypes"

    void GetVarTypes( const ASL_pfgh * asl, VariableTypes * vartypes ) {

        int i;

        /* Initialization */
        vartypes->n_var_fixed = 0;
        vartypes->n_var_range = 0;
        vartypes->n_var_below = 0;
        vartypes->n_var_above = 0;
        vartypes->n_var_free = 0;

        for( i = 0; i < n_var; i++ ) {
            if( negInfinity < LUv[i] && Uvx[i] < Infinity ) {
                if( LUv[i] == Uvx[i] ) {
                    vartypes->n_var_fixed++;
                } else {
                    vartypes->n_var_range++;
                }
            } else {
                if( negInfinity < LUv[i] ) {
                    vartypes->n_var_below++;
                } else if( Uvx[i] < Infinity ) {
                    vartypes->n_var_above++;
                } else
                    vartypes->n_var_free++;
            }
        }

        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "GetConsTypes"

    void GetConTypes( const ASL_pfgh * asl, ConstraintTypes * constypes ) {

        int i;

        /* Initialization */
        constypes->n_con_eq = 0;
        constypes->n_con_range = 0;
        constypes->n_con_ineq = 0;
        constypes->n_con_compl = 0;
        constypes->n_con_free = 0;

        for( i = 0; i < n_con; i++ ) {
            if( ( negInfinity < LUrhs[i] ) && ( Urhsx[i] < Infinity ) ) {
                if( LUrhs[i] == Urhsx[i] )
                    constypes->n_con_eq++;
                else
                    constypes->n_con_range++;
            } else if( LUrhs[i] <= negInfinity && Infinity <= Urhsx[i] )
                constypes->n_con_free++;
            else
                constypes->n_con_ineq++;
        }

        constypes->n_con_compl = n_cc;

        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "Detailed_Stats"

    void Detailed_Stats( void ) {

        int nc = n_con ? n_con : 1;

        printf( "\n\n" );
        printf( " ===============================" );
        printf( " Statistics ===============================\n\n" );
        printf( "   Code name            %10s\t  Problem name           %10s\n",
                code_name, prob_name );
        printf( " # variables            %10d\t# constraints            %10d\n",
                n_var, n_con );
        printf( " # fixed variables      %10d\t# linear constraints     %10d\n",
                vartypes->n_var_fixed, n_con - nlc );
        printf( " # bounded below        %10d\t# equalities             %10d\n",
                vartypes->n_var_below, contypes->n_con_eq );
        printf( " # bounded above        %10d\t# inequalities           %10d\n",
                vartypes->n_var_above, contypes->n_con_ineq );
        printf( " # ranged variables     %10d\t# range constraints      %10d\n",
                vartypes->n_var_range, contypes->n_con_range );
        printf
            ( " # free variables       %10d\t# complementarities      %10d\n\n",
              vartypes->n_var_free, contypes->n_con_compl );
        printf( " # objective functions  %10d\t# objective gradients    %10d\n",
                ncalls.feval, ncalls.geval );
        printf( " # objective Hessians   %10d\t# Hessian-vector prdct   %10d\n",
                ncalls.Heval, ncalls.Hprod );
        printf
            ( " # constraint functions %10.3g\t# constraint gradients   %10.3g\n",
              ncalls.ceval / nc, ncalls.Jeval / nc );
        printf( " # constraint Hessians  %10.3g\n\n", ncalls.cHess / nc );
        printf
            ( "   Setup time (seconds) %#10.2g\t  Solve time (seconds)   %#10.2g\n",
              w_clock.setup_time, w_clock.solve_time );
        printf( " ===============================" );
        printf( "===========================================\n\n" );

        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "creprt"

    void CREPRT( float *calls, float *time ) {

        int nc = n_con ? n_con : 1;

        GalahadValidPointer( calls );
        GalahadValidPointer( time );

        *( calls + 0 ) = ( float )ncalls.feval;
        *( calls + 1 ) = ( float )ncalls.geval;
        *( calls + 2 ) = ( float )ncalls.Heval;
        *( calls + 3 ) = ( float )ncalls.Hprod;
        *( calls + 4 ) = ( float )ncalls.ceval / nc;
        *( calls + 5 ) = ( float )ncalls.Jeval / nc;
        *( calls + 6 ) = ( float )ncalls.cHess / nc;

        *( time + 0 ) = ( float )w_clock.setup_time;
        *( time + 1 ) = ( float )w_clock.solve_time;

        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "ureprt"

    void UREPRT( float *calls, float *time ) {

        GalahadValidPointer( calls );
        GalahadValidPointer( time );

        *( calls + 0 ) = ( float )ncalls.feval;
        *( calls + 1 ) = ( float )ncalls.geval;
        *( calls + 2 ) = ( float )ncalls.Heval;
        *( calls + 3 ) = ( float )ncalls.Hprod;

        *( time + 0 ) = ( float )w_clock.setup_time;
        *( time + 1 ) = ( float )w_clock.solve_time;

        return;
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "dummy_objective"

    /* Dummy objective used in feasibility problems
       where no (constant) objective was defined */

    real dummy_objective( int nobj, real * x, fint * nerror ) {

        return ZERO;

    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "dummy_gradient"

    /* Dummy objective gradient used in feasibility problems
       where no (constant) objective was defined */

    void dummy_gradient( int nobj, real * x, real * G, fint * nerror ) {

        int i;

        for( i = 0; i < n_var; i++ )
            G[i] = ZERO;
        return;

    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "elfun"

    void ELFUN( GalahadReal * FUVALS, GalahadReal * XVALUE,
                GalahadReal * EPVALU, integer * ncalcf, integer * ITYPEE,
                integer * ISTAEV, integer * IELVAR, integer * INTVAR,
                integer * ISTADH, integer * ISTEPA, integer * ICALCF,
                integer * ltypee, integer * lstaev, integer * lelvar,
                integer * lntvar, integer * lstadh, integer * lstepa,
                integer * lcalcf, integer * lfvalu, integer * lxvalu,
                integer * lepvlu, integer * ifflag, integer * ifstat ) {

        SETERRQ( NOT_YET_IMPLEMENTED, "Not yet available" );
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "range"

    void RANGE( integer * ielemn, logical * transp, GalahadReal * W1,
                GalahadReal * W2, integer * nelvar, integer * ninvar,
                integer * itype, integer * lw1, integer * lw2 ) {

        SETERRQ( NOT_YET_IMPLEMENTED, "Not yet available" );
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "group"

    void GROUP( GalahadReal * GVALUE, integer * lgvalu, GalahadReal * FVALUE,
                GalahadReal * GPVALU, integer * ncalcg, integer * ITYPEG,
                integer * ISTGPA, integer * ICALCG, integer * ltypeg,
                integer * lstgpa, integer * lcalcg, integer * lfvalu,
                integer * lgpvlu, logical * derivs, integer * igstat ) {

        SETERRQ( NOT_YET_IMPLEMENTED, "Not yet available" );
    }

/* ========================================================================== */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "itoa"

    char *itoa( int n ) {

        int i, nchar = 0;
        char *answer, neg = 0;

        /* Get number of digits in n */
        if( n < 0 ) {
            n = -n;
            nchar++;
            neg = 1;
        }
        nchar += ( int )log10( ( double )MAX( n, 1 ) ) + 1;

        /* Also make room for the trailing '\0' */
        answer = ( char * )malloc( ( nchar + 1 ) * sizeof( char ) );
        answer[nchar] = 0;

        for( i = 0; i < nchar; i++ )
            answer[nchar - i - 1] = '0' + ( ( n / ( int )pow( 10, i ) ) % 10 );

        if( neg )
            answer[0] = '-';

        return answer;
    }

/* ========================================================================== */

#ifdef __cplusplus
}                           /* To prevent C++ compilers from mangling symbols */
#endif
