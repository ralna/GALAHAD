//* \file trb_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2022-09-21 AT 10:00 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_TRB PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. September 21st 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_trb.h"

/* Nested TRS, GLTR, PSLS, LMS and SHA control and inform prototypes */
//bool trs_update_control(struct trs_control_type *control,
//                        PyObject *py_options);
//PyObject* trs_make_inform_dict(const struct trs_inform_type *inform);
//bool gltr_update_control(struct gltr_control_type *control,
//                         PyObject *py_options);
//PyObject* gltr_make_inform_dict(const struct gltr_inform_type *inform);
//bool psls_update_control(struct psls_control_type *control,
//                         PyObject *py_options);
//PyObject* psls_make_inform_dict(const struct psls_inform_type *inform);
//bool lms_update_control(struct lms_control_type *control,
//                        PyObject *py_options);
//PyObject* lms_make_inform_dict(const struct lms_inform_type *inform);
//bool sha_update_control(struct sha_control_type *control,
//                        PyObject *py_options);
//PyObject* sha_make_inform_dict(const struct sha_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct trb_control_type control;  // control struct
static struct trb_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   CALLBACK FUNCTIONS    -*-*-*-*-*-*-*-*-*-*

/* Python eval_* function pointers */
static PyObject *py_eval_f = NULL;
static PyObject *py_eval_g = NULL;
static PyObject *py_eval_h = NULL;

/* C eval_* function wrappers */
static int eval_f(int n, const double x[], double *f, const void *userdata){

    // Wrap input array as NumPy array
    npy_intp xdim[] = {n};
    PyObject *py_x = PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(O)", py_x);

    // Call Python eval_f
    PyObject *result = PyObject_CallObject(py_eval_f, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Extract single double return value
    if(!parse_double("eval_f return value", result, f)){
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

static int eval_g(int n, const double x[], double g[], const void *userdata){

    // Wrap input array as NumPy array
    npy_intp xdim[] = {n};
    PyArrayObject *py_x = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(O)", py_x);

    // Call Python eval_g
    PyObject *result = PyObject_CallObject(py_eval_g, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Check return value is of correct type, size, and shape
    if(!check_array_double("eval_g return value", (PyArrayObject*) result, n)){
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Get return value data pointer and copy data into g
    const double *gval = (double *) PyArray_DATA((PyArrayObject*) result);
    for(int i=0; i<n; i++) {
        g[i] = gval[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

static int eval_h(int n, int ne, const double x[], double hval[], const void *userdata){

    // Wrap input array as NumPy array
    npy_intp xdim[] = {n};
    PyArrayObject *py_x = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(O)", py_x);

    // Call Python eval_h
    PyObject *result = PyObject_CallObject(py_eval_h, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Check return value is of correct type, size, and shape
    if(!check_array_double("eval_h return value", (PyArrayObject*) result, ne)){
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Get return value data pointer and copy data into hval
    const double *val = (double *) PyArray_DATA((PyArrayObject*) result);
    for(int i=0; i<ne; i++) {
        hval[i] = val[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool trb_update_control(struct trb_control_type *control,
                               PyObject *py_options){

    // Use C defaults if Python options not passed
    if(!py_options) return true;

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    const char* key_name;



    // Iterate over Python options dictionary
    while(PyDict_Next(py_options, &pos, &key, &value)) {

        // Parse options key
        if(!parse_options_key(key, &key_name))
            return false;

        // Parse each int option
        if(strcmp(key_name, "f_indexing") == 0){
            if(!parse_bool_option(value, "f_indexing",
                                  &control->f_indexing))
                return false;
            continue;
        }
        if(strcmp(key_name, "error") == 0){
            if(!parse_int_option(value, "error",
                                  &control->error))
                return false;
            continue;
        }
        if(strcmp(key_name, "out") == 0){
            if(!parse_int_option(value, "out",
                                  &control->out))
                return false;
            continue;
        }
        if(strcmp(key_name, "print_level") == 0){
            if(!parse_int_option(value, "print_level",
                                  &control->print_level))
                return false;
            continue;
        }
        if(strcmp(key_name, "start_print") == 0){
            if(!parse_int_option(value, "start_print",
                                  &control->start_print))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_print") == 0){
            if(!parse_int_option(value, "stop_print",
                                  &control->stop_print))
                return false;
            continue;
        }
        if(strcmp(key_name, "print_gap") == 0){
            if(!parse_int_option(value, "print_gap",
                                  &control->print_gap))
                return false;
            continue;
        }
        if(strcmp(key_name, "maxit") == 0){
            if(!parse_int_option(value, "maxit",
                                  &control->maxit))
                return false;
            continue;
        }
        if(strcmp(key_name, "alive_unit") == 0){
            if(!parse_int_option(value, "alive_unit",
                                  &control->alive_unit))
                return false;
            continue;
        }
        if(strcmp(key_name, "more_toraldo") == 0){
            if(!parse_int_option(value, "more_toraldo",
                                  &control->more_toraldo))
                return false;
            continue;
        }
        if(strcmp(key_name, "non_monotone") == 0){
            if(!parse_int_option(value, "non_monotone",
                                  &control->non_monotone))
                return false;
            continue;
        }
        if(strcmp(key_name, "model") == 0){
            if(!parse_int_option(value, "model",
                                  &control->model))
                return false;
            continue;
        }
        if(strcmp(key_name, "norm") == 0){
            if(!parse_int_option(value, "norm",
                                  &control->norm))
                return false;
            continue;
        }
        if(strcmp(key_name, "semi_bandwidth") == 0){
            if(!parse_int_option(value, "semi_bandwidth",
                                  &control->semi_bandwidth))
                return false;
            continue;
        }
        if(strcmp(key_name, "lbfgs_vectors") == 0){
            if(!parse_int_option(value, "lbfgs_vectors",
                                  &control->lbfgs_vectors))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_dxg") == 0){
            if(!parse_int_option(value, "max_dxg",
                                  &control->max_dxg))
                return false;
            continue;
        }
        if(strcmp(key_name, "icfs_vectors") == 0){
            if(!parse_int_option(value, "icfs_vectors",
                                  &control->icfs_vectors))
                return false;
            continue;
        }
        if(strcmp(key_name, "mi28_lsize") == 0){
            if(!parse_int_option(value, "mi28_lsize",
                                  &control->mi28_lsize))
                return false;
            continue;
        }
        if(strcmp(key_name, "mi28_rsize") == 0){
            if(!parse_int_option(value, "mi28_rsize",
                                  &control->mi28_rsize))
                return false;
            continue;
        }
        if(strcmp(key_name, "advanced_start") == 0){
            if(!parse_int_option(value, "advanced_start",
                                  &control->advanced_start))
                return false;
            continue;
        }

        // Parse each float/double option
        if(strcmp(key_name, "infinity") == 0){
            if(!parse_double_option(value, "infinity",
                                  &control->infinity))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_pg_absolute") == 0){
            if(!parse_double_option(value, "stop_pg_absolute",
                                  &control->stop_pg_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_pg_relative") == 0){
            if(!parse_double_option(value, "stop_pg_relative",
                                  &control->stop_pg_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_s") == 0){
            if(!parse_double_option(value, "stop_s",
                                  &control->stop_s))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_radius") == 0){
            if(!parse_double_option(value, "initial_radius",
                                  &control->initial_radius))
                return false;
            continue;
        }
        if(strcmp(key_name, "maximum_radius") == 0){
            if(!parse_double_option(value, "maximum_radius",
                                  &control->maximum_radius))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_rel_cg") == 0){
            if(!parse_double_option(value, "stop_rel_cg",
                                  &control->stop_rel_cg))
                return false;
            continue;
        }
        if(strcmp(key_name, "eta_successful") == 0){
            if(!parse_double_option(value, "eta_successful",
                                  &control->eta_successful))
                return false;
            continue;
        }
        if(strcmp(key_name, "eta_very_successful") == 0){
            if(!parse_double_option(value, "eta_very_successful",
                                  &control->eta_very_successful))
                return false;
            continue;
        }
        if(strcmp(key_name, "eta_too_successful") == 0){
            if(!parse_double_option(value, "eta_too_successful",
                                  &control->eta_too_successful))
                return false;
            continue;
        }
        if(strcmp(key_name, "radius_increase") == 0){
            if(!parse_double_option(value, "radius_increase",
                                  &control->radius_increase))
                return false;
            continue;
        }
        if(strcmp(key_name, "radius_reduce") == 0){
            if(!parse_double_option(value, "radius_reduce",
                                  &control->radius_reduce))
                return false;
            continue;
        }
        if(strcmp(key_name, "radius_reduce_max") == 0){
            if(!parse_double_option(value, "radius_reduce_max",
                                  &control->radius_reduce_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "obj_unbounded") == 0){
            if(!parse_double_option(value, "obj_unbounded",
                                  &control->obj_unbounded))
                return false;
            continue;
        }
        if(strcmp(key_name, "cpu_time_limit") == 0){
            if(!parse_double_option(value, "cpu_time_limit",
                                  &control->cpu_time_limit))
                return false;
            continue;
        }
        if(strcmp(key_name, "clock_time_limit") == 0){
            if(!parse_double_option(value, "clock_time_limit",
                                  &control->clock_time_limit))
                return false;
            continue;
        }

        // Parse each bool option
        if(strcmp(key_name, "hessian_available") == 0){
            if(!parse_bool_option(value, "hessian_available",
                                  &control->hessian_available))
                return false;
            continue;
        }
        if(strcmp(key_name, "subproblem_direct") == 0){
            if(!parse_bool_option(value, "subproblem_direct",
                                  &control->subproblem_direct))
                return false;
            continue;
        }
        if(strcmp(key_name, "retrospective_trust_region") == 0){
            if(!parse_bool_option(value, "retrospective_trust_region",
                                  &control->retrospective_trust_region))
                return false;
            continue;
        }
        if(strcmp(key_name, "renormalize_radius") == 0){
            if(!parse_bool_option(value, "renormalize_radius",
                                  &control->renormalize_radius))
                return false;
            continue;
        }
        if(strcmp(key_name, "two_norm_tr") == 0){
            if(!parse_bool_option(value, "two_norm_tr",
                                  &control->two_norm_tr))
                return false;
            continue;
        }
        if(strcmp(key_name, "exact_gcp") == 0){
            if(!parse_bool_option(value, "exact_gcp",
                                  &control->exact_gcp))
                return false;
            continue;
        }
        if(strcmp(key_name, "accurate_bqp") == 0){
            if(!parse_bool_option(value, "accurate_bqp",
                                  &control->accurate_bqp))
                return false;
            continue;
        }
        if(strcmp(key_name, "space_critical") == 0){
            if(!parse_bool_option(value, "space_critical",
                                  &control->space_critical))
                return false;
            continue;
        }
        if(strcmp(key_name, "deallocate_error_fatal") == 0){
            if(!parse_bool_option(value, "deallocate_error_fatal",
                                  &control->deallocate_error_fatal))
                return false;
            continue;
        }

        // Parse each char option
        if(strcmp(key_name, "prefix") == 0){
            if(!parse_char_option(value, "prefix",
                                  control->prefix))
                return false;
            continue;
        }
        if(strcmp(key_name, "alive_file") == 0){
            if(!parse_char_option(value, "alive_file",
                                  control->alive_file))
                return false;
            continue;
        }

        // Parse nested control options
        //if(strcmp(key_name, "trs_options") == 0){
        //    if(!trs_update_control(&control->trs_control, value))
        //        return false;
        //    continue;
        //}
        //if(strcmp(key_name, "gltr_options") == 0){
        //    if(!gltr_update_control(&control->gltr_control, value))
        //        return false;
        //    continue;
        //}
        //if(strcmp(key_name, "psls_options") == 0){
        //    if(!psls_update_control(&control->psls_control, value))
        //        return false;
        //    continue;
        //}
        //if(strcmp(key_name, "lms_options") == 0){
        //    if(!lms_update_control(&control->lms_control, value))
        //        return false;
        //    continue;
        //}
        //if(strcmp(key_name, "lms_prec_options") == 0){
        //    if(!lms_update_control(&control->lms_control_prec, value))
        //        return false;
        //    continue;
        //}
        //if(strcmp(key_name, "sha_options") == 0){
        //    if(!sha_update_control(&control->sha_control, value))
        //        return false;
        //    continue;
        //}

        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
          "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* trb_make_time_dict(const struct trb_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries
    PyDict_SetItemString(py_time, "total", 
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "preprocess",
                         PyFloat_FromDouble(time->preprocess));
    PyDict_SetItemString(py_time, "analyse",
                         PyFloat_FromDouble(time->analyse));
    PyDict_SetItemString(py_time, "factorize",
                         PyFloat_FromDouble(time->factorize));
    PyDict_SetItemString(py_time, "solve",
                         PyFloat_FromDouble(time->solve));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_preprocess",
                         PyFloat_FromDouble(time->clock_preprocess));
    PyDict_SetItemString(py_time, "clock_analyse",
                         PyFloat_FromDouble(time->clock_analyse));
    PyDict_SetItemString(py_time, "clock_factorize",
                         PyFloat_FromDouble(time->clock_factorize));
    PyDict_SetItemString(py_time, "clock_solve",
                         PyFloat_FromDouble(time->clock_solve));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* trb_make_inform_dict(const struct trb_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    // Set int inform entries
    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "iter",
                         PyLong_FromLong(inform->iter));
    PyDict_SetItemString(py_inform, "cg_iter",
                         PyLong_FromLong(inform->cg_iter));
    PyDict_SetItemString(py_inform, "cg_maxit",
                         PyLong_FromLong(inform->cg_maxit));
    PyDict_SetItemString(py_inform, "f_eval",
                         PyLong_FromLong(inform->f_eval));
    PyDict_SetItemString(py_inform, "g_eval",
                         PyLong_FromLong(inform->g_eval));
    PyDict_SetItemString(py_inform, "h_eval",
                         PyLong_FromLong(inform->h_eval));
    PyDict_SetItemString(py_inform, "n_free",
                         PyLong_FromLong(inform->n_free));
    PyDict_SetItemString(py_inform, "factorization_status",
                         PyLong_FromLong(inform->factorization_status));
    PyDict_SetItemString(py_inform, "factorization_max",
                         PyLong_FromLong(inform->factorization_max));
    PyDict_SetItemString(py_inform, "max_entries_factors",
                         PyLong_FromLong(inform->max_entries_factors));
    PyDict_SetItemString(py_inform, "factorization_integer",
                         PyLong_FromLong(inform->factorization_integer));
    PyDict_SetItemString(py_inform, "factorization_real",
                         PyLong_FromLong(inform->factorization_real));

    // Set float/double inform entries
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "norm_pg",
                         PyFloat_FromDouble(inform->norm_pg));
    PyDict_SetItemString(py_inform, "radius",
                         PyFloat_FromDouble(inform->radius));

    // Set bool inform entries
    //PyDict_SetItemString(py_inform, "used_grad", 
    //                     PyBool_FromLong(inform->used_grad));

    // Set char inform entries
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time", 
                         trb_make_time_dict(&inform->time));
    // Set TRS, GLTR, PSLS, LMS and SHA nested dictionaries
    //PyDict_SetItemString(py_inform, "trs_inform",
    //                     trs_make_inform_dict(&inform->trs_inform));
    //PyDict_SetItemString(py_inform, "gltr_inform",
    //                     gltr_make_inform_dict(&inform->gltr_inform));
    //PyDict_SetItemString(py_inform, "psls_inform",
    //                     psls_make_inform_dict(&inform->psls_inform));
    //PyDict_SetItemString(py_inform, "lms_inform",
    //                     lms_make_inform_dict(&inform->lms_inform));
    //PyDict_SetItemString(py_inform, "sha_inform",
    //                     sha_make_inform_dict(&inform->sha_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   TRB_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_trb_initialize_doc,
"trb.initialize()\n"
"\n"
"Set default option values and initialize private data\n"
"\n"
"Returns\n"
"-------\n"
"options : dict\n"
"  dictionary containing default control options:\n"
"    error : int\n"
"      error and warning diagnostics occur on stream error.\n"
"    out : int\n"
"      general output occurs on stream out.\n"
"    print_level : int\n"
"      the level of output required. Possible values are:\n"
"\n"
"       * <= 0\n"
"\n"
"         no output\n"
"       * 1\n"
"\n"
"         a one-line summary for every improvement\n"
"       * 2\n"
"\n"
"         a summary of each iteration\n"
"       * >= 3\n"
"\n"
"         increasingly verbose (debugging) output.\n"
"    start_print : int\n"
"      any printing will start on this iteration.\n"
"    stop_print : int\n"
"      any printing will stop on this iteration.\n"
"    print_gap : int\n"
"      the number of iterations between printing.\n"
"    maxit : int\n"
"      the maximum number of iterations performed.\n"
"    alive_unit : int\n"
"      removal of the file alive_file from unit alive_unit\n"
"      terminates execution.\n"
"    alive_file : str\n"
"      see alive_unit.\n"
"    more_toraldo : int\n"
"      more_toraldo >= 1 gives the number of More'-Toraldo projected\n"
"      searches to be used to improve upon the Cauchy point,\n"
"      anything else is for the standard add-one-at-a-time CG search.\n"
"    non_monotone : int\n"
"      non-monotone <= 0 monotone strategy used, anything else\n"
"      non-monotone strategy with this history length used.\n"
"    model : int\n"
"      the model used.  Possible values are\n"
"\n"
//"      * 0 dynamic (*not yet implemented*)\n"
"      * 1\n"
"\n"
"        first-order (no Hessian)\n"
"      * 2\n"
"\n"
"        second-order (exact Hessian)\n"
"      * 3\n"
"\n"
"        barely second-order (identity Hessian)\n"
"      * 4\n"
"\n"
"        secant second-order (sparsity-based)\n"
//"      * 5 secant second-order (limited-memory BFGS, with
//"         ``lbfgs_vectors``  history) (*not yet implemented*)\n"
//"      * 6 secant second-order (limited-memory SR1, with\n"
//"          ``lbfgs_vectors``  history) (*not yet implemented*).\n"
"    norm : int\n"
"      The norm is defined via $||v||^2 = v^T P v$, and will define\n"
"      the preconditioner used for iterative methods. Possible\n"
"      values for $P$ are\n"
"\n"
"      * -3\n"
"\n"
"        users own preconditioner\n"
"      * -2\n"
"\n"
"        $P =$ limited-memory BFGS matrix (with ``lbfgs_vectors`` history)\n"
"      * -1\n"
"\n"
"        identity (= Euclidan two-norm)\n"
"      * 0\n"
"\n"
"        automatic (*not yet implemented*)\n"
"      * 1\n"
"\n"
"        diagonal, $P =$ diag( max( Hessian, ``min_diagonal`` ) )\n"
"      * 2\n"
"\n"
"        banded, $P =$ band( Hessian ) with semi-bandwidth ``semi_bandwidth``\n"
"      * 3\n"
"\n"
"        re-ordered band, P=band(order(A)) with semi-bandwidth ``semi_bandwidth``\n"
"      * 4\n"
"\n"
"        full factorization, $P =$ Hessian,  Schnabel-Eskow modification\n"
//"      * 5 full factorization, $P =$ Hessian, GMPS modification \n"
//"           (*not yet implemented*)\n"
"      * 6\n"
"\n"
"        incomplete factorization of Hessian, Lin-More'\n"
"      * 7\n"
"\n"
"        incomplete factorization of Hessian, HSL_MI28\n"
//"      * 8 incomplete factorization of Hessian, Munskgaard  (*not\n"
//"          yet implemented*)\n"
//"      * 9 expanding band of Hessian (*not yet implemented*).\n"
"    semi_bandwidth : int\n"
"      specify the semi-bandwidth of the band matrix P if required.\n"
"    lbfgs_vectors : int\n"
"      number of vectors used by the L-BFGS matrix P if required.\n"
"    max_dxg : int\n"
"      number of vectors used by the sparsity-based secant Hessian\n"
"      if required.\n"
"    icfs_vectors : int\n"
"      number of vectors used by the Lin-More' incomplete\n"
"      factorization matrix P if required.\n"
"    mi28_lsize : int\n"
"      the maximum number of fill entries within each column of the\n"
"      incomplete factor L computed by HSL_MI28. In general,\n"
"      increasing ``mi28_lsize`` improve the quality of the\n"
"      preconditioner but increases the time to compute and then\n"
"      apply the preconditioner. Values less than 0 are treated as 0.\n"
"    mi28_rsize : int\n"
"      the maximum number of entries within each column of the\n"
"      strictly lower triangular matrix $R$ used in the computation\n"
"      of the preconditioner by HSL_MI28. Rank-1 arrays of size\n"
"      ``mi28_rsize`` * n are allocated internally to hold $R$. Thus\n"
"      the amount of memory used, as well as the amount of work\n"
"      involved in computing the preconditioner, depends on\n"
"      ``mi28_rsize.`` Setting ``mi28_rsize`` > 0 generally leads to\n"
"      a higher quality preconditioner than using ``mi28_rsize`` =\n"
"      0, and choosing ``mi28_rsize`` >= ``mi28_lsize`` is generally\n"
"      recommended.\n"
"    advanced_start : int\n"
"      iterates of a variant on the strategy of Sartenaer SISC\n"
"      18(6)1990:1788-1803.\n"
"    infinity : float\n"
"      any bound larger than infinity in modulus will be regarded as\n"
"      infinite.\n"
"    stop_pg_absolute : float\n"
"      overall convergence tolerances. The iteration will terminate\n"
"      when the norm of the gradient of the objective function is\n"
"      smaller than MAX( ``stop_pg_absolute,`` ``stop_pg_relative``\n"
"      * norm of the initial gradient or if the step is less than\n"
"      ``stop_s``.\n"
"    stop_pg_relative : float\n"
"      see stop_pg_absolute.\n"
"    stop_s : float\n"
"      see stop_pg_absolute.\n"
"    initial_radius : float\n"
"      initial value for the trust-region radius.\n"
"    maximum_radius : float\n"
"      maximum permitted trust-region radius.\n"
"    stop_rel_cg : float\n"
"      required relative reduction in the resuiduals from CG.\n"
"    eta_successful : float\n"
"      a potential iterate will only be accepted if the actual\n"
"      decrease f - f(x_new) is larger than ``eta_successful`` times\n"
"      that predicted by a quadratic model of the decrease. The\n"
"      trust-region radius will be increased if this relative\n"
"      decrease is greater than ``eta_very_successful`` but smaller\n"
"      than ``eta_too_successful``.\n"
"    eta_very_successful : float\n"
"      see eta_successful.\n"
"    eta_too_successful : float\n"
"      see eta_successful.\n"
"    radius_increase : float\n"
"      on very successful iterations, the trust-region radius will\n"
"      be increased the factor ``radius_increase,`` while if the\n"
"      iteration is unsucceful, the radius will be decreased by a\n"
"      factor ``radius_reduce`` but no more than\n"
"      ``radius_reduce_max``.\n"
"    radius_reduce : float\n"
"      see radius_increase.\n"
"    radius_reduce_max : float\n"
"      see radius_increase.\n"
"    obj_unbounded : float\n"
"      the smallest value the objective function may take before the\n"
"      problem is marked as unbounded.\n"
"    cpu_time_limit : float\n"
"      the maximum CPU time allowed (-ve means infinite).\n"
"    clock_time_limit : float\n"
"      the maximum elapsed clock time allowed (-ve means infinite).\n"
"    hessian_available : bool\n"
"      is the Hessian matrix of second derivatives available or is\n"
"      access only via matrix-vector products?.\n"
"    subproblem_direct : bool\n"
"      use a direct (factorization) or (preconditioned) iterative\n"
"      method to find the search direction.\n"
"    retrospective_trust_region : bool\n"
"      is a retrospective strategy to be used to update the\n"
"      trust-region radius.\n"
"    renormalize_radius : bool\n"
"      should the radius be renormalized to account for a change in\n"
"      preconditioner?.\n"
"    two_norm_tr : bool\n"
"      should an ellipsoidal trust-region be used rather than an\n"
"      infinity norm one?.\n"
"    exact_gcp : bool\n"
"      is the exact Cauchy point required rather than an\n"
"      approximation?.\n"
"    accurate_bqp : bool\n"
"      should the minimizer of the quadratic model within the\n"
"      intersection of the trust-region and feasible box be found\n"
"      (to a prescribed accuracy) rather than a (much) cheaper\n"
"      approximation?.\n"
"    space_critical : bool\n"
"      if ``space_critical`` True, every effort will be made to use\n"
"      as little space as possible. This may result in longer\n"
"      computation time.\n"
"    deallocate_error_fatal : bool\n"
"      if ``deallocate_error_fatal`` is True, any array/pointer\n"
"      deallocation error will terminate execution. Otherwise,\n"
"      computation will continue.\n"
"    prefix : str\n"
"      all output lines will be prefixed by the string contained\n"
"      in quotes within ``prefix``, e.g. 'word' (note the qutoes)\n"
"      will result in the prefix word.\n"
"    trs_options : dict\n"
"      default control options for TRS (see ``trs.initialize``).\n"
"    gltr_options : dict\n"
"      default control options for GLTR (see ``gltr.initialize``).\n"
"    psls_options : dict\n"
"      default control options for PSLS (see ``psls.initialize``).\n"
"    lms_options : dict\n"
"      default control options for LMS (see ``lms.initialize``).\n"
"    lms_prec_options : dict\n"
"      default control options for LMS (see ``lms.initialize``).\n"
"    sha_options : dict\n"
"      default control options for SHA (see ``sha.initialize``).\n"
"\n"
);

static PyObject* py_trb_initialize(PyObject *self){

    // Call trb_initialize
    trb_initialize(&data, &control, &status);

    // Record that TRB has been initialised
    init_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   TRB_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*
//  NB import is a python reserved keyword so changed to load here

PyDoc_STRVAR(py_trb_load_doc,
"trb.load(n, x_l, x_u, H_type, H_ne, H_row, H_col, H_ptr, options=None)\n"
"\n"
"Import problem data into internal storage prior to solution.\n"
"\n"
"Parameters\n"
"----------\n"
"n : int\n"
"    holds the number of variables.\n"
"x_l : ndarray(n)\n"
"    holds the values $x^l$ of the lower bounds on the\n"
"    optimization variables $x$.\n"
"x_u : ndarray(n)\n"
"    holds the values $x^u$ of the upper bounds on the\n"
"    optimization variables $x$.\n"
"H_type : string\n"
"    specifies the symmetric storage scheme used for the Hessian.\n"
"    It should be one of 'coordinate', 'sparse_by_rows', 'dense',\n"
"    'diagonal' or 'absent', the latter if access to the Hessian\n"
"    is via matrix-vector products; lower or upper case variants\n"
"    are allowed.\n"
"H_ne : int\n"
"    holds the number of entries in the  lower triangular part of\n"
"    $H$ in the sparse co-ordinate storage scheme. It need\n"
"    not be set for any of the other three schemes.\n"
"H_row : ndarray(H_ne)\n"
"    holds the row indices of the lower triangular part of $H$\n"
"    in the sparse co-ordinate storage scheme. It need not be set for\n"
"    any of the other three schemes, and in this case can be None\n"
"H_col : ndarray(H_ne)\n"
"    holds the column indices of the  lower triangular part of\n"
"    $H$ in either the sparse co-ordinate, or the sparse row-wise\n"
"    storage scheme. It need not be set when the dense or diagonal\n"
"    storage schemes are used, and in this case can be None\n"
"H_ptr : ndarray(n+1)\n"
"    holds the starting position of each row of the lower triangular\n"
"    part of $H$, as well as the total number of entries plus one,\n"
"    in the sparse row-wise storage scheme. It need not be set when the\n"
"    other schemes are used, and in this case can be None\n"
"options : dict, optional\n"
"    dictionary of control options (see ``trb.initialize``).\n"
"\n"
);

static PyObject* py_trb_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_x_l, *py_x_u, *py_H_row, *py_H_col, *py_H_ptr;
    PyObject *py_options = NULL;
    double *x_l, *x_u;
    int *H_row = NULL, *H_col = NULL, *H_ptr = NULL;
    const char *H_type;
    int n, H_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","x_l","x_u","H_type","H_ne",
                             "H_row","H_col","H_ptr","options",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iOOsiOOO|O", kwlist, &n,
                                    &py_x_l, &py_x_u, &H_type, &H_ne, &py_H_row,
                                    &py_H_col, &py_H_ptr, &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
//    if((
    if(!(
        check_array_double("x_l", py_x_l, n) &&
        check_array_double("x_u", py_x_u, n) &&
        check_array_int("H_row", py_H_row, H_ne) &&
        check_array_int("H_col", py_H_col, H_ne) &&
        check_array_int("H_ptr", py_H_ptr, n+1)
        ))
        return NULL;

    // Get array data pointers
    x_l = (double *) PyArray_DATA(py_x_l);
    x_u = (double *) PyArray_DATA(py_x_u);

    // Convert 64bit integer H_row array to 32bit
    if((PyObject *) py_H_row != Py_None){
        H_row = malloc(H_ne * sizeof(int));
        long int *H_row_long = (long int *) PyArray_DATA(py_H_row);
        for(int i = 0; i < H_ne; i++) H_row[i] = (int) H_row_long[i];
    }

    // Convert 64bit integer H_col array to 32bit
    if((PyObject *) py_H_col != Py_None){
        H_col = malloc(H_ne * sizeof(int));
        long int *H_col_long = (long int *) PyArray_DATA(py_H_col);
        for(int i = 0; i < H_ne; i++) H_col[i] = (int) H_col_long[i];
    }

    // Convert 64bit integer H_ptr array to 32bit
    if((PyObject *) py_H_ptr != Py_None){
        H_ptr = malloc((n+1) * sizeof(int));
        long int *H_ptr_long = (long int *) PyArray_DATA(py_H_ptr);
        for(int i = 0; i < n+1; i++) H_ptr[i] = (int) H_ptr_long[i];
    }

    // Reset control options
    trb_reset_control(&control, &data, &status);

    // Update TRB control options
    if(!trb_update_control(&control, py_options))
        return NULL;

    // Call trb_import
    trb_import(&control, &data, &status, n, x_l, x_u, H_type, H_ne,
               H_row, H_col, H_ptr);

    // Free allocated memory
    if(H_row != NULL) free(H_row);
    if(H_col != NULL) free(H_col);
    if(H_ptr != NULL) free(H_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   TRB_SOLVE   -*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_trb_solve_doc,
"x, g = trb.solve(n, H_ne, x, g, eval_f, eval_g, eval_h)\n"
"\n"
"Find an approximate local minimizer of a given function subject\n"
"to simple bounds on the variables using a trust-region method.\n"
"\n"
"Parameters\n"
"----------\n"
"n : int\n"
"    holds the number of variables.\n"
"H_ne : int\n"
"    holds the number of entries in the lower triangular part of $H$.\n"
"x : ndarray(n)\n"
"    holds the values of optimization variables $x$.\n"
"eval_f : callable\n"
"    a user-defined function that must have the signature:\n"
"\n"
"     ``f = eval_f(x)``\n"
"\n"
"    The value of the objective function $f(x)$\n"
"    evaluated at $x$ must be assigned to ``f``.\n"
"eval_g : callable\n"
"    a user-defined function that must have the signature:\n"
"\n"
"     ``g = eval_g(x)``\n"
"\n"
"    The components of the gradient $\\nabla f(x)$ of the\n"
"    objective function evaluated at $x$ must be assigned to ``g``.\n"
"eval_h : callable\n"
"    a user-defined function that must have the signature:\n"
"\n"
"     ``h = eval_h(x)``\n"
"\n"
"    The components of the nonzeros in the lower triangle of the Hessian\n"
"    $\\nabla^2 f(x)$ of the objective function evaluated at\n"
"    $x$ must be assigned to ``h`` in the same order as specified\n"
"    in the sparsity pattern in ``trb.load``.\n"
"\n"
"Returns\n"
"-------\n"
"x : ndarray(n)\n"
"    holds the value of the approximate global minimizer $x$ after\n"
"    a successful call.\n"
"g : ndarray(n)\n"
"    holds the gradient $\\nabla f(x)$ of the objective function.\n"
"\n"
);

static PyObject* py_trb_solve(PyObject *self, PyObject *args){
    PyArrayObject *py_x;
    PyObject *temp_f, *temp_g, *temp_h;
    double *x;
    int n, H_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    if(!PyArg_ParseTuple(args, "iiOOOO", &n, &H_ne, &py_x,
                         &temp_f, &temp_g, &temp_h))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("x", py_x, n))
        return NULL;

    // Get array data pointers
    x = (double *) PyArray_DATA(py_x);
    // g = (double *) PyArray_DATA(py_g);

    // Check that functions are callable
    if(!(
        check_callable(temp_f) &&
        check_callable(temp_g) &&
        check_callable(temp_h)
        ))
        return NULL;

    // Store functions
    Py_XINCREF(temp_f);         /* Add a reference to new callback */
    Py_XDECREF(py_eval_f);      /* Dispose of previous callback */
    py_eval_f = temp_f;         /* Remember new callback */
    Py_XINCREF(temp_g);         /* Add a reference to new callback */
    Py_XDECREF(py_eval_g);      /* Dispose of previous callback */
    py_eval_g = temp_g;         /* Remember new callback */
    Py_XINCREF(temp_h);         /* Add a reference to new callback */
    Py_XDECREF(py_eval_h);      /* Dispose of previous callback */
    py_eval_h = temp_h;         /* Remember new callback */

    // Create empty C array for g
    double g[n];

    // Call trb_solve_direct
    status = 1; // set status to 1 on entry
    trb_solve_with_mat(&data, NULL, &status, n, x, g, H_ne, eval_f, eval_g,
                       eval_h, NULL);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;
    // Wrap C array as NumPy array
    npy_intp gdim[] = {n}; // size of g
    PyObject *py_g = PyArray_SimpleNewFromData(1, gdim, 
                        NPY_DOUBLE, (void *) g); // create NumPy g array

    // Return x and g
    return Py_BuildValue("OO", py_x, py_g);
}

//  *-*-*-*-*-*-*-*-*-*-   TRB_INFORMATION   -*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_trb_information_doc,
"inform = trb.information()\n"
"\n"
"Provide optional output information\n"
"\n"
"Returns\n"
"-------\n"
"inform : dict\n"
"   dictionary containing output information:\n"
"\n"
"    status : int\n"
"      return status.  Possible values are:\n"
"\n"
"      * 0\n"
"\n"
"        The run was succesful.\n"
"\n"
"      * -1\n"
"\n"
"        An allocation error occurred. A message indicating the\n"
"        offending array is written on unit control['error'], and\n"
"        the returned allocation status and a string containing\n"
"        the name of the offending array are held in\n"
"        inform['alloc_status'] and inform['bad_alloc'] respectively.\n"
"\n"
"      * -2\n"
"\n"
"        A deallocation error occurred.  A message indicating the\n"
"        offending array is written on unit control['error'] and \n"
"        the returned allocation status and a string containing\n"
"        the name of the offending array are held in \n"
"        inform['alloc_status'] and inform['bad_alloc'] respectively.\n"
"\n"
"      * -3\n"
"\n"
"        The restriction n > 0 or requirement that type contains\n"
"        its relevant string 'dense', 'coordinate', 'sparse_by_rows',\n"
"        'diagonal' or 'absent' has been violated.\n"
"\n"
"      * -7\n"
"\n"
"        The objective function appears to be unbounded from below.\n"
"\n"
"      * -9\n"
"\n"
"        The analysis phase of the factorization failed; the return\n"
"        status from the factorization package is given by\n"
"        inform['factor_status'].\n"
"\n"
"      * -10\n"
"\n"
"        The factorization failed; the return status from the\n"
"        factorization package is given by inform['factor_status'].\n"
"\n"
"      * -11\n"
"\n"
"        The solution of a set of linear equations using factors\n"
"        from the factorization package failed; the return status\n"
"        from the factorization package is given by\n"
"        inform['factor_status'].\n"
"\n"
"      * -16\n"
"\n"
"        The problem is so ill-conditioned that further progress\n"
"        is impossible.\n"
"\n"
"      * -18\n"
"\n"
"        Too many iterations have been performed. This may happen if\n"
"        control['maxit'] is too small, but may also be symptomatic\n"
"        of a badly scaled problem.\n"
"\n"
"      * -19\n"
"\n"
"        The CPU time limit has been reached. This may happen if\n"
"        control['cpu_time_limit'] is too small, but may also be\n"
"        symptomatic of a badly scaled problem.\n"
"\n"
"      * -82\n"
"\n"
"        The user has forced termination of the solver by removing\n"
"        the file named control['alive_file'] from unit\n"
"        control['alive_unit'].\n"
"\n"
"    alloc_status : int\n"
"      the status of the last attempted allocation/deallocation.\n"
"    bad_alloc : str\n"
"      the name of the array for which an allocation/deallocation\n"
"      error ocurred.\n"
"    iter : int\n"
"      the total number of iterations performed.\n"
"    cg_iter : int\n"
"      the total number of CG iterations performed.\n"
"    cg_maxit : int\n"
"      the maximum number of CG iterations allowed per iteration.\n"
"    f_eval : int\n"
"      the total number of evaluations of the objective function.\n"
"    g_eval : int\n"
"      the total number of evaluations of the gradient of the\n"
"      objective function.\n"
"    h_eval : int\n"
"      the total number of evaluations of the Hessian of the\n"
"      objective function.\n"
"    n_free : int\n"
"      the number of variables that are free from their bounds.\n"
"    factorization_max : int\n"
"      the maximum number of factorizations in a sub-problem solve.\n"
"    factorization_status : int\n"
"      the return status from the factorization.\n"
"    max_entries_factors : int\n"
"      the maximum number of entries in the factors.\n"
"    factorization_integer : int\n"
"      the total integer workspace required for the factorization.\n"
"    factorization_real : int\n"
"      the total real workspace required for the factorization.\n"
"    obj : float\n"
"      the value of the objective function at the best estimate of\n"
"      the solution determined by TRB_solve.\n"
"    norm_pg : float\n"
"      the norm of the projected gradient of the objective function\n"
"      at the best estimate of the solution determined by TRB_solve.\n"
"    radius : float\n"
"      the current value of the trust-region radius.\n"
"    time : dict\n"
"      dictionary containing timing information:\n"
"        total : float\n"
"          the total CPU time spent in the package.\n"
"        preprocess : float\n"
"          the CPU time spent preprocessing the problem.\n"
"        analyse : float\n"
"          the CPU time spent analysing the required matrices prior to\n"
"          factorization.\n"
"        factorize : float\n"
"          the CPU time spent factorizing the required matrices.\n"
"        solve : float\n"
"          the CPU time spent computing the search direction.\n"
"        clock_total : float\n"
"          the total clock time spent in the package.\n"
"        clock_preprocess : float\n"
"          the clock time spent preprocessing the problem.\n"
"        clock_analyse : float\n"
"          the clock time spent analysing the required matrices prior to\n"
"          factorization.\n"
"        clock_factorize : float\n"
"          the clock time spent factorizing the required matrices.\n"
"        clock_solve : float\n"
"          the clock time spent computing the search direction.\n"
"    trs_inform : dict\n"
"      inform parameters for TRS (see ``trs.information``).\n"
"    gltr_inform : dict\n"
"      inform parameters for GLTR (see ``gltr.information``).\n"
"    psls_inform : dict\n"
"      inform parameters for PSLS (see ``psls.information``).\n"
"    lms_inform : dict\n"
"      inform parameters for LMS (see ``lms.information``).\n"
"    lms_prec_inform : dict\n"
"      inform parameters for LMS used for preconditioning\n"
"      (see ``lms.information``).\n"
"    sha_inform : dict\n"
"      inform parameters for SHA (see ``sha.information``).\n"
"\n"
);

static PyObject* py_trb_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call trb_information
    trb_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = trb_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   TRB_TERMINATE   -*-*-*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_trb_terminate_doc,
"trb.terminate()\n"
"\n"
"Deallocate all internal private storage\n"
"\n"
);

static PyObject* py_trb_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call trb_terminate
    trb_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE TRB PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* trb python module method table */
static PyMethodDef trb_module_methods[] = {
    {"initialize", (PyCFunction) py_trb_initialize, METH_NOARGS,
      py_trb_initialize_doc},
    {"load", (PyCFunction) py_trb_load, METH_VARARGS | METH_KEYWORDS,
      py_trb_load_doc},
    {"solve", (PyCFunction) py_trb_solve, METH_VARARGS,
      py_trb_solve_doc},
    {"information", (PyCFunction) py_trb_information, METH_NOARGS,
      py_trb_information_doc},
    {"terminate", (PyCFunction) py_trb_terminate, METH_NOARGS,
      py_trb_terminate_doc},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* trb python module documentation */

PyDoc_STRVAR(trb_module_doc,
"The trb package uses a trust-region method to find a (local)\n"
"minimizer of a differentiable objective function $f(x)$ of m\n"
"any variables $x$, where the variables satisfy the simple bounds \n"
"$x^l <= x <= x^u$.  The method offers the choice of \n"
"direct and iterative solution of the key subproblems, and\n"
"is most suitable for large problems. First derivatives are required,\n"
"and if second derivatives can be calculated, they will be exploited---if\n"
"the product of second derivatives with a vector may be found but\n"
"not the derivatives themselves, that may also be exploited.\n"
"\n"
"See Section 4 of $GALAHAD/doc/trb.pdf for a brief description of the\n"
"method employed and other details.\n" 
"\n"
"matrix storage\n"
"--------------\n"
"\n"
"The symmetric $n$ by $n$ matrix $H = \\nabla_{xx}f$ may\n"
"be presented and stored in a variety of formats. But crucially symmetry \n"
"is exploited by only storing values from the lower triangular part \n"
"(i.e, those entries that lie on or below the leading diagonal).\n"
"\n"
"Dense storage format:\n"
"The matrix $H$ is stored as a compact  dense matrix by rows, that \n"
"is, the values of the entries of each row in turn are stored in order \n"
"within an appropriate real one-dimensional array. Since $H$ is \n"
"symmetric, only the lower triangular part (that is the part\n"
"$H_{ij}$ for $0 <= j <= i <= n-1$) need be held. \n"
"In this case the lower triangle should be stored by rows, that is\n"
"component $i * i / 2 + j$  of the storage array H_val\n"
"will hold the value $H_{ij}$ (and, by symmetry, $H_{ji}$)\n"
"for $0 <= j <= i <= n-1$.\n"
"\n"
"Sparse co-ordinate storage format:\n"
"Only the nonzero entries of the matrices are stored.\n"
"For the $l$-th entry, $0 <= l <= ne-1$, of $H$,\n"
"its row index i, column index j and value $H_{ij}$, \n"
"$0 <= j <= i <= n-1$,  are stored as the $l$-th \n"
"components of the integer arrays H_row and H_col and real array H_val, \n"
"respectively, while the number of nonzeros is recorded as \n"
"H_ne = $ne$. Note that only the entries in the lower triangle \n"
"should be stored.\n"
"\n"
"Sparse row-wise storage format:\n"
"Again only the nonzero entries are stored, but this time\n"
"they are ordered so that those in row i appear directly before those\n"
"in row i+1. For the i-th row of $H$ the i-th component of the\n"
"integer array H_ptr holds the position of the first entry in this row,\n"
"while H_ptr(n) holds the total number of entries plus one.\n"
"The column indices j, $0 <= j <= i$, and values \n"
"$H_{ij}$ of the  entries in the i-th row are stored in components\n"
"l = H_ptr(i), ..., H_ptr(i+1)-1 of the\n"
"integer array H_col, and real array H_val, respectively. Note that\n"
"as before only the entries in the lower triangle should be stored. For\n"
"sparse matrices, this scheme almost always requires less storage than\n"
"its predecessor.\n"
"\n"
"call order\n"
"----------\n"
"The functions should be called in the following order, with\n"
"[] indicating an optional call\n"
"\n"
"  ``trb.initialize``\n"
"\n"
"  ``trb.load``\n"
"\n"
"  ``trb.solve``\n"
"\n"
"  [``trb.information``]\n"
"\n"
"  ``trb.terminate``\n"
);

/* trb python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "trb",               /* name of module */
   trb_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   trb_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_trb(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
