//* \file nls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2022-10-13 AT 14:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_NLS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_nls.h"

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
static struct nls_control_type control;  // control struct
static struct nls_inform_type inform;    // inform struct
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
static bool nls_update_control(struct nls_control_type *control,
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
static PyObject* nls_make_time_dict(const struct nls_time_type *time){
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
static PyObject* nls_make_inform_dict(const struct nls_inform_type *inform){
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
                         nls_make_time_dict(&inform->time));
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

//  *-*-*-*-*-*-*-*-*-*-   NLS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*


static PyObject* py_nls_initialize(PyObject *self){

    // Call nls_initialize
    nls_initialize(&data, &control, &status);

    // Record that NLS has been initialised
    init_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   NLS_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_nls_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_J_row, *py_J_col, *py_J_ptr;
    PyArrayObject *py_H_row, *py_H_col, *py_H_ptr;
    PyArrayObject *py_P_row, *py_P_col, *py_P_ptr;
    PyArrayObject *py_w;
    PyObject *py_options = NULL;
    int *J_row = NULL, *J_col = NULL, *J_ptr = NULL;
    int *H_row = NULL, *H_col = NULL, *H_ptr = NULL;
    int *P_row = NULL, *P_col = NULL, *P_ptr = NULL;
    const char *J_type, *H_type, *P_type;
    double *w;
    int n, m, J_ne, H_ne, P_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","m",
                             "J_type","J_ne","J_row","J_col","J_ptr",
                             "H_type","H_ne","H_row","H_col","H_ptr",
                             "H_type","H_ne","H_row","H_col","H_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOO|siOOOsiOOOOO",
                                    kwlist, &n, &m,
                                    &J_type, &J_ne, &py_J_row,
                                    &py_J_col, &py_J_ptr,
                                    &H_type, &H_ne, &py_H_row,
                                    &py_H_col, &py_H_ptr,
                                    &P_type, &P_ne, &py_P_row,
                                    &py_P_col, &py_P_ptr, &py_w,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("J_row", py_J_row, J_ne) &&
        check_array_int("J_col", py_J_col, J_ne) &&
        check_array_int("J_ptr", py_J_ptr, n+1)
        ))
        return NULL;
    if(!(
        check_array_int("H_row", py_H_row, H_ne) &&
        check_array_int("H_col", py_H_col, H_ne) &&
        check_array_int("H_ptr", py_H_ptr, n+1)
        ))
        return NULL;
    if(!(
        check_array_int("P_row", py_P_row, P_ne) &&
        check_array_int("P_col", py_P_col, P_ne) &&
        check_array_int("P_ptr", py_P_ptr, m+1)
        ))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("w", py_w, m))
        return NULL;

    // Get array data pointer
    w = (double *) PyArray_DATA(py_w);

    // Convert 64bit integer J_row array to 32bit
    if((PyObject *) py_J_row != Py_None){
        J_row = malloc(J_ne * sizeof(int));
        long int *J_row_long = (long int *) PyArray_DATA(py_J_row);
        for(int i = 0; i < J_ne; i++) J_row[i] = (int) J_row_long[i];
    }

    // Convert 64bit integer J_col array to 32bit
    if((PyObject *) py_J_col != Py_None){
        J_col = malloc(J_ne * sizeof(int));
        long int *J_col_long = (long int *) PyArray_DATA(py_J_col);
        for(int i = 0; i < J_ne; i++) J_col[i] = (int) J_col_long[i];
    }

    // Convert 64bit integer J_ptr array to 32bit
    if((PyObject *) py_J_ptr != Py_None){
        J_ptr = malloc((n+1) * sizeof(int));
        long int *J_ptr_long = (long int *) PyArray_DATA(py_J_ptr);
        for(int i = 0; i < n+1; i++) J_ptr[i] = (int) J_ptr_long[i];
    }

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

    // Convert 64bit integer P_row array to 32bit
    if((PyObject *) py_P_row != Py_None){
        P_row = malloc(P_ne * sizeof(int));
        long int *P_row_long = (long int *) PyArray_DATA(py_P_row);
        for(int i = 0; i < P_ne; i++) P_row[i] = (int) P_row_long[i];
    }

    // Convert 64bit integer P_col array to 32bit
    if((PyObject *) py_P_col != Py_None){
        P_col = malloc(P_ne * sizeof(int));
        long int *P_col_long = (long int *) PyArray_DATA(py_P_col);
        for(int i = 0; i < P_ne; i++) P_col[i] = (int) P_col_long[i];
    }

    // Convert 64bit integer P_ptr array to 32bit
    if((PyObject *) py_P_ptr != Py_None){
        P_ptr = malloc((n+1) * sizeof(int));
        long int *P_ptr_long = (long int *) PyArray_DATA(py_P_ptr);
        for(int i = 0; i < m+1; i++) P_ptr[i] = (int) P_ptr_long[i];
    }

    // Reset control options
    nls_reset_control(&control, &data, &status);

    // Update NLS control options
    if(!nls_update_control(&control, py_options))
        return NULL;

    // Call nls_import
    nls_import(&control, &data, &status, n, m,
                J_type, J_ne, J_row, J_col, J_ptr,
                H_type, H_ne, H_row, H_col, H_ptr,
                P_type, P_ne, P_row, P_col, P_ptr, w);

    // Free allocated memory
    if(J_row != NULL) free(J_row);
    if(J_col != NULL) free(J_col);
    if(J_ptr != NULL) free(J_ptr);
    if(H_row != NULL) free(H_row);
    if(H_col != NULL) free(H_col);
    if(H_ptr != NULL) free(H_ptr);
    if(P_row != NULL) free(P_row);
    if(P_col != NULL) free(P_col);
    if(P_ptr != NULL) free(P_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   NLS_SOLVE   -*-*-*-*-*-*-*-*

static PyObject* py_nls_solve(PyObject *self, PyObject *args){
    PyArrayObject *py_x, *py_c, *py_g;
    PyObject *temp_c, *temp_j, *temp_h, *temp_hprods;
    double *x, *c, *g;
    int n, m, J_ne, H_ne, P_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    if(!PyArg_ParseTuple(args, "iiOOOOiO|iOiO", &n, &m,
                         &py_x, &py_c, &py_g, &temp_c, &J_ne, &temp_j,
                         &H_ne, &temp_h, &P_ne, &temp_hprods))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("x", py_x, n))
        return NULL;

    // Get array data pointer
    x = (double *) PyArray_DATA(py_x);

    // Check that functions are callable
    if(!(
        check_callable(temp_c) &&
        check_callable(temp_j) &&
        check_callable(temp_h) &&
        check_callable(temp_hprods)
        ))
        return NULL;

    // Store functions
    Py_XINCREF(temp_c);            /* Add a reference to new callback */
    Py_XDECREF(py_eval_c);         /* Dispose of previous callback */
    py_eval_c = temp_c;            /* Remember new callback */
    Py_XINCREF(temp_j);            /* Add a reference to new callback */
    Py_XDECREF(py_eval_j);         /* Dispose of previous callback */
    py_eval_j = temp_j;            /* Remember new callback */
    Py_XINCREF(temp_h);            /* Add a reference to new callback */
    Py_XDECREF(py_eval_h);         /* Dispose of previous callback */
    py_eval_h = temp_h;            /* Remember new callback */
    Py_XINCREF(temp_hprods);       /* Add a reference to new callback */
    Py_XDECREF(py_eval_hprods);    /* Dispose of previous callback */
    py_eval_hprods = temp_hprods;  /* Remember new callback */

    // Create empty C arrays for c and g
    double c[m];
    double g[n];

    // Call nls_solve_direct
    status = 1; // set status to 1 on entry
    nls_solve_with_mat(&data, NULL, &status, n, m, x, c, g, eval_c,
                       J_ne, eval_j, H_ne, eval_h, P_ne, eval_hprods);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;
    // Wrap C arrays as NumPy arrays
    npy_intp cdim[] = {m}; // size of c
    PyObject *py_c = PyArray_SimpleNewFromData(1, cdim,
                        NPY_DOUBLE, (void *) c); // create NumPy c array
    npy_intp gdim[] = {n}; // size of g
    PyObject *py_g = PyArray_SimpleNewFromData(1, gdim,
                        NPY_DOUBLE, (void *) g); // create NumPy g array

    // Return x and g
    return Py_BuildValue("OOO", py_x, py_g, py_c);
}

//  *-*-*-*-*-*-*-*-*-*-   NLS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_nls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call nls_information
    nls_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = nls_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   NLS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_nls_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call nls_terminate
    nls_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE NLS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* nls python module method table */
static PyMethodDef nls_module_methods[] = {
    {"initialize", (PyCFunction) py_nls_initialize, METH_NOARGS,NULL},
    {"load", (PyCFunction) py_nls_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve", (PyCFunction) py_nls_solve, METH_VARARGS, NULL},
    {"information", (PyCFunction) py_nls_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_nls_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* nls python module documentation */

PyDoc_STRVAR(nls_module_doc,

"The nls package uses a regularization method to find a (local) unconstrained\n"
"minimizer of a differentiable weighted sum-of-squares objective function\n"
"$$\mathbf{f(x) :=\n"
"   \frac{1}{2} \sum_{i=1}^m w_i c_i^2(x) \equiv rac{1}{2} \|c(x)\|^2_W}$$\n"
"of many variables $f{x}$ involving positive weights $w_i$, $i=1,\ldots,m$.\n"
"The method offers the choice of direct and iterative solution of the key\n"
"regularization subproblems, and is most suitable for large problems.\n"
"First derivatives of the residual function $c(x)$ are required, and if\n"
"second derivatives of the $c_i(x)$ can be calculated, they may be exploited.\n"
"\n"
"See $GALAHAD/html/Python/nls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* nls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "nls",               /* name of module */
   nls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   nls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_nls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
