//* \file tru_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2022-11-22 AT 10:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_TRU PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_tru.h"

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
static struct tru_control_type control;  // control struct
static struct tru_inform_type inform;    // inform struct
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
static bool tru_update_control(struct tru_control_type *control,
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
        if(strcmp(key_name, "stop_g_absolute") == 0){
            if(!parse_double_option(value, "stop_g_absolute",
                                  &control->stop_g_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_g_relative") == 0){
            if(!parse_double_option(value, "stop_g_relative",
                                  &control->stop_g_relative))
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
        //if(strcmp(key_name, "dps_options") == 0){
        //    if(!dps_update_control(&control->dps_control, value))
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
        //if(strcmp(key_name, "lms_cont_options") == 0){
        //    if(!lms_update_control(&control->lms_control_prec, value))
        //        return false;
        //    continue;
        //}
        //if(strcmp(key_name, "sec_options") == 0){
        //    if(!sec_update_control(&control->sec_control, value))
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
static PyObject* tru_make_time_dict(const struct tru_time_type *time){
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
static PyObject* tru_make_inform_dict(const struct tru_inform_type *inform){
    PyObject *py_inform = PyDict_New();

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
    PyDict_SetItemString(py_inform, "f_eval",
                         PyLong_FromLong(inform->f_eval));
    PyDict_SetItemString(py_inform, "g_eval",
                         PyLong_FromLong(inform->g_eval));
    PyDict_SetItemString(py_inform, "h_eval",
                         PyLong_FromLong(inform->h_eval));
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
    PyDict_SetItemString(py_inform, "factorization_average",
                         PyFloat_FromDouble(inform->factorization_average));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "norm_g",
                         PyFloat_FromDouble(inform->norm_g));
    PyDict_SetItemString(py_inform, "radius",
                         PyFloat_FromDouble(inform->radius));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         tru_make_time_dict(&inform->time));
    // Set TRS, DPS, GLTR, PSLS, LMS, SEC and SHA nested dictionaries
    //PyDict_SetItemString(py_inform, "trs_inform",
    //                     trs_make_inform_dict(&inform->trs_inform));
    //PyDict_SetItemString(py_inform, "dps_inform",
    //                     dps_make_inform_dict(&inform->dps_inform));
    //PyDict_SetItemString(py_inform, "gltr_inform",
    //                     gltr_make_inform_dict(&inform->gltr_inform));
    //PyDict_SetItemString(py_inform, "psls_inform",
    //                     psls_make_inform_dict(&inform->psls_inform));
    //PyDict_SetItemString(py_inform, "lms_inform",
    //                     lms_make_inform_dict(&inform->lms_inform));
    //PyDict_SetItemString(py_inform, "lms_info_inform",
    //                     lms_make_inform_dict(&inform->lms_info_inform));
    //PyDict_SetItemString(py_inform, "sec_inform",
    //                     sec_make_inform_dict(&inform->sec_inform));
    //PyDict_SetItemString(py_inform, "sha_inform",
    //                     sha_make_inform_dict(&inform->sha_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   TRU_INITIALIZE    -*-*-*-*-*-*-*-*-*-*


static PyObject* py_tru_initialize(PyObject *self){

    // Call tru_initialize
    tru_initialize(&data, &control, &status);

    // Record that TRU has been initialised
    init_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   TRU_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_tru_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_H_row, *py_H_col, *py_H_ptr;
    PyObject *py_options = NULL;
    int *H_row = NULL, *H_col = NULL, *H_ptr = NULL;
    const char *H_type;
    int n, H_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","H_type","H_ne",
                             "H_row","H_col","H_ptr","options",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "isiOOO|O", kwlist, &n,
                                    &H_type, &H_ne, &py_H_row,
                                    &py_H_col, &py_H_ptr, &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
//    if((
    if(!(
        check_array_int("H_row", py_H_row, H_ne) &&
        check_array_int("H_col", py_H_col, H_ne) &&
        check_array_int("H_ptr", py_H_ptr, n+1)
        ))
        return NULL;

    // Get array data pointers

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
    tru_reset_control(&control, &data, &status);

    // Update TRU control options
    if(!tru_update_control(&control, py_options))
        return NULL;

    // Call tru_import
    tru_import(&control, &data, &status, n, H_type, H_ne, H_row, H_col, H_ptr);

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

//  *-*-*-*-*-*-*-*-*-*-   TRU_SOLVE   -*-*-*-*-*-*-*-*

static PyObject* py_tru_solve(PyObject *self, PyObject *args){
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

    // Call tru_solve_direct
    status = 1; // set status to 1 on entry
    tru_solve_with_mat(&data, NULL, &status, n, x, g, H_ne, eval_f, eval_g,
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

//  *-*-*-*-*-*-*-*-*-*-   TRU_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_tru_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call tru_information
    tru_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = tru_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   TRU_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_tru_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call tru_terminate
    tru_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE TRU PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* tru python module method table */
static PyMethodDef tru_module_methods[] = {
    {"initialize", (PyCFunction) py_tru_initialize, METH_NOARGS,NULL},
    {"load", (PyCFunction) py_tru_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve", (PyCFunction) py_tru_solve, METH_VARARGS, NULL},
    {"information", (PyCFunction) py_tru_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_tru_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* tru python module documentation */

PyDoc_STRVAR(tru_module_doc,
"The tru package uses a trust-region method to find a (local)\n"
"minimizer of a differentiable objective function f(x) of \n"
"many variables x. The method offers the choice of \n"
"direct and iterative solution of the key subproblems, and\n"
"is most suitable for large problems. First derivatives are required,\n"
"and if second derivatives can be calculated, they will be exploited.\n"
"\n"
"See $GALAHAD/html/Python/tru.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* tru python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "tru",               /* name of module */
   tru_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   tru_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_tru(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
