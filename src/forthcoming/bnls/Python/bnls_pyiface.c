//* \file bnls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.1 - 2024-07-14 AT 14:10 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_BNLS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 5.1. July 14th 2024
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_bnls.h"

/* Nested RQS, GLRT, PSLS, LMS and SHA control and inform prototypes */
bool rqs_update_control(struct rqs_control_type *control,
                        PyObject *py_options);
PyObject* rqs_make_options_dict(const struct rqs_control_type *control);
PyObject* rqs_make_inform_dict(const struct rqs_inform_type *inform);
bool glrt_update_control(struct glrt_control_type *control,
                         PyObject *py_options);
PyObject* glrt_make_options_dict(const struct glrt_control_type *control);
PyObject* glrt_make_inform_dict(const struct glrt_inform_type *inform);
bool psls_update_control(struct psls_control_type *control,
                         PyObject *py_options);
PyObject* psls_make_options_dict(const struct psls_control_type *control);
PyObject* psls_make_inform_dict(const struct psls_inform_type *inform);
bool bsc_update_control(struct bsc_control_type *control,
                        PyObject *py_options);
PyObject* bsc_make_options_dict(const struct bsc_control_type *control);
PyObject* bsc_make_inform_dict(const struct bsc_inform_type *inform);
bool roots_update_control(struct roots_control_type *control,
                        PyObject *py_options);
PyObject* roots_make_options_dict(const struct roots_control_type *control);
PyObject* roots_make_inform_dict(const struct roots_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct bnls_control_type control;  // control struct
static struct bnls_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   CALLBACK FUNCTIONS    -*-*-*-*-*-*-*-*-*-*

/* Python eval_* function pointers */
static PyObject *py_eval_c = NULL;
static PyObject *py_eval_j = NULL;
static PyObject *py_eval_h = NULL;
static PyObject *py_eval_hprods = NULL;
static PyObject *bnls_solve_return = NULL;
//static PyObject *py_c = NULL;
//static PyObject *py_g = NULL;

/* C eval_* function wrappers */
static int eval_c(int n, int m, const double x[], double c[],
                  const void *userdata){

    // Wrap input array as NumPy array
    npy_intp xdim[] = {n};
    PyObject *py_x = PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(O)", py_x);

    // Call Python eval_c
    PyObject *result = PyObject_CallObject(py_eval_c, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Get return value data pointer and copy data intoc
    const double *cval = (double *) PyArray_DATA((PyArrayObject*) result);
    for(int i=0; i<m; i++) {
        c[i] = cval[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

static int eval_j(int n, int m, int jne, const double x[], double jval[],
                  const void *userdata){

    // Wrap input array as NumPy array
    npy_intp xdim[] = {n};
    PyArrayObject *py_x = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(O)", py_x);

    // Call Python eval_j
    PyObject *result = PyObject_CallObject(py_eval_j, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Check return value is of correct type, size, and shape
    if(!check_array_double("eval_j return value",
                           (PyArrayObject*) result, jne)){
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Get return value data pointer and copy data into jval
    const double *val = (double *) PyArray_DATA((PyArrayObject*) result);
    for(int i=0; i<jne; i++) {
        jval[i] = val[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

static int eval_h(int n, int m, int hne, const double x[], const double y[],
                  double hval[], const void *userdata){

    // Wrap input arrays as NumPy arrays
    npy_intp xdim[] = {n};
    PyArrayObject *py_x = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);
    npy_intp ydim[] = {m};
    PyArrayObject *py_y = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, ydim, NPY_DOUBLE, (void *) y);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(OO)", py_x, py_y);

    // Call Python eval_h
    PyObject *result = PyObject_CallObject(py_eval_h, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Check return value is of correct type, size, and shape
    if(!check_array_double("eval_h return value",
                           (PyArrayObject*) result, hne)){
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Get return value data pointer and copy data into hval
    const double *val = (double *) PyArray_DATA((PyArrayObject*) result);
    for(int i=0; i<hne; i++) {
        hval[i] = val[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

static int eval_hprods(int n, int m, int pne, const double x[], const double v[],
                  double pval[], bool got_h, const void *userdata){

    // Wrap input arrays as NumPy arrays
    npy_intp xdim[] = {n};
    PyArrayObject *py_x = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    npy_intp vdim[] = {m};
    PyArrayObject *py_v = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, vdim, NPY_DOUBLE, (void *) v);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(OO)", py_x, py_v);

    // Call Python eval_h
    PyObject *result = PyObject_CallObject(py_eval_hprods, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(py_v);    // Free py_v memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Check return value is of correct type, size, and shape
    if(!check_array_double("eval_hprods return value",
                           (PyArrayObject*) result, pne)){
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Get return value data pointer and copy data into hval
    const double *val = (double *) PyArray_DATA((PyArrayObject*) result);
    for(int i=0; i<pne; i++) {
        pval[i] = val[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

//  *-*-*-*-*-*-*-*-*-*-   UPDATE SUBPROBLEM CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the subproblem control options: use C defaults but update any
   passed via Python*/
static bool bnls_update_subproblem_control(
                               struct bnls_subproblem_control_type *control,
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

        // Parse each option
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
        if(strcmp(key_name, "alive_file") == 0){
            if(!parse_char_option(value, "alive_file",
                                  control->alive_file,
                                  sizeof(control->alive_file)))
                return false;
            continue;
        }
        if(strcmp(key_name, "jacobian_available") == 0){
            if(!parse_int_option(value, "jacobian_available",
                                  &control->jacobian_available))
                return false;
            continue;
        }
        if(strcmp(key_name, "hessian_available") == 0){
            if(!parse_int_option(value, "hessian_available",
                                  &control->hessian_available))
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
        if(strcmp(key_name, "non_monotone") == 0){
            if(!parse_int_option(value, "non_monotone",
                                  &control->non_monotone))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_update_strategy") == 0){
            if(!parse_int_option(value, "weight_update_strategy",
                                  &control->weight_update_strategy))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_c_absolute") == 0){
            if(!parse_double_option(value, "stop_c_absolute",
                                  &control->stop_c_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_c_relative") == 0){
            if(!parse_double_option(value, "stop_c_relative",
                                  &control->stop_c_relative))
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
        if(strcmp(key_name, "power") == 0){
            if(!parse_double_option(value, "power",
                                  &control->power))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_weight") == 0){
            if(!parse_double_option(value, "initial_weight",
                                  &control->initial_weight))
                return false;
            continue;
        }
        if(strcmp(key_name, "minimum_weight") == 0){
            if(!parse_double_option(value, "minimum_weight",
                                  &control->minimum_weight))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_inner_weight") == 0){
            if(!parse_double_option(value, "initial_inner_weight",
                                  &control->initial_inner_weight))
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
        if(strcmp(key_name, "weight_decrease_min") == 0){
            if(!parse_double_option(value, "weight_decrease_min",
                                  &control->weight_decrease_min))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_decrease") == 0){
            if(!parse_double_option(value, "weight_decrease",
                                  &control->weight_decrease))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_increase") == 0){
            if(!parse_double_option(value, "weight_increase",
                                  &control->weight_increase))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_increase_max") == 0){
            if(!parse_double_option(value, "weight_increase_max",
                                  &control->weight_increase_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "reduce_gap") == 0){
            if(!parse_double_option(value, "reduce_gap",
                                  &control->reduce_gap))
                return false;
            continue;
        }
        if(strcmp(key_name, "tiny_gap") == 0){
            if(!parse_double_option(value, "tiny_gap",
                                  &control->tiny_gap))
                return false;
            continue;
        }
        if(strcmp(key_name, "large_root") == 0){
            if(!parse_double_option(value, "large_root",
                                  &control->large_root))
                return false;
            continue;
        }
        if(strcmp(key_name, "switch_to_newton") == 0){
            if(!parse_double_option(value, "switch_to_newton",
                                  &control->switch_to_newton))
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
        if(strcmp(key_name, "subproblem_direct") == 0){
            if(!parse_bool_option(value, "subproblem_direct",
                                  &control->subproblem_direct))
                return false;
            continue;
        }
        if(strcmp(key_name, "renormalize_weight") == 0){
            if(!parse_bool_option(value, "renormalize_weight",
                                  &control->renormalize_weight))
                return false;
            continue;
        }
        if(strcmp(key_name, "magic_step") == 0){
            if(!parse_bool_option(value, "magic_step",
                                  &control->magic_step))
                return false;
            continue;
        }
        if(strcmp(key_name, "print_obj") == 0){
            if(!parse_bool_option(value, "print_obj",
                                  &control->print_obj))
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
                                  control->prefix,
                                  sizeof(control->prefix)))
                return false;
            continue;
        }

        if(strcmp(key_name, "rqs_options") == 0){
            if(!rqs_update_control(&control->rqs_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "glrt_options") == 0){
            if(!glrt_update_control(&control->glrt_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "psls_options") == 0){
            if(!psls_update_control(&control->psls_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "bsc_options") == 0){
            if(!bsc_update_control(&control->bsc_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "roots_options") == 0){
            if(!roots_update_control(&control->roots_control, value))
                return false;
            continue;
        }

        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
          "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool bnls_update_control(struct bnls_control_type *control,
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

        // Parse each option
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
        if(strcmp(key_name, "alive_file") == 0){
            if(!parse_char_option(value, "alive_file",
                                  control->alive_file,
                                  sizeof(control->alive_file)))
                return false;
            continue;
        }
        if(strcmp(key_name, "jacobian_available") == 0){
            if(!parse_int_option(value, "jacobian_available",
                                  &control->jacobian_available))
                return false;
            continue;
        }
        if(strcmp(key_name, "hessian_available") == 0){
            if(!parse_int_option(value, "hessian_available",
                                  &control->hessian_available))
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
        if(strcmp(key_name, "non_monotone") == 0){
            if(!parse_int_option(value, "non_monotone",
                                  &control->non_monotone))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_update_strategy") == 0){
            if(!parse_int_option(value, "weight_update_strategy",
                                  &control->weight_update_strategy))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_c_absolute") == 0){
            if(!parse_double_option(value, "stop_c_absolute",
                                  &control->stop_c_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_c_relative") == 0){
            if(!parse_double_option(value, "stop_c_relative",
                                  &control->stop_c_relative))
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
        if(strcmp(key_name, "power") == 0){
            if(!parse_double_option(value, "power",
                                  &control->power))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_weight") == 0){
            if(!parse_double_option(value, "initial_weight",
                                  &control->initial_weight))
                return false;
            continue;
        }
        if(strcmp(key_name, "minimum_weight") == 0){
            if(!parse_double_option(value, "minimum_weight",
                                  &control->minimum_weight))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_inner_weight") == 0){
            if(!parse_double_option(value, "initial_inner_weight",
                                  &control->initial_inner_weight))
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
        if(strcmp(key_name, "weight_decrease_min") == 0){
            if(!parse_double_option(value, "weight_decrease_min",
                                  &control->weight_decrease_min))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_decrease") == 0){
            if(!parse_double_option(value, "weight_decrease",
                                  &control->weight_decrease))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_increase") == 0){
            if(!parse_double_option(value, "weight_increase",
                                  &control->weight_increase))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_increase_max") == 0){
            if(!parse_double_option(value, "weight_increase_max",
                                  &control->weight_increase_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "reduce_gap") == 0){
            if(!parse_double_option(value, "reduce_gap",
                                  &control->reduce_gap))
                return false;
            continue;
        }
        if(strcmp(key_name, "tiny_gap") == 0){
            if(!parse_double_option(value, "tiny_gap",
                                  &control->tiny_gap))
                return false;
            continue;
        }
        if(strcmp(key_name, "large_root") == 0){
            if(!parse_double_option(value, "large_root",
                                  &control->large_root))
                return false;
            continue;
        }
        if(strcmp(key_name, "switch_to_newton") == 0){
            if(!parse_double_option(value, "switch_to_newton",
                                  &control->switch_to_newton))
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
        if(strcmp(key_name, "subproblem_direct") == 0){
            if(!parse_bool_option(value, "subproblem_direct",
                                  &control->subproblem_direct))
                return false;
            continue;
        }
        if(strcmp(key_name, "renormalize_weight") == 0){
            if(!parse_bool_option(value, "renormalize_weight",
                                  &control->renormalize_weight))
                return false;
            continue;
        }
        if(strcmp(key_name, "magic_step") == 0){
            if(!parse_bool_option(value, "magic_step",
                                  &control->magic_step))
                return false;
            continue;
        }
        if(strcmp(key_name, "print_obj") == 0){
            if(!parse_bool_option(value, "print_obj",
                                  &control->print_obj))
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
                                  control->prefix,
                                  sizeof(control->prefix)))
                return false;
            continue;
        }
        if(strcmp(key_name, "rqs_options") == 0){
            if(!rqs_update_control(&control->rqs_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "glrt_options") == 0){
            if(!glrt_update_control(&control->glrt_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "psls_options") == 0){
            if(!psls_update_control(&control->psls_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "bsc_options") == 0){
            if(!bsc_update_control(&control->bsc_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "roots_options") == 0){
            if(!roots_update_control(&control->roots_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "subproblem_options") == 0){
            if(!bnls_update_subproblem_control(&control->subproblem_control,
                                              value))
                return false;
            continue;
        }

        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
          "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE SUBPROBLEM OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
static PyObject* bnls_make_subproblem_options_dict(const struct
                                                  bnls_subproblem_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "start_print",
                         PyLong_FromLong(control->start_print));
    PyDict_SetItemString(py_options, "stop_print",
                         PyLong_FromLong(control->stop_print));
    PyDict_SetItemString(py_options, "print_gap",
                         PyLong_FromLong(control->print_gap));
    PyDict_SetItemString(py_options, "maxit",
                         PyLong_FromLong(control->maxit));
    PyDict_SetItemString(py_options, "alive_unit",
                         PyLong_FromLong(control->alive_unit));
    PyDict_SetItemString(py_options, "alive_file",
                         PyUnicode_FromString(control->alive_file));
    PyDict_SetItemString(py_options, "jacobian_available",
                         PyLong_FromLong(control->jacobian_available));
    PyDict_SetItemString(py_options, "hessian_available",
                         PyLong_FromLong(control->hessian_available));
    PyDict_SetItemString(py_options, "model",
                         PyLong_FromLong(control->model));
    PyDict_SetItemString(py_options, "norm",
                         PyLong_FromLong(control->norm));
    PyDict_SetItemString(py_options, "non_monotone",
                         PyLong_FromLong(control->non_monotone));
    PyDict_SetItemString(py_options, "weight_update_strategy",
                         PyLong_FromLong(control->weight_update_strategy));
    PyDict_SetItemString(py_options, "stop_c_absolute",
                         PyFloat_FromDouble(control->stop_c_absolute));
    PyDict_SetItemString(py_options, "stop_c_relative",
                         PyFloat_FromDouble(control->stop_c_relative));
    PyDict_SetItemString(py_options, "stop_g_absolute",
                         PyFloat_FromDouble(control->stop_g_absolute));
    PyDict_SetItemString(py_options, "stop_g_relative",
                         PyFloat_FromDouble(control->stop_g_relative));
    PyDict_SetItemString(py_options, "stop_s",
                         PyFloat_FromDouble(control->stop_s));
    PyDict_SetItemString(py_options, "power",
                         PyFloat_FromDouble(control->power));
    PyDict_SetItemString(py_options, "initial_weight",
                         PyFloat_FromDouble(control->initial_weight));
    PyDict_SetItemString(py_options, "minimum_weight",
                         PyFloat_FromDouble(control->minimum_weight));
    PyDict_SetItemString(py_options, "initial_inner_weight",
                         PyFloat_FromDouble(control->initial_inner_weight));
    PyDict_SetItemString(py_options, "eta_successful",
                         PyFloat_FromDouble(control->eta_successful));
    PyDict_SetItemString(py_options, "eta_very_successful",
                         PyFloat_FromDouble(control->eta_very_successful));
    PyDict_SetItemString(py_options, "eta_too_successful",
                         PyFloat_FromDouble(control->eta_too_successful));
    PyDict_SetItemString(py_options, "weight_decrease_min",
                         PyFloat_FromDouble(control->weight_decrease_min));
    PyDict_SetItemString(py_options, "weight_decrease",
                         PyFloat_FromDouble(control->weight_decrease));
    PyDict_SetItemString(py_options, "weight_increase",
                         PyFloat_FromDouble(control->weight_increase));
    PyDict_SetItemString(py_options, "weight_increase_max",
                         PyFloat_FromDouble(control->weight_increase_max));
    PyDict_SetItemString(py_options, "reduce_gap",
                         PyFloat_FromDouble(control->reduce_gap));
    PyDict_SetItemString(py_options, "tiny_gap",
                         PyFloat_FromDouble(control->tiny_gap));
    PyDict_SetItemString(py_options, "large_root",
                         PyFloat_FromDouble(control->large_root));
    PyDict_SetItemString(py_options, "switch_to_newton",
                         PyFloat_FromDouble(control->switch_to_newton));
    PyDict_SetItemString(py_options, "cpu_time_limit",
                         PyFloat_FromDouble(control->cpu_time_limit));
    PyDict_SetItemString(py_options, "clock_time_limit",
                         PyFloat_FromDouble(control->clock_time_limit));
    PyDict_SetItemString(py_options, "subproblem_direct",
                         PyBool_FromLong(control->subproblem_direct));
    PyDict_SetItemString(py_options, "renormalize_weight",
                         PyBool_FromLong(control->renormalize_weight));
    PyDict_SetItemString(py_options, "magic_step",
                         PyBool_FromLong(control->magic_step));
    PyDict_SetItemString(py_options, "print_obj",
                         PyBool_FromLong(control->print_obj));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "rqs_options",
                         rqs_make_options_dict(&control->rqs_control));
    PyDict_SetItemString(py_options, "glrt_options",
                         glrt_make_options_dict(&control->glrt_control));
    PyDict_SetItemString(py_options, "psls_options",
                         psls_make_options_dict(&control->psls_control));
    PyDict_SetItemString(py_options, "bsc_options",
                         bsc_make_options_dict(&control->bsc_control));
    PyDict_SetItemString(py_options, "roots_options",
                         roots_make_options_dict(&control->roots_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
static PyObject* bnls_make_options_dict(const struct bnls_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "start_print",
                         PyLong_FromLong(control->start_print));
    PyDict_SetItemString(py_options, "stop_print",
                         PyLong_FromLong(control->stop_print));
    PyDict_SetItemString(py_options, "print_gap",
                         PyLong_FromLong(control->print_gap));
    PyDict_SetItemString(py_options, "maxit",
                         PyLong_FromLong(control->maxit));
    PyDict_SetItemString(py_options, "alive_unit",
                         PyLong_FromLong(control->alive_unit));
    PyDict_SetItemString(py_options, "alive_file",
                         PyUnicode_FromString(control->alive_file));
    PyDict_SetItemString(py_options, "jacobian_available",
                         PyLong_FromLong(control->jacobian_available));
    PyDict_SetItemString(py_options, "hessian_available",
                         PyLong_FromLong(control->hessian_available));
    PyDict_SetItemString(py_options, "model",
                         PyLong_FromLong(control->model));
    PyDict_SetItemString(py_options, "norm",
                         PyLong_FromLong(control->norm));
    PyDict_SetItemString(py_options, "non_monotone",
                         PyLong_FromLong(control->non_monotone));
    PyDict_SetItemString(py_options, "weight_update_strategy",
                         PyLong_FromLong(control->weight_update_strategy));
    PyDict_SetItemString(py_options, "stop_c_absolute",
                         PyFloat_FromDouble(control->stop_c_absolute));
    PyDict_SetItemString(py_options, "stop_c_relative",
                         PyFloat_FromDouble(control->stop_c_relative));
    PyDict_SetItemString(py_options, "stop_g_absolute",
                         PyFloat_FromDouble(control->stop_g_absolute));
    PyDict_SetItemString(py_options, "stop_g_relative",
                         PyFloat_FromDouble(control->stop_g_relative));
    PyDict_SetItemString(py_options, "stop_s",
                         PyFloat_FromDouble(control->stop_s));
    PyDict_SetItemString(py_options, "power",
                         PyFloat_FromDouble(control->power));
    PyDict_SetItemString(py_options, "initial_weight",
                         PyFloat_FromDouble(control->initial_weight));
    PyDict_SetItemString(py_options, "minimum_weight",
                         PyFloat_FromDouble(control->minimum_weight));
    PyDict_SetItemString(py_options, "initial_inner_weight",
                         PyFloat_FromDouble(control->initial_inner_weight));
    PyDict_SetItemString(py_options, "eta_successful",
                         PyFloat_FromDouble(control->eta_successful));
    PyDict_SetItemString(py_options, "eta_very_successful",
                         PyFloat_FromDouble(control->eta_very_successful));
    PyDict_SetItemString(py_options, "eta_too_successful",
                         PyFloat_FromDouble(control->eta_too_successful));
    PyDict_SetItemString(py_options, "weight_decrease_min",
                         PyFloat_FromDouble(control->weight_decrease_min));
    PyDict_SetItemString(py_options, "weight_decrease",
                         PyFloat_FromDouble(control->weight_decrease));
    PyDict_SetItemString(py_options, "weight_increase",
                         PyFloat_FromDouble(control->weight_increase));
    PyDict_SetItemString(py_options, "weight_increase_max",
                         PyFloat_FromDouble(control->weight_increase_max));
    PyDict_SetItemString(py_options, "reduce_gap",
                         PyFloat_FromDouble(control->reduce_gap));
    PyDict_SetItemString(py_options, "tiny_gap",
                         PyFloat_FromDouble(control->tiny_gap));
    PyDict_SetItemString(py_options, "large_root",
                         PyFloat_FromDouble(control->large_root));
    PyDict_SetItemString(py_options, "switch_to_newton",
                         PyFloat_FromDouble(control->switch_to_newton));
    PyDict_SetItemString(py_options, "cpu_time_limit",
                         PyFloat_FromDouble(control->cpu_time_limit));
    PyDict_SetItemString(py_options, "clock_time_limit",
                         PyFloat_FromDouble(control->clock_time_limit));
    PyDict_SetItemString(py_options, "subproblem_direct",
                         PyBool_FromLong(control->subproblem_direct));
    PyDict_SetItemString(py_options, "renormalize_weight",
                         PyBool_FromLong(control->renormalize_weight));
    PyDict_SetItemString(py_options, "magic_step",
                         PyBool_FromLong(control->magic_step));
    PyDict_SetItemString(py_options, "print_obj",
                         PyBool_FromLong(control->print_obj));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "rqs_options",
                         rqs_make_options_dict(&control->rqs_control));
    PyDict_SetItemString(py_options, "glrt_options",
                         glrt_make_options_dict(&control->glrt_control));
    PyDict_SetItemString(py_options, "psls_options",
                         psls_make_options_dict(&control->psls_control));
    PyDict_SetItemString(py_options, "bsc_options",
                         bsc_make_options_dict(&control->bsc_control));
    PyDict_SetItemString(py_options, "roots_options",
                         roots_make_options_dict(&control->roots_control));
    PyDict_SetItemString(py_options, "subproblem_options",
                         bnls_make_subproblem_options_dict(&control->subproblem_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* bnls_make_time_dict(const struct bnls_time_type *time){
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

//  *-*-*-*-*-*-*-*-*-*-   MAKE SUBPROBLEM INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the subproblem inform struct from C and turn it into a python dictionary */
static PyObject* bnls_make_subproblem_inform_dict(
    const struct bnls_subproblem_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "bad_eval",
                         PyUnicode_FromString(inform->bad_eval));
    PyDict_SetItemString(py_inform, "iter",
                         PyLong_FromLong(inform->iter));
    PyDict_SetItemString(py_inform, "cg_iter",
                         PyLong_FromLong(inform->cg_iter));
    PyDict_SetItemString(py_inform, "c_eval",
                         PyLong_FromLong(inform->c_eval));
    PyDict_SetItemString(py_inform, "j_eval",
                         PyLong_FromLong(inform->j_eval));
    PyDict_SetItemString(py_inform, "h_eval",
                         PyLong_FromLong(inform->h_eval));
    PyDict_SetItemString(py_inform, "factorization_max",
                         PyLong_FromLong(inform->factorization_max));
    PyDict_SetItemString(py_inform, "factorization_status",
                         PyLong_FromLong(inform->factorization_status));
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
    PyDict_SetItemString(py_inform, "norm_c",
                         PyFloat_FromDouble(inform->norm_c));
    PyDict_SetItemString(py_inform, "norm_g",
                         PyFloat_FromDouble(inform->norm_g));
    PyDict_SetItemString(py_inform, "weight",
                         PyFloat_FromDouble(inform->weight));
    PyDict_SetItemString(py_inform, "time",
                         bnls_make_time_dict(&inform->time));
    PyDict_SetItemString(py_inform, "rqs_inform",
                         rqs_make_inform_dict(&inform->rqs_inform));
    PyDict_SetItemString(py_inform, "glrt_inform",
                         glrt_make_inform_dict(&inform->glrt_inform));
    PyDict_SetItemString(py_inform, "psls_inform",
                         psls_make_inform_dict(&inform->psls_inform));
    PyDict_SetItemString(py_inform, "bsc_inform",
                         bsc_make_inform_dict(&inform->bsc_inform));
    PyDict_SetItemString(py_inform, "roots_inform",
                         roots_make_inform_dict(&inform->roots_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* bnls_make_inform_dict(const struct bnls_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "bad_eval",
                         PyUnicode_FromString(inform->bad_eval));
    PyDict_SetItemString(py_inform, "iter",
                         PyLong_FromLong(inform->iter));
    PyDict_SetItemString(py_inform, "cg_iter",
                         PyLong_FromLong(inform->cg_iter));
    PyDict_SetItemString(py_inform, "c_eval",
                         PyLong_FromLong(inform->c_eval));
    PyDict_SetItemString(py_inform, "j_eval",
                         PyLong_FromLong(inform->j_eval));
    PyDict_SetItemString(py_inform, "h_eval",
                         PyLong_FromLong(inform->h_eval));
    PyDict_SetItemString(py_inform, "factorization_max",
                         PyLong_FromLong(inform->factorization_max));
    PyDict_SetItemString(py_inform, "factorization_status",
                         PyLong_FromLong(inform->factorization_status));
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
    PyDict_SetItemString(py_inform, "norm_c",
                         PyFloat_FromDouble(inform->norm_c));
    PyDict_SetItemString(py_inform, "norm_g",
                         PyFloat_FromDouble(inform->norm_g));
    PyDict_SetItemString(py_inform, "weight",
                         PyFloat_FromDouble(inform->weight));
    PyDict_SetItemString(py_inform, "time",
                         bnls_make_time_dict(&inform->time));
    PyDict_SetItemString(py_inform, "rqs_inform",
                         rqs_make_inform_dict(&inform->rqs_inform));
    PyDict_SetItemString(py_inform, "glrt_inform",
                         glrt_make_inform_dict(&inform->glrt_inform));
    PyDict_SetItemString(py_inform, "psls_inform",
                         psls_make_inform_dict(&inform->psls_inform));
    PyDict_SetItemString(py_inform, "bsc_inform",
                         bsc_make_inform_dict(&inform->bsc_inform));
    PyDict_SetItemString(py_inform, "roots_inform",
                         roots_make_inform_dict(&inform->roots_inform));
    PyDict_SetItemString(py_inform, "subproblem_inform",
                         bnls_make_subproblem_inform_dict(
                         &inform->subproblem_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   BNLS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_bnls_initialize(PyObject *self){

    // Call bnls_initialize
    bnls_initialize(&data, &control, &inform);

    // Record that BNLS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = bnls_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   BNLS_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_bnls_load(PyObject *self, PyObject *args, PyObject *keywds){
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
                             "P_type","P_ne","P_row","P_col","P_ptr",
                             "w","options",NULL};

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
    bnls_reset_control(&control, &data, &status);

    // Update BNLS control options
    if(!bnls_update_control(&control, py_options))
        return NULL;

    // Call bnls_import
    bnls_import(&control, &data, &status, n, m,
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

//  *-*-*-*-*-*-*-*-*-*-   BNLS_SOLVE   -*-*-*-*-*-*-*-*

static PyObject* py_bnls_solve(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_x;
    PyObject *temp_c, *temp_j, *temp_h, *temp_hprods;
    double *x;
    int n, m, J_ne, H_ne, P_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "m", "x", "eval_c", "J_ne", "eval_j",
                             "H_ne", "eval_h", "P_ne", "eval_hprod", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiOOiO|iOiO", kwlist, &n, &m,
                                    &py_x, &temp_c, &J_ne, &temp_j,
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

   // Create NumPy output arrays
    npy_intp mdim[] = {m}; // size of c
    PyArrayObject *py_c =
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_DOUBLE);
    double *c = (double *) PyArray_DATA(py_c);
    npy_intp ndim[] = {n}; // size of g
    PyArrayObject *py_g =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *g = (double *) PyArray_DATA(py_g);

    // Call bnls_solve_direct
    status = 1; // set status to 1 on entry
    bnls_solve_with_mat(&data, NULL, &status, n, m, x, c, g, eval_c,
                       J_ne, eval_j, H_ne, eval_h, P_ne, eval_hprods);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x, c and g
    bnls_solve_return = Py_BuildValue("OOO", py_x, py_c, py_g);
    Py_XINCREF(bnls_solve_return);
    return bnls_solve_return;
}

//  *-*-*-*-*-*-*-*-*-*-   BNLS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_bnls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call bnls_information
    bnls_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = bnls_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   BNLS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_bnls_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call bnls_terminate
    bnls_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE BNLS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* bnls python module method table */
static PyMethodDef bnls_module_methods[] = {
    {"initialize", (PyCFunction) py_bnls_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_bnls_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve", (PyCFunction) py_bnls_solve, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_bnls_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_bnls_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* bnls python module documentation */

PyDoc_STRVAR(bnls_module_doc,

"The bnls package uses a regularization method to find a (local) unconstrained\n"
"minimizer of a differentiable weighted sum-of-squares objective function\n"
"f(x) :=\n"
"   1/2 sum_{i=1}^m w_i c_i^2(x) == 1/2 ||c(x)||^2_W\n"
"of many variables f{x} involving positive weights w_i, i=1,...,m.\n"
"The method offers the choice of direct and iterative solution of the key\n"
"regularization subproblems, and is most suitable for large problems.\n"
"First derivatives of the residual function c(x) are required, and if\n"
"second derivatives of the c_i(x) can be calculated, they may be exploited.\n"
"\n"
"See $GALAHAD/html/Python/bnls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* bnls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "bnls",               /* name of module */
   bnls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   bnls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_bnls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
