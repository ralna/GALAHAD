//* \file expo_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.3 - 2025-07-27 AT 15:10 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_EXPO PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 5.3. July 27th 2025
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_expo.h"

/* Nested BSC, TRU and SSLS control and inform prototypes */
bool bsc_update_control(struct bsc_control_type *control,
                        PyObject *py_options);
PyObject* bsc_make_options_dict(const struct bsc_control_type *control);
PyObject* bsc_make_inform_dict(const struct bsc_inform_type *inform);
bool tru_update_control(struct tru_control_type *control,
                        PyObject *py_options);
PyObject* tru_make_options_dict(const struct tru_control_type *control);
PyObject* tru_make_inform_dict(const struct tru_inform_type *inform);
bool ssls_update_control(struct ssls_control_type *control,
                         PyObject *py_options);
PyObject* ssls_make_options_dict(const struct ssls_control_type *control);
PyObject* ssls_make_inform_dict(const struct ssls_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct expo_control_type control;  // control struct
static struct expo_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   CALLBACK FUNCTIONS    -*-*-*-*-*-*-*-*-*-*

/* Python eval_* function pointers */
static PyObject *py_eval_fc = NULL;
static PyObject *py_eval_gj = NULL;
static PyObject *py_eval_hl = NULL;
static PyObject *expo_solve_return = NULL;

/* C eval_* function wrappers */
static int eval_fc(int n, int m, const double x[], double f, double c[],
                   const void *userdata){

    // Wrap input array as NumPy array
    npy_intp xdim[] = {n};
    PyObject *py_x = PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(O)", py_x);

    // Call Python eval_fc
    PyObject *result = PyObject_CallObject(py_eval_fc, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Extract eval_fc return values (one double and one array)
    PyObject *py_c;
    if(!PyArg_ParseTuple(result, "dO", f, py_c)){
        PyErr_SetString(PyExc_TypeError,
        "unable to parse eval_fc return values");
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Copy data into c
    const double *cval = (double *) PyArray_DATA((PyArrayObject*) py_c);
    for(int i=0; i<m; i++) {
        c[i] = cval[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

static int eval_gj(int n, int m, int jne, const double x[],
                   double g[], double jval[], const void *userdata){

    // Wrap input array as NumPy array
    npy_intp xdim[] = {n};
    PyArrayObject *py_x = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(O)", py_x);

    // Call Python eval_gj
    PyObject *result = PyObject_CallObject(py_eval_gj, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

     // Extract eval_gj return values (two arrays)
    PyObject *py_g, *py_j;
    if(!PyArg_ParseTuple(result, "OO", py_g, py_j)){
        PyErr_SetString(PyExc_TypeError,
        "unable to parse eval_gj return values");
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Copy data into g and j
    const double *gval = (double *) PyArray_DATA((PyArrayObject*) py_g);
    for(int i=0; i<n; i++) {
        g[i] = gval[i];
    }
    const double *val = (double *) PyArray_DATA((PyArrayObject*) py_j);
    for(int i=0; i<jne; i++) {
        jval[i] = val[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

static int eval_hl(int n, int m, int hne, const double x[], const double y[],
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

    // Call Python eval_hl
    PyObject *result = PyObject_CallObject(py_eval_hl, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Check return value is of correct type, size, and shape
    if(!check_array_double("eval_hl return value",
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

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool expo_update_control(struct expo_control_type *control,
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
        if(strcmp(key_name, "max_it") == 0){
            if(!parse_int_option(value, "max_it",
                                  &control->max_it))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_eval") == 0){
            if(!parse_int_option(value, "max_eval",
                                  &control->max_eval))
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
        if(strcmp(key_name, "update_multipliers_itmin") == 0){
            if(!parse_int_option(value, "update_multipliers_itmin",
                                  &control->update_multipliers_itmin))
                return false;
            continue;
        }
        if(strcmp(key_name, "update_multipliers_tol") == 0){
            if(!parse_double_option(value, "update_multipliers_tol",
                                  &control->update_multipliers_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "infinity") == 0){
            if(!parse_double_option(value, "infinity",
                                  &control->infinity))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_abs_p") == 0){
            if(!parse_double_option(value, "stop_abs_p",
                                  &control->stop_abs_p))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_rel_p") == 0){
            if(!parse_double_option(value, "stop_rel_p",
                                  &control->stop_rel_p))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_abs_d") == 0){
            if(!parse_double_option(value, "stop_abs_d",
                                  &control->stop_abs_d))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_rel_d") == 0){
            if(!parse_double_option(value, "stop_rel_d",
                                  &control->stop_rel_d))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_abs_c") == 0){
            if(!parse_double_option(value, "stop_abs_c",
                                  &control->stop_abs_c))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_rel_c") == 0){
            if(!parse_double_option(value, "stop_rel_c",
                                  &control->stop_rel_c))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_s") == 0){
            if(!parse_double_option(value, "stop_s",
                                  &control->stop_s))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_mu") == 0){
            if(!parse_double_option(value, "initial_mu",
                                  &control->initial_mu))
                return false;
            continue;
        }
        if(strcmp(key_name, "mu_reduce") == 0){
            if(!parse_double_option(value, "mu_reduce",
                                  &control->mu_reduce))
                return false;
            continue;
        }
        if(strcmp(key_name, "obj_unbounded") == 0){
            if(!parse_double_option(value, "obj_unbounded",
                                  &control->obj_unbounded))
                return false;
            continue;
        }
        if(strcmp(key_name, "try_advanced_start") == 0){
            if(!parse_double_option(value, "try_advanced_start",
                                  &control->try_advanced_start))
                return false;
            continue;
        }
        if(strcmp(key_name, "try_sqp_start") == 0){
            if(!parse_double_option(value, "try_sqp_start",
                                  &control->try_sqp_start))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_advanced_start") == 0){
            if(!parse_double_option(value, "stop_advanced_start",
                                  &control->stop_advanced_start))
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
        if(strcmp(key_name, "hessian_available") == 0){
            if(!parse_bool_option(value, "hessian_available",
                                  &control->hessian_available))
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
        if(strcmp(key_name, "bsc_options") == 0){
            if(!bsc_update_control(&control->bsc_control, value))
                return false;
            continue;
        }
        }
        if(strcmp(key_name, "tru_options") == 0){
            if(!tru_update_control(&control->tru_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "ssls_options") == 0){
            if(!ssls_update_control(&control->ssls_control, value))
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

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
static PyObject* expo_make_options_dict(const struct expo_control_type *control){
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
    PyDict_SetItemString(py_options, "max_it",
                         PyLong_FromLong(control->max_it));
    PyDict_SetItemString(py_options, "max_eval",
                         PyLong_FromLong(control->max_eval));
    PyDict_SetItemString(py_options, "alive_unit",
                         PyLong_FromLong(control->alive_unit));
    PyDict_SetItemString(py_options, "alive_file",
                         PyUnicode_FromString(control->alive_file));
    PyDict_SetItemString(py_options, "update_multipliers_itmin",
                         PyLong_FromLong(control->update_multipliers_itmin));
    PyDict_SetItemString(py_options, "update_multipliers_tol",
                         PyFloat_FromDouble(control->update_multipliers_tol));
    PyDict_SetItemString(py_options, "infinity",
                         PyFloat_FromDouble(control->infinity));
    PyDict_SetItemString(py_options, "stop_abs_p",
                         PyFloat_FromDouble(control->stop_abs_p));
    PyDict_SetItemString(py_options, "stop_rel_p",
                         PyFloat_FromDouble(control->stop_rel_p));
    PyDict_SetItemString(py_options, "stop_abs_d",
                         PyFloat_FromDouble(control->stop_abs_d));
    PyDict_SetItemString(py_options, "stop_rel_d",
                         PyFloat_FromDouble(control->stop_rel_d));
    PyDict_SetItemString(py_options, "stop_abs_c",
                         PyFloat_FromDouble(control->stop_abs_c));
    PyDict_SetItemString(py_options, "stop_rel_c",
                         PyFloat_FromDouble(control->stop_rel_c));
    PyDict_SetItemString(py_options, "stop_s",
                         PyFloat_FromDouble(control->stop_s));
    PyDict_SetItemString(py_options, "initial_mu",
                         PyFloat_FromDouble(control->initial_mu));
    PyDict_SetItemString(py_options, "mu_reduce",
                         PyFloat_FromDouble(control->mu_reduce));
    PyDict_SetItemString(py_options, "obj_unbounded",
                         PyFloat_FromDouble(control->obj_unbounded));
    PyDict_SetItemString(py_options, "try_advanced_start",
                         PyFloat_FromDouble(control->try_advanced_start));
    PyDict_SetItemString(py_options, "try_sqp_start",
                         PyFloat_FromDouble(control->try_sqp_start));
    PyDict_SetItemString(py_options, "stop_advanced_start",
                         PyFloat_FromDouble(control->stop_advanced_start));
    PyDict_SetItemString(py_options, "cpu_time_limit",
                         PyFloat_FromDouble(control->cpu_time_limit));
    PyDict_SetItemString(py_options, "clock_time_limit",
                         PyFloat_FromDouble(control->clock_time_limit));
    PyDict_SetItemString(py_options, "hessian_available",
                         PyBool_FromLong(control->hessian_available));
    PyDict_SetItemString(py_options, "subproblem_direct",
                         PyBool_FromLong(control->subproblem_direct));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "bsc_options",
                         bsc_make_options_dict(&control->bsc_control));
    PyDict_SetItemString(py_options, "tru_options",
                         tru_make_options_dict(&control->tru_control));
    PyDict_SetItemString(py_options, "ssls_options",
                         ssls_make_options_dict(&control->ssls_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* expo_make_time_dict(const struct expo_time_type *time){
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
static PyObject* expo_make_inform_dict(const struct expo_inform_type *inform){
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
    PyDict_SetItemString(py_inform, "fc_eval",
                         PyLong_FromLong(inform->fc_eval));
    PyDict_SetItemString(py_inform, "gj_eval",
                         PyLong_FromLong(inform->gj_eval));
    PyDict_SetItemString(py_inform, "hl_eval",
                         PyLong_FromLong(inform->hl_eval));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "primal_infeasibility",
                         PyFloat_FromDouble(inform->primal_infeasibility));
    PyDict_SetItemString(py_inform, "dual_infeasibility",
                         PyFloat_FromDouble(inform->dual_infeasibility));
    PyDict_SetItemString(py_inform, "complementary_slackness",
                         PyFloat_FromDouble(inform->complementary_slackness));
    PyDict_SetItemString(py_inform, "time",
                         expo_make_time_dict(&inform->time));
    PyDict_SetItemString(py_inform, "bsc_inform",
                         bsc_make_inform_dict(&inform->bsc_inform));
    PyDict_SetItemString(py_inform, "tru_inform",
                         tru_make_inform_dict(&inform->tru_inform));
    PyDict_SetItemString(py_inform, "ssls_inform",
                         ssls_make_inform_dict(&inform->ssls_inform));
    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   EXPO_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_expo_initialize(PyObject *self){

    // Call expo_initialize
    expo_initialize(&data, &control, &inform);

    // Record that EXPO has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = expo_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   EXPO_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_expo_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_J_row, *py_J_col, *py_J_ptr;
    PyArrayObject *py_H_row, *py_H_col, *py_H_ptr;
    PyObject *py_options = NULL;
    int *J_row = NULL, *J_col = NULL, *J_ptr = NULL;
    int *H_row = NULL, *H_col = NULL, *H_ptr = NULL;
    const char *J_type, *H_type;
    int n, m, J_ne, H_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","m",
                             "J_type","J_ne","J_row","J_col","J_ptr",
                             "H_type","H_ne","H_row","H_col","H_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOO|siOOOO",
                                    kwlist, &n, &m,
                                    &J_type, &J_ne, &py_J_row,
                                    &py_J_col, &py_J_ptr,
                                    &H_type, &H_ne, &py_H_row,
                                    &py_H_col, &py_H_ptr,
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

    // Reset control options
    expo_reset_control(&control, &data, &status);

    // Update EXPO control options
    if(!expo_update_control(&control, py_options))
        return NULL;

    // Call expo_import
    expo_import(&control, &data, &status, n, m,
                J_type, J_ne, J_row, J_col, J_ptr,
                H_type, H_ne, H_row, H_col, H_ptr);

    // Free allocated memory
    if(J_row != NULL) free(J_row);
    if(J_col != NULL) free(J_col);
    if(J_ptr != NULL) free(J_ptr);
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

//  *-*-*-*-*-*-*-*-*-*-   EXPO_SOLVE   -*-*-*-*-*-*-*-*

static PyObject* py_expo_solve(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_c_l, *py_c_u, *py_x_l, *py_x_u, *py_x;

    PyObject *temp_fc, *temp_gj, *temp_hl;
    double *c_l, *c_u, *x_l, *x_u, *x;
    int n, m, J_ne, H_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "m", "J_ne", "H_ne",
                             "c_l", "c_u", "x_l", "x_u", "x",
                             "eval_fc", "eval_gj", "eval_hl", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiiiOOOOOOOO", kwlist,
                                    &n, &m, &J_ne, &H_ne,
                                    &py_c_l, &py_c_u, &py_x_l, &py_x_u, &py_x,
                                    &temp_fc, &temp_gj, &temp_hl))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("c_l", py_c_l, m))
        return NULL;
    if(!check_array_double("c_u", py_c_u, m))
        return NULL;
    if(!check_array_double("x_l", py_x_l, n))
        return NULL;
    if(!check_array_double("x_u", py_x_u, n))
        return NULL;
    if(!check_array_double("x", py_x, n))
        return NULL;

    // Get array data pointer
    c_l = (double *) PyArray_DATA(py_c_l);
    c_u = (double *) PyArray_DATA(py_c_u);
    x_l = (double *) PyArray_DATA(py_x_l);
    x_u = (double *) PyArray_DATA(py_x_u);
    x = (double *) PyArray_DATA(py_x);

    // Check that functions are callable
    if(!(
        check_callable(temp_fc) &&
        check_callable(temp_gj) &&
        check_callable(temp_hl)
        ))
        return NULL;

    // Store functions
    Py_XINCREF(temp_fc);            /* Add a reference to new callback */
    Py_XDECREF(py_eval_fc);         /* Dispose of previous callback */
    py_eval_fc = temp_fc;            /* Remember new callback */
    Py_XINCREF(temp_gj);            /* Add a reference to new callback */
    Py_XDECREF(py_eval_gj);         /* Dispose of previous callback */
    py_eval_gj = temp_gj;            /* Remember new callback */
    Py_XINCREF(temp_hl);            /* Add a reference to new callback */
    Py_XDECREF(py_eval_hl);         /* Dispose of previous callback */
    py_eval_hl = temp_hl;            /* Remember new callback */

   // Create NumPy output arrays
    npy_intp mdim[] = {m}; // size of y and c
    PyArrayObject *py_y =
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_DOUBLE);
    double *y = (double *) PyArray_DATA(py_y);
    npy_intp ndim[] = {n}; // size of z and gl
    PyArrayObject *py_z =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *z = (double *) PyArray_DATA(py_z);
    PyArrayObject *py_c =
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_DOUBLE);
    double *c = (double *) PyArray_DATA(py_c);
    PyArrayObject *py_gl =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *gl = (double *) PyArray_DATA(py_gl);

    // Call expo_solve_direct
    status = 1; // set status to 1 on entry
    expo_solve_hessian_direct(&data, NULL, &status, n, m, J_ne, H_ne,
                        c_l, c_u, x_l, x_u, x, y, z, c, gl,
                        eval_fc, eval_gj, eval_hl);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x, y, z, c and gl
    expo_solve_return = Py_BuildValue("OOOOO", py_x, py_y, py_z, py_c, py_gl);
    Py_XINCREF(expo_solve_return);
    return expo_solve_return;
}

//  *-*-*-*-*-*-*-*-*-*-   EXPO_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_expo_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call expo_information
    expo_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = expo_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   EXPO_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_expo_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call expo_terminate
    expo_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE EXPO PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* expo python module method table */
static PyMethodDef expo_module_methods[] = {
    {"initialize", (PyCFunction) py_expo_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_expo_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve", (PyCFunction) py_expo_solve, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_expo_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_expo_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* expo python module documentation */

PyDoc_STRVAR(expo_module_doc,

"The expo package uses a regularization method to find a (local) unconstrained\n"
"minimizer of a differentiable weighted sum-of-squares objective function\n"
"f(x) :=\n"
"   1/2 sum_{i=1}^m w_i c_i^2(x) == 1/2 ||c(x)||^2_W\n"
"of many variables f{x} involving positive weights w_i, i=1,...,m.\n"
"The method offers the choice of direct and iterative solution of the key\n"
"regularization subproblems, and is most suitable for large problems.\n"
"First derivatives of the residual function c(x) are required, and if\n"
"second derivatives of the c_i(x) can be calculated, they may be exploited.\n"
"\n"
"See $GALAHAD/html/Python/expo.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* expo python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "expo",               /* name of module */
   expo_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   expo_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_expo(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
