//* \file dgo_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.0 - 2024-06-15 AT 11:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_DGO PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. September 19th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_dgo.h"

/* Nested UGO control and inform prototypes */
bool ugo_update_control(struct ugo_control_type *control,
                        PyObject *py_options);
PyObject* ugo_make_options_dict(const struct ugo_control_type *control);
PyObject* ugo_make_inform_dict(const struct ugo_inform_type *inform);
//bool trb_update_control(struct trb_control_type *control,
//                        PyObject *py_options);
//PyObject* trb_make_inform_dict(const struct trb_inform_type *inform);
bool hash_update_control(struct hash_control_type *control,
                         PyObject *py_options);
PyObject* hash_make_options_dict(const struct hash_control_type *control);
PyObject* hash_make_inform_dict(const struct hash_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct dgo_control_type control;  // control struct
static struct dgo_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   CALLBACK FUNCTIONS    -*-*-*-*-*-*-*-*-*-*

/* Python eval_* function pointers */
static PyObject *py_eval_f = NULL;
static PyObject *py_eval_g = NULL;
static PyObject *py_eval_h = NULL;
static PyObject *dgo_solve_return = NULL;

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
static bool dgo_update_control(struct dgo_control_type *control,
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
        if(strcmp(key_name, "max_evals") == 0){
            if(!parse_int_option(value, "max_evals",
                                 &control->max_evals))
                return false;
            continue;
        }
        if(strcmp(key_name, "dictionary_size") == 0){
            if(!parse_int_option(value, "dictionary_size",
                                 &control->dictionary_size))
                return false;
            continue;
        }
        if(strcmp(key_name, "alive_unit") == 0){
            if(!parse_int_option(value, "alive_unit",
                                 &control->alive_unit))
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
        if(strcmp(key_name, "lipschitz_lower_bound") == 0){
            if(!parse_double_option(value, "lipschitz_lower_bound",
                                 &control->lipschitz_lower_bound))
                return false;
            continue;
        }
        if(strcmp(key_name, "lipschitz_reliability") == 0){
            if(!parse_double_option(value, "lipschitz_reliability",
                                 &control->lipschitz_reliability))
                return false;
            continue;
        }
        if(strcmp(key_name, "lipschitz_control") == 0){
            if(!parse_double_option(value, "lipschitz_control",
                                 &control->lipschitz_control))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_length") == 0){
            if(!parse_double_option(value, "stop_length",
                                 &control->stop_length))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_f") == 0){
            if(!parse_double_option(value, "stop_f",
                                 &control->stop_f))
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
        if(strcmp(key_name, "prune") == 0){
            if(!parse_bool_option(value, "prune",
                                 &control->prune))
                return false;
            continue;
        }
        if(strcmp(key_name, "perform_local_optimization") == 0){
            if(!parse_bool_option(value, "perform_local_optimization",
                                 &control->perform_local_optimization))
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
                                  control->prefix,
                                  sizeof(control->prefix)))
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

        // Parse nested control
        //if(strcmp(key_name, "trb_options") == 0){
        //    if(!trb_update_control(&control->trb_control, value))
        //        return false;
        //    continue;
        //}
        if(strcmp(key_name, "ugo_options") == 0){
            if(!ugo_update_control(&control->ugo_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "hash_options") == 0){
            if(!hash_update_control(&control->hash_control, value))
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
static PyObject* dgo_make_options_dict(const struct dgo_control_type *control){
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
    PyDict_SetItemString(py_options, "max_evals",
                         PyLong_FromLong(control->max_evals));
    PyDict_SetItemString(py_options, "dictionary_size",
                         PyLong_FromLong(control->dictionary_size));
    PyDict_SetItemString(py_options, "alive_unit",
                         PyLong_FromLong(control->alive_unit));
    PyDict_SetItemString(py_options, "alive_file",
                         PyUnicode_FromString(control->alive_file));
    PyDict_SetItemString(py_options, "infinity",
                         PyFloat_FromDouble(control->infinity));
    PyDict_SetItemString(py_options, "lipschitz_lower_bound",
                         PyFloat_FromDouble(control->lipschitz_lower_bound));
    PyDict_SetItemString(py_options, "lipschitz_reliability",
                         PyFloat_FromDouble(control->lipschitz_reliability));
    PyDict_SetItemString(py_options, "lipschitz_control",
                         PyFloat_FromDouble(control->lipschitz_control));
    PyDict_SetItemString(py_options, "stop_length",
                         PyFloat_FromDouble(control->stop_length));
    PyDict_SetItemString(py_options, "stop_f",
                         PyFloat_FromDouble(control->stop_f));
    PyDict_SetItemString(py_options, "obj_unbounded",
                         PyFloat_FromDouble(control->obj_unbounded));
    PyDict_SetItemString(py_options, "cpu_time_limit",
                         PyFloat_FromDouble(control->cpu_time_limit));
    PyDict_SetItemString(py_options, "clock_time_limit",
                         PyFloat_FromDouble(control->clock_time_limit));
    PyDict_SetItemString(py_options, "hessian_available",
                         PyBool_FromLong(control->hessian_available));
    PyDict_SetItemString(py_options, "prune",
                         PyBool_FromLong(control->prune));
    PyDict_SetItemString(py_options, "perform_local_optimization",
                         PyBool_FromLong(control->perform_local_optimization));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "hash_options",
                         hash_make_options_dict(&control->hash_control));
    PyDict_SetItemString(py_options, "ugo_options",
                         ugo_make_options_dict(&control->ugo_control));
    // PyDict_SetItemString(py_options, "trb_options",
    //                     trb_make_options_dict(&control->trb_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* dgo_make_time_dict(const struct dgo_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries
    PyDict_SetItemString(py_time, "total", PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "univariate_global",
           PyFloat_FromDouble(time->univariate_global));
    PyDict_SetItemString(py_time, "multivariate_local",
           PyFloat_FromDouble(time->multivariate_local));
    PyDict_SetItemString(py_time, "clock_total",
           PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_univariate_global",
           PyFloat_FromDouble(time->clock_univariate_global));
    PyDict_SetItemString(py_time, "clock_multivariate_local",
           PyFloat_FromDouble(time->clock_multivariate_local));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* dgo_make_inform_dict(const struct dgo_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    // Set int inform entries
    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "iter",
                         PyLong_FromLong(inform->iter));
    PyDict_SetItemString(py_inform, "f_eval",
                         PyLong_FromLong(inform->f_eval));
    PyDict_SetItemString(py_inform, "g_eval",
                         PyLong_FromLong(inform->g_eval));
    PyDict_SetItemString(py_inform, "h_eval",
                         PyLong_FromLong(inform->h_eval));

    // Set float/double inform entries
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "norm_pg",
                         PyFloat_FromDouble(inform->norm_pg));
    PyDict_SetItemString(py_inform, "length_ratio",
                         PyFloat_FromDouble(inform->length_ratio));
    PyDict_SetItemString(py_inform, "f_gap",
                         PyFloat_FromDouble(inform->f_gap));

    // Set bool inform entries
    //PyDict_SetItemString(py_inform, "used_grad", PyBool_FromLong(inform->used_grad));

    // Set char inform entries
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "why_stop",
                         PyUnicode_FromString(inform->why_stop));
     // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time", dgo_make_time_dict(&inform->time));

    // Set TRB, UGO and LHS nested dictionaries
    //PyDict_SetItemString(py_inform, "trb_inform",
    //                     trb_make_inform_dict(&inform->trb_inform));
    PyDict_SetItemString(py_inform, "ugo_inform",
                         ugo_make_inform_dict(&inform->ugo_inform));
    PyDict_SetItemString(py_inform, "hash_inform",
                         hash_make_inform_dict(&inform->hash_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   DGO_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_dgo_initialize(PyObject *self){

    // Call dgo_initialize
    dgo_initialize(&data, &control, &status);

    // Record that DGO has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = dgo_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   DGO_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_dgo_load(PyObject *self, PyObject *args, PyObject *keywds){
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
    dgo_reset_control(&control, &data, &status);

    // Update DGO control options
    if(!dgo_update_control(&control, py_options))
        return NULL;

    // Call dgo_import
    dgo_import(&control, &data, &status, n, x_l, x_u, H_type, H_ne,
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

//  *-*-*-*-*-*-*-*-*-*-   DGO_SOLVE   -*-*-*-*-*-*-*-*

static PyObject* py_dgo_solve(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_x;
    PyObject *temp_f, *temp_g, *temp_h;
    double *x;
    int n, H_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n","H_ne","x","eval_f","eval_g","eval_h",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiOOOO", kwlist, &n, &H_ne, &py_x,
                                    &temp_f, &temp_g, &temp_h))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("x", py_x, n))
        return NULL;

    // Get array data pointers
    x = (double *) PyArray_DATA(py_x);

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

    // Create NumPy output array
    npy_intp ndim[] = {n}; // size of g
    PyArrayObject *py_g =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *g = (double *) PyArray_DATA(py_g);

    // Call dgo_solve_direct
    status = 1; // set status to 1 on entry
    dgo_solve_with_mat(&data, NULL, &status, n, x, g, H_ne, eval_f, eval_g,
                       eval_h, NULL, NULL);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x and g
    dgo_solve_return = Py_BuildValue("OO", py_x, py_g);
    Py_XINCREF(dgo_solve_return);
    return dgo_solve_return;
}

//  *-*-*-*-*-*-*-*-*-*-   DGO_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_dgo_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call dgo_information
    dgo_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = dgo_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   DGO_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_dgo_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call dgo_terminate
    dgo_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE DGO PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* dgo python module method table */
static PyMethodDef dgo_module_methods[] = {
    {"initialize", (PyCFunction) py_dgo_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_dgo_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve", (PyCFunction) py_dgo_solve, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_dgo_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_dgo_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* dgo python module documentation */
PyDoc_STRVAR(dgo_module_doc,
"The dgo package uses a deterministic partition-and-bound trust-region\n"
"method to find an approximation to the global minimizer of a\n"
"differentiable objective function $f(x)$ of n variables $x$,\n"
"subject to simple bounds $x^l <= x <= x^u$ on the variables.\n"
"Here, any of the components of the vectors of bounds $x^l$ and $x^u$\n"
"may be infinite. The method offers the choice of direct and\n"
"iterative solution of the key trust-region subproblems, and\n"
"is suitable for large problems. First derivatives are required,\n"
"and if second derivatives can be calculated, they will be exploited -\n"
"if the product of second derivatives with a vector may be found but\n"
"not the derivatives themselves, that may also be exploited.\n"
"\n"
"Although there are theoretical guarantees, these may require a large\n"
"number of evaluations as the dimension and nonconvexity increase.\n"
"The alternative package ``bgo`` may sometimes be preferred.\n"
"\n"
"See $GALAHAD/html/Python/dgo.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* dgo python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "dgo",               /* name of module */
   dgo_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   dgo_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_dgo(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
