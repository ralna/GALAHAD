//* \file dgo_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2022-09-19 AT 13:50 GMT.
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
PyObject* ugo_make_inform_dict(const struct ugo_inform_type *inform);
//bool trb_update_control(struct trb_control_type *control,
//                        PyObject *py_options);
//PyObject* trb_make_inform_dict(const struct trb_inform_type *inform);
//bool hash_update_control(struct hash_control_type *control,
//                         PyObject *py_options);
//PyObject* hash_make_inform_dict(const struct hash_inform_type *inform);

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
        if(strcmp(key_name, "cpu_time_limit ") == 0){
            if(!parse_double_option(value, "cpu_time_limit ",
                                    &control->cpu_time_limit ))
                return false;
            continue;
        }
        if(strcmp(key_name, "clock_time_limit ") == 0){
            if(!parse_double_option(value, "clock_time_limit ",
                                    &control->clock_time_limit ))
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
        //if(strcmp(key_name, "hash_options") == 0){
        //    if(!hash_update_control(&control->hash_control, value))
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
    //PyDict_SetItemString(py_inform, "hash_inform",
    //                     hash_make_inform_dict(&inform->hash_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   DGO_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_dgo_initialize_doc,
"dgo.initialize()\n"
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
"    max_evals : int\n"
"      the maximum number of function evaluations made.\n"
"    dictionary_size : int\n"
"      the size of the initial hash dictionary.\n"
"    alive_unit : int\n"
"      removal of the file alive_file from unit alive_unit\n"
"      terminates execution.\n"
"    alive_file : str\n"
"      see alive_unit.\n"
"    infinity : float\n"
"      any bound larger than infinity in modulus will be regarded as\n"
"      infinite.\n"
"    lipschitz_lower_bound : float\n"
"      a small positive constant (<= 1e-6) that ensure that the\n"
"      estimted gradient Lipschitz constant is not too small.\n"
"    lipschitz_reliability : float\n"
"      the Lipschitz reliability parameter, the Lipschiz constant\n"
"      used will be a factor lipschitz_reliability times the largest\n"
"      value observed.\n"
"    lipschitz_control : float\n"
"      the reliablity control parameter, the actual reliability\n"
"      parameter used will be ``lipschitz_reliability`` + MAX( 1, n\n"
"      - 1 ) * ``lipschitz_control`` / iteration.\n"
"    stop_length : float\n"
"      the iteration will stop if the length, delta, of the diagonal\n"
"      in the box with the smallest-found objective function is\n"
"      smaller than ``stop_length`` times that of the original bound\n"
"      box, delta_0.\n"
"    stop_f : float\n"
"      the iteration will stop if the gap between the best objective\n"
"      value found and the smallest lower bound is smaller than\n"
"      ``stop_f``.\n"
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
"    prune : bool\n"
"      should boxes that cannot contain the global minimizer be\n"
"      pruned (i.e., removed from further consideration)?.\n"
"    perform_local_optimization : bool\n"
"      should approximate minimizers be impoved by judicious local\n"
"      minimization?.\n"
"    space_critical : bool\n"
"      if ``space_critical`` is True, every effort will be made to\n"
"      use as little space as possible. This may result in longer\n"
"      computation time.\n"
"    deallocate_error_fatal : bool\n"
"      if ``deallocate_error_fatal`` is True, any array/pointer\n"
"      deallocation error will terminate execution. Otherwise,\n"
"      computation will continue.\n"
"    prefix : str\n"
"      all output lines will be prefixed by the string contained\n"
"      in quotes within ``prefix``, e.g. 'word' (note the qutoes)\n"
"      will result in the prefix word.\n"
"    ugo_options : dict\n"
"      default control options for UGO (see ``ugo.initialize``).\n"
"    hash_options : dict\n"
"      default control options for HASH (see ``hash.initialize``).\n"
"    trb_options : dict\n"
"      default control options for TRB (see ``trb.initialize``).\n"
"\n"
);

static PyObject* py_dgo_initialize(PyObject *self){

    // Call dgo_initialize
    dgo_initialize(&data, &control, &status);

    // Record that DGO has been initialised
    init_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   DGO_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*
//  NB import is a python reserved keyword so changed to load here

PyDoc_STRVAR(py_dgo_load_doc,
"dgo.load(n, x_l, x_u, H_type, H_ne, H_row, H_col, H_ptr, options=None)\n"
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
"    dictionary of control options (see ``dgo.initialize``).\n"
"\n"
);

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

PyDoc_STRVAR(py_dgo_solve_doc,
"x, g = dgo.solve(n, H_ne, x, g, eval_f, eval_g, eval_h)\n"
"\n"
"Find an approximation to the global minimizer of a given function\n"
"subject to simple bounds on the variables using a multistart\n"
"trust-region method.\n"
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
"    in the sparsity pattern in ``dgo.load``.\n"
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

static PyObject* py_dgo_solve(PyObject *self, PyObject *args){
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

//  *-*-*-*-*-*-*-*-*-*-   DGO_INFORMATION   -*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_dgo_information_doc,
"inform = dgo.information()\n"
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
"      * -91\n"
"\n"
"        The hash table used to store the dictionary of vertices\n"
"        of the sub-boxes is full, and there is no room to increase\n"
"        it further\n"
"\n"
"    alloc_status : int\n"
"      the status of the last attempted allocation/deallocation.\n"
"    bad_alloc : str\n"
"      the name of the array for which an allocation/deallocation\n"
"      error ocurred.\n"
"    iter : int\n"
"      the total number of iterations performed.\n"
"    f_eval : int\n"
"      the total number of evaluations of the objective function.\n"
"    g_eval : int\n"
"      the total number of evaluations of the gradient of the\n"
"      objective function.\n"
"    h_eval : int\n"
"      the total number of evaluations of the Hessian of the\n"
"      objective function.\n"
"    obj : float\n"
"      the value of the objective function at the best estimate of\n"
"      the solution determined by ``dgo.solve``.\n"
"    norm_pg : float\n"
"      the norm of the projected gradient of the objective function\n"
"      at the best estimate of the solution determined by ``dgo.solve``.\n"
"    length_ratio : float\n"
"      the ratio of the final to the initial box lengths.\n"
"    f_gap : float\n"
"      the gap between the best objective value found and the lowest\n"
"      bound.\n"
"    why_stop : str\n"
"      why did the iteration stop? This wil be 'D' if the box length\n"
"      is small enough, 'F' if the objective gap is small enough,\n"
"      and ' ' otherwise.\n"
"    time : dict\n"
"      dictionary containing timing information:\n"
"       total : float\n"
"         the total CPU time spent in the package.\n"
"       univariate_global : float\n"
"         the CPU time spent performing univariate global optimization.\n"
"       multivariate_local : float\n"
"         the CPU time spent performing multivariate local optimization.\n"
"       clock_total : float\n"
"         the total clock time spent in the package.\n"
"       clock_univariate_global : float\n"
"         the clock time spent performing univariate global\n"
"         optimization.\n"
"       clock_multivariate_local : float\n"
"         the clock time spent performing multivariate local\n"
"         optimization.\n"
"    ugo_inform : dict\n"
"      inform parameters for UGO (see ``ugo.information``).\n"
"    lhs_inform : dict\n"
"      inform parameters for HASH (see ``hash.information``).\n"
"    trb_inform : dict\n"
"      inform parameters for TRB (see ``trb.information``).\n"
"\n"
);

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

PyDoc_STRVAR(py_dgo_terminate_doc,
"dgo.terminate()\n"
"\n"
"Deallocate all internal private storage\n"
"\n"
);

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
    {"initialize", (PyCFunction) py_dgo_initialize, METH_NOARGS,
      py_dgo_initialize_doc},
    {"load", (PyCFunction) py_dgo_load, METH_VARARGS | METH_KEYWORDS,
      py_dgo_load_doc},
    {"solve", (PyCFunction) py_dgo_solve, METH_VARARGS,
      py_dgo_solve_doc},
    {"information", (PyCFunction) py_dgo_information, METH_NOARGS,
      py_dgo_information_doc},
    {"terminate", (PyCFunction) py_dgo_terminate, METH_NOARGS,
      py_dgo_terminate_doc},
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
"See Section 4 of $GALAHAD/doc/dgo.pdf for a brief description of the\n"
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
"  ``dgo.initialize``\n"
"\n"
"  ``dgo.load``\n"
"\n"
"  ``dgo.solve``\n"
"\n"
"  [``dgo.information``]\n"
"\n"
"  ``dgo.terminate``\n"
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
