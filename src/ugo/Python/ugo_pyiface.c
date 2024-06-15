//* \file ugo_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-03-28 AT 10:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_UGO PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. August 17th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_ugo.h"

/* Module global variables */
static void *data;                       // private internal data
static struct ugo_control_type control;  // control struct
static struct ugo_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   CALLBACK FUNCTIONS    -*-*-*-*-*-*-*-*-*-*

/* Python eval_* function pointers */
static PyObject *py_eval_fgh = NULL;

/* C eval_* function wrappers */
static int eval_fgh(double x, double *f, double *g, double *h,
                    const void *userdata){

    // Build Python argument list (pass x from double)
    PyObject *arglist = Py_BuildValue("(d)", x);

    // Call Python eval_fgh
    PyObject *result =  PyObject_CallObject(py_eval_fgh, arglist);
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Extract return values (three doubles)
    if(!PyArg_ParseTuple(result, "ddd", f, g, h)){
        PyErr_SetString(PyExc_TypeError,
        "unable to parse eval_fgh return values");
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within BGO Python interface
bool ugo_update_control(struct ugo_control_type *control, PyObject *py_options){

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
        if(strcmp(key_name, "initial_points") == 0){
            if(!parse_int_option(value, "initial_points",
                                 &control->initial_points))
                return false;
            continue;
        }
        if(strcmp(key_name, "storage_increment") == 0){
            if(!parse_int_option(value, "storage_increment",
                                 &control->storage_increment))
                return false;
            continue;
        }
        if(strcmp(key_name, "buffer") == 0){
            if(!parse_int_option(value, "buffer",
                                 &control->buffer))
                return false;
            continue;
        }
        if(strcmp(key_name, "lipschitz_estimate_used") == 0){
            if(!parse_int_option(value, "lipschitz_estimate_used",
                                 &control->lipschitz_estimate_used))
                return false;
            continue;
        }
        if(strcmp(key_name, "next_interval_selection") == 0){
            if(!parse_int_option(value, "next_interval_selection",
                                 &control->next_interval_selection))
                return false;
            continue;
        }
        if(strcmp(key_name, "refine_with_newton") == 0){
            if(!parse_int_option(value, "refine_with_newton",
                                 &control->refine_with_newton))
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
        if(strcmp(key_name, "stop_length") == 0){
            if(!parse_double_option(value, "stop_length",
                                    &control->stop_length))
                return false;
            continue;
        }
        if(strcmp(key_name, "small_g_for_newton") == 0){
            if(!parse_double_option(value, "small_g_for_newton",
                                    &control->small_g_for_newton))
                return false;
            continue;
        }
        if(strcmp(key_name, "small_g") == 0){
            if(!parse_double_option(value, "small_g",
                                    &control->small_g))
                return false;
            continue;
        }
        if(strcmp(key_name, "obj_sufficient") == 0){
            if(!parse_double_option(value, "obj_sufficient",
                                    &control->obj_sufficient))
                return false;
            continue;
        }
        if(strcmp(key_name, "global_lipschitz_constant") == 0){
            if(!parse_double_option(value, "global_lipschitz_constant",
                                    &control->global_lipschitz_constant))
                return false;
            continue;
        }
        if(strcmp(key_name, "reliability_parameter") == 0){
            if(!parse_double_option(value, "reliability_parameter",
                                    &control->reliability_parameter))
                return false;
            continue;
        }
        if(strcmp(key_name, "lipschitz_lower_bound") == 0){
            if(!parse_double_option(value, "lipschitz_lower_bound",
                                    &control->lipschitz_lower_bound))
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
        if(strcmp(key_name, "second_derivative_available") == 0){
            if(!parse_bool_option(value, "second_derivative_available",
                                  &control->space_critical))
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

        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
            "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
// NB not static as it is used for nested inform within BGO Python interface
PyObject* ugo_make_options_dict(const struct ugo_control_type *control){
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
    PyDict_SetItemString(py_options, "initial_points",
                         PyLong_FromLong(control->initial_points));
    PyDict_SetItemString(py_options, "storage_increment",
                         PyLong_FromLong(control->storage_increment));
    PyDict_SetItemString(py_options, "buffer",
                         PyLong_FromLong(control->buffer));
    PyDict_SetItemString(py_options, "lipschitz_estimate_used",
                         PyLong_FromLong(control->lipschitz_estimate_used));
    PyDict_SetItemString(py_options, "next_interval_selection",
                         PyLong_FromLong(control->next_interval_selection));
    PyDict_SetItemString(py_options, "refine_with_newton",
                         PyLong_FromLong(control->refine_with_newton));
    PyDict_SetItemString(py_options, "alive_unit",
                         PyLong_FromLong(control->alive_unit));
    PyDict_SetItemString(py_options, "alive_file",
                         PyUnicode_FromString(control->alive_file));
    PyDict_SetItemString(py_options, "stop_length",
                         PyFloat_FromDouble(control->stop_length));
    PyDict_SetItemString(py_options, "small_g_for_newton",
                         PyFloat_FromDouble(control->small_g_for_newton));
    PyDict_SetItemString(py_options, "small_g",
                         PyFloat_FromDouble(control->small_g));
    PyDict_SetItemString(py_options, "obj_sufficient",
                         PyFloat_FromDouble(control->obj_sufficient));
    PyDict_SetItemString(py_options, "global_lipschitz_constant",
                         PyFloat_FromDouble(control->global_lipschitz_constant))
;
    PyDict_SetItemString(py_options, "reliability_parameter",
                         PyFloat_FromDouble(control->reliability_parameter));
    PyDict_SetItemString(py_options, "lipschitz_lower_bound",
                         PyFloat_FromDouble(control->lipschitz_lower_bound));
    PyDict_SetItemString(py_options, "cpu_time_limit",
                         PyFloat_FromDouble(control->cpu_time_limit));
    PyDict_SetItemString(py_options, "clock_time_limit",
                         PyFloat_FromDouble(control->clock_time_limit));
    PyDict_SetItemString(py_options, "second_derivative_available",
                         PyBool_FromLong(control->second_derivative_available));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* ugo_make_time_dict(const struct ugo_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries
    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested inform within BGO Python interface
PyObject* ugo_make_inform_dict(const struct ugo_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    // Set int inform entries
    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "eval_status",
                         PyLong_FromLong(inform->eval_status));
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

    // Set char inform entries
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time", ugo_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   UGO_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_ugo_initialize(PyObject *self){

    // Call ugo_initialize
    ugo_initialize(&data, &control, &status);

    // Record that UGO has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = ugo_make_options_dict(&control);
    return Py_BuildValue("O", py_options);

    // Return None boilerplate
    // Py_INCREF(Py_None);
    // return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   UGO_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_ugo_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyObject *py_options = NULL;
    double x_l, x_u;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments * Note sentinel at end
    static char *kwlist[] = {"x_l","x_u","options", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "dd|O", kwlist,
                                    &x_l, &x_u, &py_options))
        return NULL;

    // Reset control options
    ugo_reset_control(&control, &data, &status);

    // Update UGO control options
    if(!ugo_update_control(&control, py_options))
       return NULL;

    // Call ugo_import
    ugo_import(&control, &data, &status, &x_l, &x_u);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   UGO_SOLVE   -*-*-*-*-*-*-*-*

static PyObject* py_ugo_solve(PyObject *self, PyObject *args, PyObject *keywds){
    PyObject *temp;
    double x, f, g, h;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional function argument
    static char *kwlist[] = {"eval_fgh", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "O:eval_fgh", kwlist, &temp))
        return NULL;
    if(!check_callable(temp)) // Check that function is callable
        return NULL;
    Py_XINCREF(temp);         /* Add a reference to new callback */
    Py_XDECREF(py_eval_fgh);  /* Dispose of previous callback */
    py_eval_fgh = temp;       /* Remember new callback */

    // Call ugo_solve_direct
    status = 1; // set status to 1 on entry
    ugo_solve_direct(&data, NULL, &status, &x, &f, &g, &h, eval_fgh);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x, f, g, h
    return Py_BuildValue("dddd", x, f, g, h);
}

//  *-*-*-*-*-*-*-*-*-*-   UGO_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_ugo_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call ugo_information
    ugo_information(&data, &inform, &status);

    // Return inform Python dictionary
    PyObject *py_inform = ugo_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   UGO_TERMINATE   -*-*-*-*-*-*-*-*-*-*

/* "Deallocate all internal private storage.\n" */

static PyObject* py_ugo_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call ugo_terminate
    ugo_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE UGO PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* ugo python module method table */
static PyMethodDef ugo_module_methods[] = {
    {"initialize", (PyCFunction) py_ugo_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_ugo_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve", (PyCFunction) py_ugo_solve, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_ugo_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_ugo_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* ugo python module documentation */
PyDoc_STRVAR(ugo_module_doc,
"The ugo package aims to find the global minimizer of a univariate\n"
"twice-continuously differentiable function f(x) of a single\n"
"variable over the finite interval x^l <= x <= x^u. Function\n"
"and derivative values are provided via a subroutine call.\n"
"Second derivatives may be used to advantage if they are available.\n"
"\n"
"See $GALAHAD/html/Python/ugo.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* ugo python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "ugo",               /* name of module */
   ugo_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module, or -1
                           if the module keeps state in global variables */
   ugo_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_ugo(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
