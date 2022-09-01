//* \file ugo_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2022-08-28 AT 09:35 GMT.
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

        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
            "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
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

    // Set float/double inform entries
    //PyDict_SetItemString(py_inform, "obj",
    //                     PyFloat_FromDouble(inform->obj));
    // ... other float/double inform entries ...

    // Set bool inform entries
    //PyDict_SetItemString(py_inform, "used_grad",
    //                     PyBool_FromLong(inform->used_grad));
    // ... other bool inform entries ...

    // Set char inform entries
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time", ugo_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   UGO_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_ugo_initialize_doc,
"ugo.initialize()\n"
"\n"
"Set default option values and initialize private data.\n"
"\n"
"Returns\n"
"-------\n"
"options : dict\n"
"    dictionary containing default control options:\n"
"     error : int\n"
"       error and warning diagnostics occur on stream error.\n"
"     out :  int\n"
"       general output occurs on stream out.\n"
"     print_level : int\n"
"       the level of output required. Possible values are:\n"
"         <= 0 no output, \n"
"         1 a one-line summary for every improvement\n"
"         2 a summary of each iteration\n"
"         >= 3 increasingly verbose (debugging) output.\n"
"     start_print : int\n"
"       any printing will start on this iteration.\n"
"     stop_print : int\n"
"       any printing will stop on this iteration.\n"
"     print_gap : int\n"
"       the number of iterations between printing.\n"
"     maxit : int\n"
"       the maximum number of iterations allowed.\n"
"     initial_point : int\n"
"       the number of initial (uniformly-spaced) evaluation points \n"
"       (<2 reset to 2).\n"
"     storage_increment : int\n"
"       incremenets of storage allocated (less that 1000 will be\n"
"       reset to 1000).\n"
"     buffer : int\n"
"       unit for any out-of-core writing when expanding arrays.\n"
"     lipschitz_estimate_used : int\n"
"       what sort of Lipschitz constant estimate will be used:\n"
"         1 = global contant provided \n"
"         2 = global contant estimated\n"
"         3 = local costants estimated.\n"
"     next_interval_selection : int\n"
"       how is the next interval for examination chosen:\n"
"         1 = traditional\n"
"         2 = local_improvement.\n"
"     refine_with_newton : int\n"
"       try refine_with_newton Newton steps from the vacinity of\n"
"       the global minimizer to try to improve the estimate.\n"
"     alive_unit : int\n"
"       removal of the file alive_file from unit alive_unit\n"
"       terminates execution.\n"
"     alive_file : str\n"
"       see alive_unit.\n"
"     stop_length : float\n"
"       overall convergence tolerances. The iteration will terminate\n"
"       when the step is less than ``stop_length``.\n"
"     small_g_for_newton : float\n"
"       if the absolute value of the gradient is smaller than \n"
"       small_g_for_newton, the next evaluation point may be at a \n"
"       Newton estimate of a local minimizer.\n"
"     small_g : float\n"
"       if the absolute value of the gradient at the end of the interval\n"
"       search is smaller than small_g, no Newton search is necessary.\n"
"     obj_sufficient : float\n"
"       stop if the objective function is smaller than a specified value.\n"
"     global_lipschitz_constant : float\n"
"       the global Lipschitz constant for the gradient\n"
"       (-ve means unknown).\n"
"     reliability_parameter : float\n"
"       the reliability parameter that is used to boost insufficiently\n"
"       large estimates of the Lipschitz constant (-ve means that\n"
"       default values will be chosen depending on whether second\n"
"       derivatives are provided or not).\n"
"     lipschitz_lower_bound : float\n"
"       a lower bound on the Lipscitz constant for the gradient \n"
"       (not zero unless the function is constant).\n"
"     cpu_time_limit : float\n"
"       the maximum CPU time allowed (-ve means infinite).\n"
"     clock_time_limit : float\n"
"       the maximum elapsed clock time allowed (-ve means infinite).\n"
"     second_derivative_available : bool\n"
"       if ``second_derivative_available`` is True, the user must provide\n"
"       them when requested. The package is generally more effective\n"
"       if second derivatives are available.\n"
"     space_critical : bool\n"
"       if ``space_critical`` is True, every effort will be made to\n"
"       use as little space as possible. This may result in longer\n"
"       computation time.\n"
"     deallocate_error_fatal : bool\n"
"       if ``deallocate_error_fatal`` is True, any array/pointer\n"
"       deallocation error will terminate execution. Otherwise,\n"
"       computation will continue.\n"
"    prefix : str\n"
"       all output lines will be prefixed by the string contained\n"
"       in quotes within ``prefix``, e.g. 'word' (note the qutoes)\n"
"       will result in the prefix word.\n"
"\n"
);

static PyObject* py_ugo_initialize(PyObject *self){

    // Call ugo_initialize
    ugo_initialize(&data, &control, &status);

    // Record that UGO has been initialised
    init_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   UGO_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*
//  NB import is a python reserved keyword so changed to load here

PyDoc_STRVAR(py_ugo_load_doc,
"ugo.load(x_l, x_u, options=None)\n"
"\n"
"Import problem data into internal storage prior to solution.\n"
"\n"
"Parameters\n"
"----------\n"
"x_l : double\n"
"    holds the value :math:`x^l` of the lower bound on the optimization\n"
"    variable :math:`x`.\n"
"x_u : double\n"
"    holds the value :math:`x^u` of the upper bound on the optimization\n"
"    variable :math:`x`.\n"
"options : dict, optional\n"
"    dictionary of control options (see ugo.initialize).\n"
"\n"
);

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

PyDoc_STRVAR(py_ugo_solve_doc,
"x, f, g, h = ugo.solve(eval_fgh)\n"
"\n"
"Find an approximation to the global minimizer of a given univariate\n"
" function with a Lipschitz gradient in an interval.\n"
"\n"
"Parameters\n"
"----------\n"
"eval_fgh : callable\n"
"    a user-defined function that must have the signature:\n"
"\n"
"     ``f, g, h = eval_fgh(x)``\n"
"\n"
"    The value of the objective function :math:`f(x)` and its first\n"
"    derivative :math:`f'(x)` evaluated at :math:`x` must be assigned\n"
"    to ``f`` and ``g`` respectively. In addition, if \n"
"    options['second_derivatives_available'] has been set to True\n"
"    when calling ``ugo.load``, the user must also assign the value of\n"
"    the second derivative :math:`f''(x)` to ``h``; it need not be\n"
"    assigned otherwise.\n"
"\n"
"Returns\n"
"-------\n"
"x : double\n"
"    holds the value of the approximate global minimizer :math:`x`\n"
"    after a successful call.\n"
"f : double\n"
"    holds the value of the objective function :math:`f(x)` at the\n"
"    approximate global minimizer :math:`x` after a successful call.\n"
"g : double\n"
"    holds the value of the gradient of the objective function\n"
"    :math:`f'(x)` at the approximate global minimizer :math:`x`\n"
"    after a successful call.\n"
"h : double\n"
"    holds the value of the second derivative of the objective function\n"
"    :math:`f''(x)` at the approximate global minimizer :math:`x` after\n"
"    a successful call.\n"
"\n"
);

static PyObject* py_ugo_solve(PyObject *self, PyObject *args){
    PyObject *temp;
    double x, f, g, h;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional function argument
    if(!PyArg_ParseTuple(args, "O:eval_fgh", &temp))
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

PyDoc_STRVAR(py_ugo_information_doc,
"inform = ugo.information()\n"
"\n"
"Provide output information.\n"
"\n"
"Returns\n"
"-------\n"
"inform : dict\n"
"   dictionary containing output information:\n"
"    status : int\n"
"     return status. Possible values are:\n"
"\n"
"       0   The run was succesful.\n"
"\n"
"      -1   An allocation error occurred. A message indicating the\n"
"           offending array is written on unit control['error'], and the\n"
"           returned allocation status and a string containing the name\n"
"           of the offending array are held in inform['alloc_status']\n"
"           and inform['bad_alloc'] respectively.\n"
"\n"
"      -2   A deallocation error occurred.  A message indicating the\n"
"           offending array is written on unit control['error'] and \n"
"           the returned allocation status and a string containing\n"
"           the name of the offending array are held in \n"
"           inform['alloc_status'] and inform['bad_alloc'] respectively.\n"
"\n"
"      -7   The objective function appears to be unbounded from below.\n"
"\n"
"      -18  Too many iterations have been performed. This may happen if\n"
"           control['maxit'] is too small, but may also be symptomatic\n"
"           of a badly scaled problem.\n"
"\n"
"      -19  The CPU time limit has been reached. This may happen if\n"
"           control['cpu_time_limit'] is too small, but may also be\n"
"           symptomatic of a badly scaled problem.\n"
"\n"
"      -40  The user has forced termination of solver by removing the\n"
"           file named control['alive_file'] from unit\n"
"           control['alive_unit'].\n"
""
//"    eval_status : int\n"
//"      evaluation status for reverse communication interface\n"
"    alloc_status : int\n"
"      the status of the last attempted internal array.\n"
"      allocation/deallocation\n"
"    bad_alloc : str\n"
"      the name of the array for which an internal array\n"
"      allocation/deallocation error ocurred.\n"
"    iter : int\n"
"      the total number of iterations performed\n"
"    f_eval : int\n"
"      the total number of evaluations of the objective function.\n"
"    g_eval : int\n"
"      the total number of evaluations of the gradient of the objective \n"
"      function.\n"
"    h_eval : int\n"
"      the total number of evaluations of the Hessian of the objective\n"
"      function.\n"
"    timings : dict\n"
"      dictionary containing timing information:\n"
"       total : float\n"
"         the total CPU time spent in the package.\n"
"       clock_total : float\n"
"         the total clock time spent in the package.\n"
"\n"
);

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

PyDoc_STRVAR(py_ugo_terminate_doc,
"ugo.terminate()\n"
"\n"
"Deallocate all internal private storage.\n"
"\n"
);

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
    {"initialize", (PyCFunction) py_ugo_initialize, METH_NOARGS,
                            py_ugo_initialize_doc},
    {"load", (PyCFunction) py_ugo_load, METH_VARARGS | METH_KEYWORDS,
                            py_ugo_load_doc},
    {"solve", (PyCFunction) py_ugo_solve, METH_VARARGS,
                            py_ugo_solve_doc},
    {"information", (PyCFunction) py_ugo_information, METH_NOARGS,
                            py_ugo_information_doc},
    {"terminate", (PyCFunction) py_ugo_terminate, METH_NOARGS,
                            py_ugo_terminate_doc},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* ugo python module documentation */
PyDoc_STRVAR(ugo_module_doc,
"The ugo package aims to find the global minimizer of a univariate\n"
"twice-continuously differentiable function :math:`f(x)` of a single\n"
"variable over the finite interval :math:`x^l <= x <= x^u`. Function\n"
"and derivative values may be provided either via a subroutine call,\n"
"or by a return to the calling program. Second derivatives may be\n"
"used to advantage if they are available.\n"
"\n"
"call order\n"
"----------\n"
"The functions should be called in the following order, with\n"
"[] indicating an optional call\n"
"\n"
"  ``ugo.initialize``\n"
"\n"
"  ``ugo.load``\n"
"\n"
"  ``ugo.solve``\n"
"\n"
"  [``ugo.information``]\n"
"\n"
"  ``ugo.terminate``\n"
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
