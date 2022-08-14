//* \file ugo_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-03-13 AT 11:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_UGO PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. July 27th 2021
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
static int eval_fgh(double x, double *f, double *g, double *h, const void *userdata){

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
        PyErr_SetString(PyExc_TypeError, "unable to parse eval_fgh return values");
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python */
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
        if(strcmp(key_name, "print_level") == 0){
            if(!parse_int_option(value, "print_level", &control->print_level))
                return false;
            continue;
        }
        if(strcmp(key_name, "maxit") == 0){
            if(!parse_int_option(value, "maxit", &control->maxit))
                return false;
            continue;
        }

        // ... other int options ...

        // Parse each float/double option
        if(strcmp(key_name, "stop_length") == 0){
            if(!parse_double_option(value, "stop_length", &control->stop_length))
                return false;
            continue;
        }
        // ... other float/double options ...

        // Parse each bool option
        if(strcmp(key_name, "space_critical") == 0){
            if(!parse_bool_option(value, "space_critical", &control->space_critical))
                return false;
            continue;
        }
        // ... other bool options ...

        // Parse each char option
        if(strcmp(key_name, "prefix") == 0){
            if(!parse_char_option(value, "prefix", control->prefix))
                return false;
            continue;
        }
        // ... other char options ...

        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError, "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* ugo_make_time_dict(const struct ugo_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries
    PyDict_SetItemString(py_time, "total", PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "clock_total", PyFloat_FromDouble(time->clock_total));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested inform within BGO Python interface
PyObject* ugo_make_inform_dict(const struct ugo_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    // Set int inform entries
    PyDict_SetItemString(py_inform, "iter", PyLong_FromLong(inform->iter));
    // ... other int inform entries ...

    // Set float/double inform entries
    //PyDict_SetItemString(py_inform, "obj", PyFloat_FromDouble(inform->obj));
    // ... other float/double inform entries ...

    // Set bool inform entries
    //PyDict_SetItemString(py_inform, "used_grad", PyBool_FromLong(inform->used_grad));
    // ... other bool inform entries ...

    // Set char inform entries
    //PyDict_SetItemString(py_inform, "name", PyUnicode_FromString(inform->name));
    // ... other char inform entries ...

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time", ugo_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   UGO_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_ugo_initialize_doc,
"ugo.initialize()\n"
"\n"
"Set default option values and initialize private data\n"
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
"    holds the value :math:`x^l` of the lower bound on the optimization variable :math:`x`.\n"
"x_u : double\n"
"    holds the value :math:`x^u` of the upper bound on the optimization variable :math:`x`.\n"
"options : dict, optional\n"
"    dictionary of control options\n"
"\n"
);

static PyObject* py_ugo_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyObject *py_options = NULL;
    double x_l, x_u;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"x_l","x_u","options", NULL}; /* Note sentinel at end */
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "dd|O", kwlist, &x_l, &x_u, &py_options))
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
"    The value of the objective function :math:`f(x)` and its first derivative\n"
"    :math:`f'(x)` evaluated at :math:`x` must be assigned to ``f`` and ``g`` respectively.\n"
"    In addition, if options['second_derivatives_available'] has been set to\n"
"    True when calling ``ugo.load``, the user must also assign the value of the\n"
"    second derivative :math:`f''(x)` to ``h``; it need not be assigned otherwise.\n"
"\n"
"Returns\n"
"-------\n"
"x : double\n"
"    holds the value of the approximate global minimizer :math:`x` after a\n"
"    successful call.\n"
"f : double\n"
"    holds the value of the objective function :math:`f(x)` at the approximate\n"
"    global minimizer :math:`x` after a successful call.\n"
"g : double\n"
"    holds the value of the gradient of the objective function :math:`f'(x)`\n"
"    at the approximate global minimizer :math:`x` after a successful call.\n"
"h : double\n"
"    holds the value of the second derivative of the objective function\n"
"    :math:`f''(x)` at the approximate global minimizer :math:`x` after a successful call.\n"
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
"Provide output information\n"
"\n"
"Returns\n"
"-------\n"
"inform : dict\n"
"    dictionary containing output information\n"
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
"Deallocate all internal private storage\n"
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
    {"initialize", (PyCFunction) py_ugo_initialize, METH_NOARGS, py_ugo_initialize_doc},
    {"load", (PyCFunction) py_ugo_load, METH_VARARGS | METH_KEYWORDS, py_ugo_load_doc},
    {"solve", (PyCFunction) py_ugo_solve, METH_VARARGS, py_ugo_solve_doc},
    {"information", (PyCFunction) py_ugo_information, METH_NOARGS, py_ugo_information_doc},
    {"terminate", (PyCFunction) py_ugo_terminate, METH_NOARGS, py_ugo_terminate_doc},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* ugo python module documentation */
PyDoc_STRVAR(ugo_module_doc,
"The ugo package aims to find the global minimizer of a univariate\n"
" twice-continuously differentiable function :math:`f(x)` of a single variable\n"
" over the finite interval :math:`x^l <= x <= x^u`. Function and\n"
" derivative values may be provided either via a subroutine call,\n"
" or by a return to the calling program. Second derivatives may be used\n"
" to advantage if they are available.\n"
);

/* ugo python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "ugo",               /* name of module */
   ugo_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,
                           or -1 if the module keeps state in global variables */
   ugo_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_ugo(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
