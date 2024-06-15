//* \file lsrt_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-20 AT 10:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LSRT PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 12th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_lsrt.h"

/* Module global variables */
static void *data;                       // private internal data
static struct lsrt_control_type control; // control struct
static struct lsrt_inform_type inform;   // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool lsrt_update_control(struct lsrt_control_type *control,
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
        if(strcmp(key_name, "itmin") == 0){
            if(!parse_int_option(value, "itmin",
                                  &control->itmin))
                return false;
            continue;
        }
        if(strcmp(key_name, "itmax") == 0){
            if(!parse_int_option(value, "itmax",
                                  &control->itmax))
                return false;
            continue;
        }
        if(strcmp(key_name, "bitmax") == 0){
            if(!parse_int_option(value, "bitmax",
                                  &control->bitmax))
                return false;
            continue;
        }
        if(strcmp(key_name, "extra_vectors") == 0){
            if(!parse_int_option(value, "extra_vectors",
                                  &control->extra_vectors))
                return false;
            continue;
        }
        if(strcmp(key_name, "stopping_rule") == 0){
            if(!parse_int_option(value, "stopping_rule",
                                  &control->stopping_rule))
                return false;
            continue;
        }
        if(strcmp(key_name, "freq") == 0){
            if(!parse_int_option(value, "freq",
                                  &control->freq))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_relative") == 0){
            if(!parse_double_option(value, "stop_relative",
                                  &control->stop_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_absolute") == 0){
            if(!parse_double_option(value, "stop_absolute",
                                  &control->stop_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "fraction_opt") == 0){
            if(!parse_double_option(value, "fraction_opt",
                                  &control->fraction_opt))
                return false;
            continue;
        }
        if(strcmp(key_name, "time_limit") == 0){
            if(!parse_double_option(value, "time_limit",
                                  &control->time_limit))
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
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
// NB not static as it is used for nested inform within QP Python interface
PyObject* lsrt_make_options_dict(const struct lsrt_control_type *control){
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
    PyDict_SetItemString(py_options, "itmin",
                         PyLong_FromLong(control->itmin));
    PyDict_SetItemString(py_options, "itmax",
                         PyLong_FromLong(control->itmax));
    PyDict_SetItemString(py_options, "bitmax",
                         PyLong_FromLong(control->bitmax));
    PyDict_SetItemString(py_options, "extra_vectors",
                         PyLong_FromLong(control->extra_vectors));
    PyDict_SetItemString(py_options, "stopping_rule",
                         PyLong_FromLong(control->stopping_rule));
    PyDict_SetItemString(py_options, "freq",
                         PyLong_FromLong(control->freq));
    PyDict_SetItemString(py_options, "stop_relative",
                         PyFloat_FromDouble(control->stop_relative));
    PyDict_SetItemString(py_options, "stop_absolute",
                         PyFloat_FromDouble(control->stop_absolute));
    PyDict_SetItemString(py_options, "fraction_opt",
                         PyFloat_FromDouble(control->fraction_opt));
    PyDict_SetItemString(py_options, "time_limit",
                         PyFloat_FromDouble(control->time_limit));
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
//static PyObject* lsrt_make_time_dict(const struct lsrt_time_type *time){
//    PyObject *py_time = PyDict_New();
//    return py_time;
//}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* lsrt_make_inform_dict(const struct lsrt_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "iter",
                         PyLong_FromLong(inform->iter));
    PyDict_SetItemString(py_inform, "iter_pass2",
                         PyLong_FromLong(inform->iter_pass2));
    PyDict_SetItemString(py_inform, "biters",
                         PyLong_FromLong(inform->biters));
    PyDict_SetItemString(py_inform, "biter_min",
                         PyLong_FromLong(inform->biter_min));
    PyDict_SetItemString(py_inform, "biter_max",
                         PyLong_FromLong(inform->biter_max));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "multiplier",
                         PyFloat_FromDouble(inform->multiplier));
    PyDict_SetItemString(py_inform, "x_norm",
                         PyFloat_FromDouble(inform->x_norm));
    PyDict_SetItemString(py_inform, "r_norm",
                         PyFloat_FromDouble(inform->r_norm));
    PyDict_SetItemString(py_inform, "Atr_norm",
                         PyFloat_FromDouble(inform->Atr_norm));
    PyDict_SetItemString(py_inform, "biter_mean",
                         PyFloat_FromDouble(inform->biter_mean));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   LSRT_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_lsrt_initialize(PyObject *self){

    // Call lsrt_initialize
    lsrt_initialize(&data, &control, &status);

    // Record that LSRT has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = lsrt_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   LSRT_LOAD_OPTIONS    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_lsrt_load_options(PyObject *self, PyObject *args, PyObject *keywds){
    PyObject *py_options = NULL;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &py_options))
        return NULL;

    // Update LSRT control options
    if(!lsrt_update_control(&control, py_options))
        return NULL;

    // Call cqp_import
    lsrt_import_control(&control, &data, &status);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   LSRT_SOLVE_PROBLEM   -*-*-*-*-*-*-*-*

static PyObject* py_lsrt_solve_problem(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_u, *py_v;
    double *u, *v;
    int status, m, n;
    double power, weight;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"status", "m", "n", "power", "weight", "u", "v", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiiddOO", kwlist, &status, &m, &n, &power, &weight,
                                    &py_u, &py_v))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("u", py_u, m))
        return NULL;
    if(!check_array_double("v", py_v, n))
        return NULL;

    // Get array data pointer
    u = (double *) PyArray_DATA(py_u);
    v = (double *) PyArray_DATA(py_v);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x
    PyArrayObject *py_x =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *x = (double *) PyArray_DATA(py_x);

    // Call lsrt_solve_direct
    lsrt_solve_problem(&data, &status, m, n, power, weight, x, u, v);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return status, x, u and v
    PyObject *solve_problem_return;

    // solve_problem_return = Py_BuildValue("O", py_x);
    solve_problem_return = Py_BuildValue("iOOO", status, py_x, py_u, py_v);
    Py_INCREF(solve_problem_return);
    return solve_problem_return;
}

//  *-*-*-*-*-*-*-*-*-*-   LSRT_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_lsrt_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call lsrt_information
    lsrt_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = lsrt_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   LSRT_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_lsrt_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call lsrt_terminate
    lsrt_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE LSRT PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* lsrt python module method table */
static PyMethodDef lsrt_module_methods[] = {
    {"initialize", (PyCFunction) py_lsrt_initialize, METH_NOARGS, NULL},
    {"load_options", (PyCFunction) py_lsrt_load_options,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_problem", (PyCFunction) py_lsrt_solve_problem, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_lsrt_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_lsrt_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* lsrt python module documentation */

PyDoc_STRVAR(lsrt_module_doc,

"The lsrt package uses a Krylov-subspace iteration to find an \n"
"approximation of the global minimizer of a regularized linear \n"
"sum-of-squares objective function. \n"
"The aim is to minimize the regularized quadratic objective function \n"
"r(x) = 1/2 || A x - b ||_2^2 + sigma / p ||x||_2^p,  \n"
"where the weight sigma >= 0, the power p >= 2, and \n"
"the l_2-norm of x is defined to be ||x||_2 = sqrt{x^T x}. \n"
"The method may be suitable for large problems as no factorization of A is \n"
"required. Reverse communication is used to obtain \n"
"matrix-vector products of the form u + A v and v + A^T u. \n"
"\n"
"See $GALAHAD/html/Python/lsrt.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* lsrt python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "lsrt",               /* name of module */
   lsrt_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   lsrt_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_lsrt(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
