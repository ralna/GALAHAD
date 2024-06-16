//* \file glrt_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-20 AT 10:20 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_GLRT PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 11th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_glrt.h"

/* Module global variables */
static void *data;                       // private internal data
static struct glrt_control_type control;  // control struct
static struct glrt_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within QP Python interfaces
bool glrt_update_control(struct glrt_control_type *control,
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
        if(strcmp(key_name, "itmax") == 0){
            if(!parse_int_option(value, "itmax",
                                  &control->itmax))
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
        if(strcmp(key_name, "extra_vectors") == 0){
            if(!parse_int_option(value, "extra_vectors",
                                  &control->extra_vectors))
                return false;
            continue;
        }
        if(strcmp(key_name, "ritz_printout_device") == 0){
            if(!parse_int_option(value, "ritz_printout_device",
                                  &control->ritz_printout_device))
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
        if(strcmp(key_name, "rminvr_zero") == 0){
            if(!parse_double_option(value, "rminvr_zero",
                                  &control->rminvr_zero))
                return false;
            continue;
        }
        if(strcmp(key_name, "f_0") == 0){
            if(!parse_double_option(value, "f_0",
                                  &control->f_0))
                return false;
            continue;
        }
        if(strcmp(key_name, "unitm") == 0){
            if(!parse_bool_option(value, "unitm",
                                  &control->unitm))
                return false;
            continue;
        }
        if(strcmp(key_name, "impose_descent") == 0){
            if(!parse_bool_option(value, "impose_descent",
                                  &control->impose_descent))
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
        if(strcmp(key_name, "print_ritz_values") == 0){
            if(!parse_bool_option(value, "print_ritz_values",
                                  &control->print_ritz_values))
                return false;
            continue;
        }
        if(strcmp(key_name, "ritz_file_name") == 0){
            if(!parse_char_option(value, "ritz_file_name",
                                  control->ritz_file_name,
                                  sizeof(control->ritz_file_name)))
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
        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
          "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
// NB not static as it is used for nested inform within QP Python interfaces
PyObject* glrt_make_options_dict(const struct glrt_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "itmax",
                         PyLong_FromLong(control->itmax));
    PyDict_SetItemString(py_options, "stopping_rule",
                         PyLong_FromLong(control->stopping_rule));
    PyDict_SetItemString(py_options, "freq",
                         PyLong_FromLong(control->freq));
    PyDict_SetItemString(py_options, "extra_vectors",
                         PyLong_FromLong(control->extra_vectors));
    PyDict_SetItemString(py_options, "ritz_printout_device",
                         PyLong_FromLong(control->ritz_printout_device));
    PyDict_SetItemString(py_options, "stop_relative",
                         PyFloat_FromDouble(control->stop_relative));
    PyDict_SetItemString(py_options, "stop_absolute",
                         PyFloat_FromDouble(control->stop_absolute));
    PyDict_SetItemString(py_options, "fraction_opt",
                         PyFloat_FromDouble(control->fraction_opt));
    PyDict_SetItemString(py_options, "rminvr_zero",
                         PyFloat_FromDouble(control->rminvr_zero));
    PyDict_SetItemString(py_options, "f_0",
                         PyFloat_FromDouble(control->f_0));
    PyDict_SetItemString(py_options, "unitm",
                         PyBool_FromLong(control->unitm));
    PyDict_SetItemString(py_options, "impose_descent",
                         PyBool_FromLong(control->impose_descent));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "print_ritz_values",
                         PyBool_FromLong(control->print_ritz_values));
    PyDict_SetItemString(py_options, "ritz_file_name",
                         PyUnicode_FromString(control->ritz_file_name));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
//static PyObject* glrt_make_time_dict(const struct glrt_time_type *time){
//    PyObject *py_time = PyDict_New();
//    return py_time;
//}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within QP Python interfaces
PyObject* glrt_make_inform_dict(const struct glrt_inform_type *inform){
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
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "obj_regularized",
                         PyFloat_FromDouble(inform->obj_regularized));
    PyDict_SetItemString(py_inform, "multiplier",
                         PyFloat_FromDouble(inform->multiplier));
    PyDict_SetItemString(py_inform, "xpo_norm",
                         PyFloat_FromDouble(inform->xpo_norm));
    PyDict_SetItemString(py_inform, "leftmost",
                         PyFloat_FromDouble(inform->leftmost));
    PyDict_SetItemString(py_inform, "negative_curvature",
                         PyBool_FromLong(inform->negative_curvature));
    PyDict_SetItemString(py_inform, "hard_case",
                         PyBool_FromLong(inform->hard_case));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   GLRT_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_glrt_initialize(PyObject *self){

    // Call glrt_initialize
    glrt_initialize(&data, &control, &status);

    // Record that GLRT has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = glrt_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   GLRT_LOAD_OPTIONS    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_glrt_load_options(PyObject *self, PyObject *args,
                                      PyObject *keywds){
    PyObject *py_options = NULL;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &py_options))
        return NULL;

    // Update GLRT control options
    if(!glrt_update_control(&control, py_options))
        return NULL;

    // Call cqp_import
    glrt_import_control(&control, &data, &status);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   GLRT_SOLVE_PROBLEM   -*-*-*-*-*-*-*-*

static PyObject* py_glrt_solve_problem(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_r, *py_v;
    double *r, *v;
    int status, n;
    double power, weight;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"status", "n", "power", "weight", "r", "v", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiddOO", kwlist, &status, 
                                    &n, &power, &weight, &py_r, &py_v))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("r", py_r, n))
        return NULL;
    if(!check_array_double("v", py_v, n))
        return NULL;

    // Get array data pointer
    r = (double *) PyArray_DATA(py_r);
    v = (double *) PyArray_DATA(py_v);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x
    PyArrayObject *py_x =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *x = (double *) PyArray_DATA(py_x);

    // Call glrt_solve_direct
    glrt_solve_problem(&data, &status, n, power, weight, x, r, v);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);
    // for( int i = 0; i < n; i++) printf("r %f\n", r[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return status, x, r and v
    PyObject *solve_problem_return;

    // solve_problem_return = Py_BuildValue("O", py_x);
    solve_problem_return = Py_BuildValue("iOOO", status, py_x, py_r, py_v);
    Py_INCREF(solve_problem_return);
    return solve_problem_return;
}

//  *-*-*-*-*-*-*-*-*-*-   GLRT_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_glrt_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call glrt_information
    glrt_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = glrt_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   GLRT_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_glrt_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call glrt_terminate
    glrt_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE GLRT PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* glrt python module method table */
static PyMethodDef glrt_module_methods[] = {
    {"initialize", (PyCFunction) py_glrt_initialize, METH_NOARGS, NULL},
    {"load_options", (PyCFunction) py_glrt_load_options,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_problem", (PyCFunction) py_glrt_solve_problem, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_glrt_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_glrt_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* glrt python module documentation */

PyDoc_STRVAR(glrt_module_doc,

"The glrt package uses a Krylov-subspace iteration to find an \n"
"approximation of the global minimizer of a regularized quadratic \n"
"objective function. \n"
"The aim is to minimize the regularized quadratic objective function \n"
"r(x) = f + g^T x + 1/2 x^T H x + sigma / p ||x||_M^p ,  \n"
"where the weight sigma >= 0, the power p >= 2, and \n"
"the M-norm of x is defined to be ||x||_M = sqrt{x^T M x}. \n"
"The method may be suitable for large problems as no factorization of H is \n"
"required. Reverse communication is used to obtain \n"
"matrix-vector products of the form H z and M^{-1} z. \n"
"\n"
"See $GALAHAD/html/Python/glrt.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* glrt python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "glrt",               /* name of module */
   glrt_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   glrt_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_glrt(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
