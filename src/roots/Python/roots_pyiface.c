//* \file roots_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-12 AT 08:40 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_ROOTS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. May 3rd 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_roots.h"

/* Module global variables */
static void *data;                       // private internal data
static struct roots_control_type control;  // control struct
static struct roots_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within QP Python interfaces
bool roots_update_control(struct roots_control_type *control,
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
        if(strcmp(key_name, "tol") == 0){
            if(!parse_double_option(value, "tol",
                                  &control->tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "zero_coef") == 0){
            if(!parse_double_option(value, "zero_coef",
                                  &control->zero_coef))
                return false;
            continue;
        }
        if(strcmp(key_name, "zero_f") == 0){
            if(!parse_double_option(value, "zero_f",
                                  &control->zero_f))
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
PyObject* roots_make_options_dict(const struct roots_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "tol",
                         PyFloat_FromDouble(control->tol));
    PyDict_SetItemString(py_options, "zero_coef",
                         PyFloat_FromDouble(control->zero_coef));
    PyDict_SetItemString(py_options, "zero_f",
                         PyFloat_FromDouble(control->zero_f));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within QP Python interfaces
PyObject* roots_make_inform_dict(const struct roots_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   ROOTS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_roots_initialize(PyObject *self){

    // Call roots_initialize
    roots_initialize(&data, &control, &status);

    // Record that ROOTS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = roots_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-   ROOTS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_roots_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call roots_information
    roots_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = roots_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   ROOTS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_roots_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call roots_terminate
    roots_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE ROOTS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* roots python module method table */
static PyMethodDef roots_module_methods[] = {
    {"initialize", (PyCFunction) py_roots_initialize, METH_NOARGS,NULL},
    {"information", (PyCFunction) py_roots_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_roots_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* roots python module documentation */

PyDoc_STRVAR(roots_module_doc,

"The roots package uses classical formulae together with Newton’s method \n"
"to find all the real roots of a real polynomial.\n"
"\n"
"See $GALAHAD/html/Python/roots.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* roots python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "roots",               /* name of module */
   roots_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   roots_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_roots(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
