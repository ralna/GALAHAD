//* \file lhs_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-03 AT 07:50 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LHS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_lhs.h"

/* Module global variables */
static void *data;                       // private internal data
static struct lhs_control_type control;  // control struct
static struct lhs_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within BGO Python interfaces
bool lhs_update_control(struct lhs_control_type *control,
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
        if(strcmp(key_name, "duplication") == 0){
            if(!parse_int_option(value, "duplication",
                                  &control->duplication))
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
// NB not static as it is used for nested inform within BGO Python interface
PyObject* lhs_make_options_dict(const struct lhs_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "duplication",
                         PyLong_FromLong(control->duplication));
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
// NB not static as it is used for nested control within BGO Python interfaces
PyObject* lhs_make_inform_dict(const struct lhs_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   LHS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_lhs_initialize(PyObject *self){

    // Call lhs_initialize
    lhs_initialize(&data, &control, &inform);

    // Record that LHS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = lhs_make_options_dict(&control);
    PyObject *py_inform = lhs_make_inform_dict(&inform);
    return Py_BuildValue("OO", py_options, py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   LHS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_lhs_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call lhs_information
    lhs_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = lhs_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   LHS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_lhs_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call lhs_terminate
    lhs_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE LHS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* lhs python module method table */
static PyMethodDef lhs_module_methods[] = {
    {"initialize", (PyCFunction) py_lhs_initialize, METH_NOARGS,NULL},
    {"information", (PyCFunction) py_lhs_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_lhs_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* lhs python module documentation */

PyDoc_STRVAR(lhs_module_doc,

"The lhs package computes an array of Latin Hypercube samples.\n"
"\n"
"See $GALAHAD/html/Python/lhs.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* lhs python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "lhs",               /* name of module */
   lhs_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   lhs_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_lhs(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
