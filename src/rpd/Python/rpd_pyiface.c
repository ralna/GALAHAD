//* \file rpd_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-12 AT 08:40 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_RPD PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_rpd.h"

/* Module global variables */
static void *data;                       // private internal data
static struct rpd_control_type control;  // control struct
static struct rpd_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within QP Python interfaces
bool rpd_update_control(struct rpd_control_type *control,
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
        if(strcmp(key_name, "qplib") == 0){
            if(!parse_int_option(value, "qplib",
                                  &control->qplib))
                return false;
            continue;
        }
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
PyObject* rpd_make_options_dict(const struct rpd_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "qplib",
                         PyLong_FromLong(control->qplib));
    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within QP Python interfaces
PyObject* rpd_make_inform_dict(const struct rpd_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "io_status",
                         PyLong_FromLong(inform->io_status));
    PyDict_SetItemString(py_inform, "line",
                         PyLong_FromLong(inform->line));
    PyDict_SetItemString(py_inform, "p_type",
                         PyUnicode_FromString(inform->p_type));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   RPD_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_rpd_initialize(PyObject *self){

    // Call rpd_initialize
    rpd_initialize(&data, &control, &status);

    // Record that RPD has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = rpd_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-   RPD_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_rpd_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call rpd_information
    rpd_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = rpd_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   RPD_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_rpd_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call rpd_terminate
    rpd_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE RPD PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* rpd python module method table */
static PyMethodDef rpd_module_methods[] = {
    {"initialize", (PyCFunction) py_rpd_initialize, METH_NOARGS,NULL},
    {"information", (PyCFunction) py_rpd_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_rpd_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* rpd python module documentation */

PyDoc_STRVAR(rpd_module_doc,

"The rpd package reads and writes quadratic programming \n"
"(and related) problem data to and from a QPLIB-format data file. \n"
"Variables may be continuous, binary or integer.\n"
"\n"
"See $GALAHAD/html/Python/rpd.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* rpd python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "rpd",               /* name of module */
   rpd_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   rpd_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_rpd(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
