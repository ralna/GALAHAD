//* \file sec_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-12 AT 08:50 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SEC PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_sec.h"

/* Module global variables */
static void *data;                       // private internal data
static struct sec_control_type control;  // control struct
static struct sec_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within TRU Python interface
bool sec_update_control(struct sec_control_type *control,
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
        if(strcmp(key_name, "h_initial") == 0){
            if(!parse_double_option(value, "h_initial",
                                  &control->h_initial))
                return false;
            continue;
        }
        if(strcmp(key_name, "update_skip_tol") == 0){
            if(!parse_double_option(value, "update_skip_tol",
                                  &control->update_skip_tol))
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
// NB not static as it is used for nested inform within TRU Python interface
PyObject* sec_make_options_dict(const struct sec_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "h_initial",
                         PyFloat_FromDouble(control->h_initial));
    PyDict_SetItemString(py_options, "update_skip_tol",
                         PyFloat_FromDouble(control->update_skip_tol));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within TRU Python interface
PyObject* sec_make_inform_dict(const struct sec_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   SEC_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_sec_initialize(PyObject *self){

    // Call sec_initialize
    sec_initialize(&control, &status);

    // Record that SEC has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = sec_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-   SEC_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_sec_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call sec_information
    sec_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = sec_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   SEC_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_sec_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call sec_terminate
    sec_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE SEC PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* sec python module method table */
static PyMethodDef sec_module_methods[] = {
    {"initialize", (PyCFunction) py_sec_initialize, METH_NOARGS,NULL},
    {"information", (PyCFunction) py_sec_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_sec_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* sec python module documentation */

PyDoc_STRVAR(sec_module_doc,

"The sec package builds and updates dense BFGS and SR1 secant \n"
"approximations to a Hessian so that the approximation B satisfies \n"
"the secant condition B s = y for given vectors s and y.\n"
"\n"
"See $GALAHAD/html/Python/sec.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* sec python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "sec",               /* name of module */
   sec_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   sec_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_sec(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
