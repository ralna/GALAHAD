//* \file lms_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-12 AT 08:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LMS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_lms.h"

/* Module global variables */
static void *data;                       // private internal data
static struct lms_control_type control;  // control struct
static struct lms_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within TRU Python interfaces
bool lms_update_control(struct lms_control_type *control,
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
        if(strcmp(key_name, "memory_length") == 0){
            if(!parse_int_option(value, "memory_length",
                                  &control->memory_length))
                return false;
            continue;
        }
        if(strcmp(key_name, "method") == 0){
            if(!parse_int_option(value, "method",
                                  &control->method))
                return false;
            continue;
        }
        if(strcmp(key_name, "any_method") == 0){
            if(!parse_bool_option(value, "any_method",
                                  &control->any_method))
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
// NB not static as it is used for nested inform within TRU Python interface
PyObject* lms_make_options_dict(const struct lms_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "memory_length",
                         PyLong_FromLong(control->memory_length));
    PyDict_SetItemString(py_options, "method",
                         PyLong_FromLong(control->method));
    PyDict_SetItemString(py_options, "any_method",
                         PyBool_FromLong(control->any_method));
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
static PyObject* lms_make_time_dict(const struct lms_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries
    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "setup",
                         PyFloat_FromDouble(time->setup));
    PyDict_SetItemString(py_time, "form",
                         PyFloat_FromDouble(time->form));
    PyDict_SetItemString(py_time, "apply",
                         PyFloat_FromDouble(time->apply));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_setup",
                         PyFloat_FromDouble(time->clock_setup));
    PyDict_SetItemString(py_time, "clock_form",
                         PyFloat_FromDouble(time->clock_form));
    PyDict_SetItemString(py_time, "clock_apply",
                         PyFloat_FromDouble(time->clock_apply));
    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within TRU Python interfaces
PyObject* lms_make_inform_dict(const struct lms_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "length",
                         PyLong_FromLong(inform->length));
    PyDict_SetItemString(py_inform, "updates_skipped",
                         PyBool_FromLong(inform->updates_skipped));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         lms_make_time_dict(&inform->time));
    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   LMS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_lms_initialize(PyObject *self){

    // Call lms_initialize
    lms_initialize(&data, &control, &status);

    // Record that LMS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = lms_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-   LMS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_lms_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call lms_information
    lms_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = lms_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   LMS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_lms_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call lms_terminate
    lms_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE LMS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* lms python module method table */
static PyMethodDef lms_module_methods[] = {
    {"initialize", (PyCFunction) py_lms_initialize, METH_NOARGS,NULL},
    {"information", (PyCFunction) py_lms_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_lms_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* lms python module documentation */

PyDoc_STRVAR(lms_module_doc,

"Given a sequence of vectors {s_k} and {y_k} and scale factors {delta_k}, \n"
"the lms package obtains the product of a limited-memory secant \n"
"approximation H_k (or its inverse) with a given vector, \n"
"using one of a variety of well-established formulae.\n"
"\n"
"See $GALAHAD/html/Python/lms.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* lms python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "lms",               /* name of module */
   lms_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   lms_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_lms(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
