//* \file bsc_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-12 AT 07:50 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_BSC PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_bsc.h"

/* Module global variables */
static void *data;                       // private internal data
static struct bsc_control_type control;  // control struct
static struct bsc_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within NLS Python interface
bool bsc_update_control(struct bsc_control_type *control,
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
        if(strcmp(key_name, "max_col") == 0){
            if(!parse_int_option(value, "max_col",
                                  &control->max_col))
                return false;
            continue;
        }
        if(strcmp(key_name, "new_a") == 0){
            if(!parse_int_option(value, "new_a",
                                  &control->new_a))
                return false;
            continue;
        }
        if(strcmp(key_name, "extra_space_s") == 0){
            if(!parse_int_option(value, "extra_space_s",
                                  &control->extra_space_s))
                return false;
            continue;
        }
        if(strcmp(key_name, "s_also_by_column") == 0){
            if(!parse_bool_option(value, "s_also_by_column",
                                  &control->s_also_by_column))
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
// NB not static as it is used for nested inform within NLS Python interface
PyObject* bsc_make_options_dict(const struct bsc_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "max_col",
                         PyLong_FromLong(control->max_col));
    PyDict_SetItemString(py_options, "new_a",
                         PyLong_FromLong(control->new_a));
    PyDict_SetItemString(py_options, "extra_space_s",
                         PyLong_FromLong(control->extra_space_s));
    PyDict_SetItemString(py_options, "s_also_by_column",
                         PyBool_FromLong(control->s_also_by_column));
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
// NB not static as it is used for nested control within NLS Python interface
PyObject* bsc_make_inform_dict(const struct bsc_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "max_col_a",
                         PyLong_FromLong(inform->max_col_a));
    PyDict_SetItemString(py_inform, "exceeds_max_col",
                         PyLong_FromLong(inform->exceeds_max_col));
    PyDict_SetItemString(py_inform, "time",
                         PyFloat_FromDouble(inform->time));
    PyDict_SetItemString(py_inform, "clock_time",
                         PyFloat_FromDouble(inform->clock_time));
    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   BSC_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_bsc_initialize(PyObject *self){

    // Call bsc_initialize
    bsc_initialize(&data, &control, &status);

    // Record that BSC has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = bsc_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-   BSC_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_bsc_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call bsc_information
    bsc_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = bsc_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   BSC_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_bsc_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call bsc_terminate
    bsc_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE BSC PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* bsc python module method table */
static PyMethodDef bsc_module_methods[] = {
    {"initialize", (PyCFunction) py_bsc_initialize, METH_NOARGS,NULL},
    {"information", (PyCFunction) py_bsc_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_bsc_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* bsc python module documentation */

PyDoc_STRVAR(bsc_module_doc,

"The bsc package takes given matrices A and (diagonal) D, and \n"
"builds the Schur complement S = A D A^T in sparse co-ordinate \n"
"(and optionally sparse column) format(s). Full advantage is taken \n"
"of any zero coefficients in the matrix A. \n"
"\n"
"See $GALAHAD/html/Python/bsc.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* bsc python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "bsc",               /* name of module */
   bsc_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   bsc_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_bsc(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
