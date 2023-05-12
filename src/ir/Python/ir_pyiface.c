//* \file ir_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-11 AT 08:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_IR PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_ir.h"

/* Module global variables */
static void *data;                       // private internal data
static struct ir_control_type control;  // control struct
static struct ir_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within CRO Python interface
bool ir_update_control(struct ir_control_type *control,
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
        if(strcmp(key_name, "itref_max") == 0){
            if(!parse_int_option(value, "itref_max",
                                  &control->itref_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "acceptable_residual_relative") == 0){
            if(!parse_double_option(value, "acceptable_residual_relative",
                                  &control->acceptable_residual_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "acceptable_residual_absolute") == 0){
            if(!parse_double_option(value, "acceptable_residual_absolute",
                                  &control->acceptable_residual_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "required_residual_relative") == 0){
            if(!parse_double_option(value, "required_residual_relative",
                                  &control->required_residual_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "record_residuals") == 0){
            if(!parse_bool_option(value, "record_residuals",
                                  &control->record_residuals))
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
// NB not static as it is used for nested inform within CRO Python interface
PyObject* ir_make_options_dict(const struct ir_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "itref_max",
                         PyLong_FromLong(control->itref_max));
    PyDict_SetItemString(py_options, "acceptable_residual_relative",
                         PyFloat_FromDouble(control->acceptable_residual_relative));
    PyDict_SetItemString(py_options, "acceptable_residual_absolute",
                         PyFloat_FromDouble(control->acceptable_residual_absolute));
    PyDict_SetItemString(py_options, "required_residual_relative",
                         PyFloat_FromDouble(control->required_residual_relative));
    PyDict_SetItemString(py_options, "record_residuals",
                         PyBool_FromLong(control->record_residuals));
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
// NB not static as it is used for nested control within CRO Python interface
PyObject* ir_make_inform_dict(const struct ir_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "norm_initial_residual",
                         PyFloat_FromDouble(inform->norm_initial_residual));
    PyDict_SetItemString(py_inform, "norm_final_residual",
                         PyFloat_FromDouble(inform->norm_final_residual));
    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   IR_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_ir_initialize(PyObject *self){

    // Call ir_initialize
    ir_initialize(&data, &control, &status);

    // Record that IR has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = ir_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-   IR_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_ir_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call ir_information
    ir_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = ir_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   IR_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_ir_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call ir_terminate
    ir_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE IR PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* ir python module method table */
static PyMethodDef ir_module_methods[] = {
    {"initialize", (PyCFunction) py_ir_initialize, METH_NOARGS,NULL},
    {"information", (PyCFunction) py_ir_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_ir_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* ir python module documentation */

PyDoc_STRVAR(ir_module_doc,

"Given a sparse symmetric n \times n matrix A = a_{ij} and the \n"
"factorization of A found by the GALAHAD package sls, the ir package \n"
"solves the system of linear equations A x = b using \n"
"iterative refinement.\n"
"\n"
"See $GALAHAD/html/Python/ir.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* ir python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "ir",               /* name of module */
   ir_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   ir_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_ir(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
