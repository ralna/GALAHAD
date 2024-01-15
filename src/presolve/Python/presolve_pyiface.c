//* \file presolve_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-12 AT 08:50 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_PRESOLVE PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_presolve.h"

/* Module global variables */
static void *data;                       // private internal data
static struct presolve_control_type control;  // control struct
static struct presolve_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within QP Python interfaces
bool presolve_update_control(struct presolve_control_type *control,
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
        if(strcmp(key_name, "termination") == 0){
            if(!parse_int_option(value, "termination",
                                  &control->termination))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_nbr_transforms") == 0){
            if(!parse_int_option(value, "max_nbr_transforms",
                                  &control->max_nbr_transforms))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_nbr_passes") == 0){
            if(!parse_int_option(value, "max_nbr_passes",
                                  &control->max_nbr_passes))
                return false;
            continue;
        }
        if(strcmp(key_name, "c_accuracy") == 0){
            if(!parse_double_option(value, "c_accuracy",
                                  &control->c_accuracy))
                return false;
            continue;
        }
        if(strcmp(key_name, "z_accuracy") == 0){
            if(!parse_double_option(value, "z_accuracy",
                                  &control->z_accuracy))
                return false;
            continue;
        }
        if(strcmp(key_name, "infinity") == 0){
            if(!parse_double_option(value, "infinity",
                                  &control->infinity))
                return false;
            continue;
        }
        if(strcmp(key_name, "out") == 0){
            if(!parse_int_option(value, "out",
                                  &control->out))
                return false;
            continue;
        }
        if(strcmp(key_name, "errout") == 0){
            if(!parse_int_option(value, "errout",
                                  &control->errout))
                return false;
            continue;
        }
        if(strcmp(key_name, "print_level") == 0){
            if(!parse_int_option(value, "print_level",
                                  &control->print_level))
                return false;
            continue;
        }
        if(strcmp(key_name, "dual_transformations") == 0){
            if(!parse_bool_option(value, "dual_transformations",
                                  &control->dual_transformations))
                return false;
            continue;
        }
        if(strcmp(key_name, "redundant_xc") == 0){
            if(!parse_bool_option(value, "redundant_xc",
                                  &control->redundant_xc))
                return false;
            continue;
        }
        if(strcmp(key_name, "primal_constraints_freq") == 0){
            if(!parse_int_option(value, "primal_constraints_freq",
                                  &control->primal_constraints_freq))
                return false;
            continue;
        }
        if(strcmp(key_name, "dual_constraints_freq") == 0){
            if(!parse_int_option(value, "dual_constraints_freq",
                                  &control->dual_constraints_freq))
                return false;
            continue;
        }
        if(strcmp(key_name, "singleton_columns_freq") == 0){
            if(!parse_int_option(value, "singleton_columns_freq",
                                  &control->singleton_columns_freq))
                return false;
            continue;
        }
        if(strcmp(key_name, "doubleton_columns_freq") == 0){
            if(!parse_int_option(value, "doubleton_columns_freq",
                                  &control->doubleton_columns_freq))
                return false;
            continue;
        }
        if(strcmp(key_name, "unc_variables_freq") == 0){
            if(!parse_int_option(value, "unc_variables_freq",
                                  &control->unc_variables_freq))
                return false;
            continue;
        }
        if(strcmp(key_name, "dependent_variables_freq") == 0){
            if(!parse_int_option(value, "dependent_variables_freq",
                                  &control->dependent_variables_freq))
                return false;
            continue;
        }
        if(strcmp(key_name, "sparsify_rows_freq") == 0){
            if(!parse_int_option(value, "sparsify_rows_freq",
                                  &control->sparsify_rows_freq))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_fill") == 0){
            if(!parse_int_option(value, "max_fill",
                                  &control->max_fill))
                return false;
            continue;
        }
        if(strcmp(key_name, "transf_file_nbr") == 0){
            if(!parse_int_option(value, "transf_file_nbr",
                                  &control->transf_file_nbr))
                return false;
            continue;
        }
        if(strcmp(key_name, "transf_buffer_size") == 0){
            if(!parse_int_option(value, "transf_buffer_size",
                                  &control->transf_buffer_size))
                return false;
            continue;
        }
        if(strcmp(key_name, "transf_file_status") == 0){
            if(!parse_int_option(value, "transf_file_status",
                                  &control->transf_file_status))
                return false;
            continue;
        }
        if(strcmp(key_name, "transf_file_name") == 0){
            if(!parse_char_option(value, "transf_file_name",
                                  control->transf_file_name,
                                  sizeof(control->transf_file_name)))
                return false;
            continue;
        }
        if(strcmp(key_name, "y_sign") == 0){
            if(!parse_int_option(value, "y_sign",
                                  &control->y_sign))
                return false;
            continue;
        }
        if(strcmp(key_name, "inactive_y") == 0){
            if(!parse_int_option(value, "inactive_y",
                                  &control->inactive_y))
                return false;
            continue;
        }
        if(strcmp(key_name, "z_sign") == 0){
            if(!parse_int_option(value, "z_sign",
                                  &control->z_sign))
                return false;
            continue;
        }
        if(strcmp(key_name, "inactive_z") == 0){
            if(!parse_int_option(value, "inactive_z",
                                  &control->inactive_z))
                return false;
            continue;
        }
        if(strcmp(key_name, "final_x_bounds") == 0){
            if(!parse_int_option(value, "final_x_bounds",
                                  &control->final_x_bounds))
                return false;
            continue;
        }
        if(strcmp(key_name, "final_z_bounds") == 0){
            if(!parse_int_option(value, "final_z_bounds",
                                  &control->final_z_bounds))
                return false;
            continue;
        }
        if(strcmp(key_name, "final_c_bounds") == 0){
            if(!parse_int_option(value, "final_c_bounds",
                                  &control->final_c_bounds))
                return false;
            continue;
        }
        if(strcmp(key_name, "final_y_bounds") == 0){
            if(!parse_int_option(value, "final_y_bounds",
                                  &control->final_y_bounds))
                return false;
            continue;
        }
        if(strcmp(key_name, "check_primal_feasibility") == 0){
            if(!parse_int_option(value, "check_primal_feasibility",
                                  &control->check_primal_feasibility))
                return false;
            continue;
        }
        if(strcmp(key_name, "check_dual_feasibility") == 0){
            if(!parse_int_option(value, "check_dual_feasibility",
                                  &control->check_dual_feasibility))
                return false;
            continue;
        }
        if(strcmp(key_name, "pivot_tol") == 0){
            if(!parse_double_option(value, "pivot_tol",
                                  &control->pivot_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "min_rel_improve") == 0){
            if(!parse_double_option(value, "min_rel_improve",
                                  &control->min_rel_improve))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_growth_factor") == 0){
            if(!parse_double_option(value, "max_growth_factor",
                                  &control->max_growth_factor))
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
// NB not static as it is used for nested inform within QP Python interface
PyObject* presolve_make_options_dict(const struct presolve_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "termination",
                         PyLong_FromLong(control->termination));
    PyDict_SetItemString(py_options, "max_nbr_transforms",
                         PyLong_FromLong(control->max_nbr_transforms));
    PyDict_SetItemString(py_options, "max_nbr_passes",
                         PyLong_FromLong(control->max_nbr_passes));
    PyDict_SetItemString(py_options, "c_accuracy",
                         PyFloat_FromDouble(control->c_accuracy));
    PyDict_SetItemString(py_options, "z_accuracy",
                         PyFloat_FromDouble(control->z_accuracy));
    PyDict_SetItemString(py_options, "infinity",
                         PyFloat_FromDouble(control->infinity));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "errout",
                         PyLong_FromLong(control->errout));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "dual_transformations",
                         PyBool_FromLong(control->dual_transformations));
    PyDict_SetItemString(py_options, "redundant_xc",
                         PyBool_FromLong(control->redundant_xc));
    PyDict_SetItemString(py_options, "primal_constraints_freq",
                         PyLong_FromLong(control->primal_constraints_freq));
    PyDict_SetItemString(py_options, "dual_constraints_freq",
                         PyLong_FromLong(control->dual_constraints_freq));
    PyDict_SetItemString(py_options, "singleton_columns_freq",
                         PyLong_FromLong(control->singleton_columns_freq));
    PyDict_SetItemString(py_options, "doubleton_columns_freq",
                         PyLong_FromLong(control->doubleton_columns_freq));
    PyDict_SetItemString(py_options, "unc_variables_freq",
                         PyLong_FromLong(control->unc_variables_freq));
    PyDict_SetItemString(py_options, "dependent_variables_freq",
                         PyLong_FromLong(control->dependent_variables_freq));
    PyDict_SetItemString(py_options, "sparsify_rows_freq",
                         PyLong_FromLong(control->sparsify_rows_freq));
    PyDict_SetItemString(py_options, "max_fill",
                         PyLong_FromLong(control->max_fill));
    PyDict_SetItemString(py_options, "transf_file_nbr",
                         PyLong_FromLong(control->transf_file_nbr));
    PyDict_SetItemString(py_options, "transf_buffer_size",
                         PyLong_FromLong(control->transf_buffer_size));
    PyDict_SetItemString(py_options, "transf_file_status",
                         PyLong_FromLong(control->transf_file_status));
    PyDict_SetItemString(py_options, "transf_file_name",
                         PyUnicode_FromString(control->transf_file_name));
    PyDict_SetItemString(py_options, "y_sign",
                         PyLong_FromLong(control->y_sign));
    PyDict_SetItemString(py_options, "inactive_y",
                         PyLong_FromLong(control->inactive_y));
    PyDict_SetItemString(py_options, "z_sign",
                         PyLong_FromLong(control->z_sign));
    PyDict_SetItemString(py_options, "inactive_z",
                         PyLong_FromLong(control->inactive_z));
    PyDict_SetItemString(py_options, "final_x_bounds",
                         PyLong_FromLong(control->final_x_bounds));
    PyDict_SetItemString(py_options, "final_z_bounds",
                         PyLong_FromLong(control->final_z_bounds));
    PyDict_SetItemString(py_options, "final_c_bounds",
                         PyLong_FromLong(control->final_c_bounds));
    PyDict_SetItemString(py_options, "final_y_bounds",
                         PyLong_FromLong(control->final_y_bounds));
    PyDict_SetItemString(py_options, "check_primal_feasibility",
                         PyLong_FromLong(control->check_primal_feasibility));
    PyDict_SetItemString(py_options, "check_dual_feasibility",
                         PyLong_FromLong(control->check_dual_feasibility));
    PyDict_SetItemString(py_options, "pivot_tol",
                         PyFloat_FromDouble(control->pivot_tol));
    PyDict_SetItemString(py_options, "min_rel_improve",
                         PyFloat_FromDouble(control->min_rel_improve));
    PyDict_SetItemString(py_options, "max_growth_factor",
                         PyFloat_FromDouble(control->max_growth_factor));
    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within QP Python interfaces
PyObject* presolve_make_inform_dict(const struct presolve_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "status_continue",
                         PyLong_FromLong(inform->status_continue));
    PyDict_SetItemString(py_inform, "status_continued",
                         PyLong_FromLong(inform->status_continued));
    PyDict_SetItemString(py_inform, "nbr_transforms",
                         PyLong_FromLong(inform->nbr_transforms));

    //npy_intp cdim[] = {3,81};
    //PyArrayObject *py_message =
    //   (PyArrayObject*) PyArray_SimpleNew(2, cdim, NPY_STRING);
    //char *message = (char *) PyArray_DATA(py_message);
    //for(int i=0; i<3; i++) {
    //  for(int j=0; j<81; j++) {
    //    *(message + (i*81) + j) = inform->message[i][j];
    //  }
    //}
    //PyDict_SetItemString(py_inform, "message", (PyObject *) py_message);

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   PRESOLVE_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_presolve_initialize(PyObject *self){

    // Call presolve_initialize
    presolve_initialize(&data, &control, &status);

    // Record that PRESOLVE has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = presolve_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-   PRESOLVE_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_presolve_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call presolve_information
    presolve_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = presolve_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   PRESOLVE_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_presolve_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call presolve_terminate
    presolve_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE PRESOLVE PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* presolve python module method table */
static PyMethodDef presolve_module_methods[] = {
    {"initialize", (PyCFunction) py_presolve_initialize, METH_NOARGS,NULL},
    {"information", (PyCFunction) py_presolve_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_presolve_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* presolve python module documentation */

PyDoc_STRVAR(presolve_module_doc,

"The presolve package transforms linear and quadratic programming data \n"
"so that the resulting problem is easier to solve. This reduced problem \n"
"may then be passed to an appropriate solver.  Once the reduced problem \n"
"has been solved, it is then a restored to recover the solution for \n"
"the original formulation.\n"
"\n"
"See $GALAHAD/html/Python/presolve.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* presolve python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "presolve",               /* name of module */
   presolve_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   presolve_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_presolve(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
