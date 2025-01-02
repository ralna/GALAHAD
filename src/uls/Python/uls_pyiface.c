//* \file uls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-06-02 AT 12:50 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_ULS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. March 24th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_uls.h"

/* Nested HSL info/inform prototypes */
PyObject* gls_make_ainfo_dict(const struct gls_ainfo_type *ainfo);
PyObject* gls_make_finfo_dict(const struct gls_finfo_type *finfo);
PyObject* gls_make_sinfo_dict(const struct gls_sinfo_type *sinfo);
//PyObject* ma48_make_ainfo_dict(const struct ma48_ainfo_type *ainfo);
//PyObject* ma48_make_finfo_dict(const struct ma48_finfo_type *finfo);
//PyObject* ma48_make_sinfo_dict(const struct ma48_sinfo_type *sinfo);

/* Module global variables */
static void *data;                       // private internal data
static struct uls_control_type control;  // control struct
static struct uls_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within SBLS Python interface
bool uls_update_control(struct uls_control_type *control,
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
        if(strcmp(key_name, "warning") == 0){
            if(!parse_int_option(value, "warning",
                                  &control->warning))
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
        if(strcmp(key_name, "print_level_solver") == 0){
            if(!parse_int_option(value, "print_level_solver",
                                  &control->print_level_solver))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_fill_in_factor") == 0){
            if(!parse_int_option(value, "initial_fill_in_factor",
                                  &control->initial_fill_in_factor))
                return false;
            continue;
        }
        if(strcmp(key_name, "min_real_factor_size") == 0){
            if(!parse_int_option(value, "min_real_factor_size",
                                  &control->min_real_factor_size))
                return false;
            continue;
        }
        if(strcmp(key_name, "min_integer_factor_size") == 0){
            if(!parse_int_option(value, "min_integer_factor_size",
                                  &control->min_integer_factor_size))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_factor_size") == 0){
            if(!parse_int64_t_option(value, "max_factor_size",
                                  &control->max_factor_size))
                return false;
            continue;
        }
        if(strcmp(key_name, "blas_block_size_factorize") == 0){
            if(!parse_int_option(value, "blas_block_size_factorize",
                                  &control->blas_block_size_factorize))
                return false;
            continue;
        }
        if(strcmp(key_name, "blas_block_size_solve") == 0){
            if(!parse_int_option(value, "blas_block_size_solve",
                                  &control->blas_block_size_solve))
                return false;
            continue;
        }
        if(strcmp(key_name, "pivot_control") == 0){
            if(!parse_int_option(value, "pivot_control",
                                  &control->pivot_control))
                return false;
            continue;
        }
        if(strcmp(key_name, "pivot_search_limit") == 0){
            if(!parse_int_option(value, "pivot_search_limit",
                                  &control->pivot_search_limit))
                return false;
            continue;
        }
        if(strcmp(key_name, "minimum_size_for_btf") == 0){
            if(!parse_int_option(value, "minimum_size_for_btf",
                                  &control->minimum_size_for_btf))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_iterative_refinements") == 0){
            if(!parse_int_option(value, "max_iterative_refinements",
                                  &control->max_iterative_refinements))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_if_singular") == 0){
            if(!parse_bool_option(value, "stop_if_singular",
                                  &control->stop_if_singular))
                return false;
            continue;
        }
        if(strcmp(key_name, "array_increase_factor") == 0){
            if(!parse_double_option(value, "array_increase_factor",
                                  &control->array_increase_factor))
                return false;
            continue;
        }
        if(strcmp(key_name, "switch_to_full_code_density") == 0){
            if(!parse_double_option(value, "switch_to_full_code_density",
                                  &control->switch_to_full_code_density))
                return false;
            continue;
        }
        if(strcmp(key_name, "array_decrease_factor") == 0){
            if(!parse_double_option(value, "array_decrease_factor",
                                  &control->array_decrease_factor))
                return false;
            continue;
        }
        if(strcmp(key_name, "relative_pivot_tolerance") == 0){
            if(!parse_double_option(value, "relative_pivot_tolerance",
                                  &control->relative_pivot_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "absolute_pivot_tolerance") == 0){
            if(!parse_double_option(value, "absolute_pivot_tolerance",
                                  &control->absolute_pivot_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "zero_tolerance") == 0){
            if(!parse_double_option(value, "zero_tolerance",
                                  &control->zero_tolerance))
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
// NB not static as it is used for nested inform within SBLS Python interface
PyObject* uls_make_options_dict(const struct uls_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "warning",
                         PyLong_FromLong(control->warning));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "print_level_solver",
                         PyLong_FromLong(control->print_level_solver));
    PyDict_SetItemString(py_options, "initial_fill_in_factor",
                         PyLong_FromLong(control->initial_fill_in_factor));
    PyDict_SetItemString(py_options, "min_real_factor_size",
                         PyLong_FromLong(control->min_real_factor_size));
    PyDict_SetItemString(py_options, "min_integer_factor_size",
                         PyLong_FromLong(control->min_integer_factor_size));
    PyDict_SetItemString(py_options, "max_factor_size",
                         PyLong_FromLong(control->max_factor_size));
    PyDict_SetItemString(py_options, "blas_block_size_factorize",
                         PyLong_FromLong(control->blas_block_size_factorize));
    PyDict_SetItemString(py_options, "blas_block_size_solve",
                         PyLong_FromLong(control->blas_block_size_solve));
    PyDict_SetItemString(py_options, "pivot_control",
                         PyLong_FromLong(control->pivot_control));
    PyDict_SetItemString(py_options, "pivot_search_limit",
                         PyLong_FromLong(control->pivot_search_limit));
    PyDict_SetItemString(py_options, "minimum_size_for_btf",
                         PyLong_FromLong(control->minimum_size_for_btf));
    PyDict_SetItemString(py_options, "max_iterative_refinements",
                         PyLong_FromLong(control->max_iterative_refinements));
    PyDict_SetItemString(py_options, "stop_if_singular",
                         PyBool_FromLong(control->stop_if_singular));
    PyDict_SetItemString(py_options, "array_increase_factor",
                         PyFloat_FromDouble(control->array_increase_factor));
    PyDict_SetItemString(py_options, "switch_to_full_code_density",
                         PyFloat_FromDouble(control->switch_to_full_code_density));
    PyDict_SetItemString(py_options, "array_decrease_factor",
                         PyFloat_FromDouble(control->array_decrease_factor));
    PyDict_SetItemString(py_options, "relative_pivot_tolerance",
                         PyFloat_FromDouble(control->relative_pivot_tolerance));
    PyDict_SetItemString(py_options, "absolute_pivot_tolerance",
                         PyFloat_FromDouble(control->absolute_pivot_tolerance));
    PyDict_SetItemString(py_options, "zero_tolerance",
                         PyFloat_FromDouble(control->zero_tolerance));
    PyDict_SetItemString(py_options, "acceptable_residual_relative",
                         PyFloat_FromDouble(control->acceptable_residual_relative));
    PyDict_SetItemString(py_options, "acceptable_residual_absolute",
                         PyFloat_FromDouble(control->acceptable_residual_absolute));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
//static PyObject* uls_make_time_dict(const struct uls_time_type *time){
//    PyObject *py_time = PyDict_New();

// Set float/double time entries
//    return py_time;
//}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within SBLS Python interface
PyObject* uls_make_inform_dict(const struct uls_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "more_info",
                         PyLong_FromLong(inform->more_info));
    PyDict_SetItemString(py_inform, "out_of_range",
                         PyLong_FromLong(inform->out_of_range));
    PyDict_SetItemString(py_inform, "duplicates",
                         PyLong_FromLong(inform->duplicates));
    PyDict_SetItemString(py_inform, "entries_dropped",
                         PyLong_FromLong(inform->entries_dropped));
    PyDict_SetItemString(py_inform, "workspace_factors",
                         PyLong_FromLong(inform->workspace_factors));
    PyDict_SetItemString(py_inform, "compresses",
                         PyLong_FromLong(inform->compresses));
    PyDict_SetItemString(py_inform, "entries_in_factors",
                         PyLong_FromLong(inform->entries_in_factors));
    PyDict_SetItemString(py_inform, "rank",
                         PyLong_FromLong(inform->rank));
    PyDict_SetItemString(py_inform, "structural_rank",
                         PyLong_FromLong(inform->structural_rank));
    PyDict_SetItemString(py_inform, "pivot_control",
                         PyLong_FromLong(inform->pivot_control));
    PyDict_SetItemString(py_inform, "iterative_refinements",
                         PyLong_FromLong(inform->iterative_refinements));
    PyDict_SetItemString(py_inform, "alternative",
                         PyBool_FromLong(inform->alternative));
    PyDict_SetItemString(py_inform, "gls_ainfo",
                         gls_make_ainfo_dict(&inform->gls_ainfo));
    PyDict_SetItemString(py_inform, "gls_finfo",
                         gls_make_finfo_dict(&inform->gls_finfo));
    PyDict_SetItemString(py_inform, "gls_sinfo",
                         gls_make_sinfo_dict(&inform->gls_sinfo));
    //PyDict_SetItemString(py_inform, "ma48_ainfo",
    //                     ma48_make_inform_dict(&inform->ma48_ainfo));
    //PyDict_SetItemString(py_inform, "ma48_finfo",
    //                     ma48_make_inform_dict(&inform->ma48_finfo));
    //PyDict_SetItemString(py_inform, "ma48_sinfo",
    //                     ma48_make_inform_dict(&inform->ma48_sinfo));
    PyDict_SetItemString(py_inform, "lapack_error",
                         PyLong_FromLong(inform->lapack_error));
    // Set time nested dictionary
    //PyDict_SetItemString(py_inform, "time",
    //                     uls_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   ULS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_uls_initialize(PyObject *self, PyObject *args, PyObject *keywds){
    const char *solver;

    static char *kwlist[] = {"solver", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "s", kwlist, &solver))
        return NULL;

    // Call uls_initialize
    uls_initialize(solver, &data, &control, &status);

    // Record that ULS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = uls_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   ULS_FACTORIZE_MATRIX    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_uls_factorize_matrix(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr, *py_A_val;
    PyObject *py_options = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    double *A_val;
    const char *A_type;
    int m, n, A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"m","n","A_type","A_ne","A_row","A_col","A_ptr",
                             "A_val","options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOOO|O", kwlist,
                                    &m, &n, &A_type, &A_ne, &py_A_row,
                                    &py_A_col, &py_A_ptr, &py_A_val,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("A_row", py_A_row, A_ne) &&
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_int("A_ptr", py_A_ptr, n+1) &&
        check_array_double("A_val", py_A_val, A_ne)
        ))
        return NULL;

    // Convert 64bit integer A_row array to 32bit
    if((PyObject *) py_A_row != Py_None){
        A_row = malloc(A_ne * sizeof(int));
        long int *A_row_long = (long int *) PyArray_DATA(py_A_row);
        for(int i = 0; i < A_ne; i++) A_row[i] = (int) A_row_long[i];
    }

    // Convert 64bit integer A_col array to 32bit
    if((PyObject *) py_A_col != Py_None){
        A_col = malloc(A_ne * sizeof(int));
        long int *A_col_long = (long int *) PyArray_DATA(py_A_col);
        for(int i = 0; i < A_ne; i++) A_col[i] = (int) A_col_long[i];
    }

    // Convert 64bit integer A_ptr array to 32bit
    if((PyObject *) py_A_ptr != Py_None){
        A_ptr = malloc((n+1) * sizeof(int));
        long int *A_ptr_long = (long int *) PyArray_DATA(py_A_ptr);
        for(int i = 0; i < n+1; i++) A_ptr[i] = (int) A_ptr_long[i];
    }

    A_val = (double *) PyArray_DATA(py_A_val);

    // Reset control options
    uls_reset_control(&control, &data, &status);

    // Update ULS control options
    if(!uls_update_control(&control, py_options))
        return NULL;

    // Call uls_analyse_matrix
    uls_factorize_matrix(&control, &data, &status, m, n,
                         A_type, A_ne, A_val, A_row, A_col, A_ptr);

    // Free allocated memory
    if(A_row != NULL) free(A_row);
    if(A_col != NULL) free(A_col);
    if(A_ptr != NULL) free(A_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   ULS_SOLVE_SYSTEM  -*-*-*-*-*-*-*-*

static PyObject* py_uls_solve_system(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_sol;
    double *sol;
    int m, n;
    bool trans;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"m", "n", "b", "trans", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiOb", kwlist, &m, &n, &py_sol, &trans))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("b", py_sol, n))
        return NULL;

    // Get array data pointers
    sol = (double *) PyArray_DATA(py_sol);

    // Call uls_solve_direct
    uls_solve_system(&data, &status, m, n, sol, trans);
    //for( int i = 0; i < n; i++) printf("x %f\n", sol[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x
    PyObject *solve_system_return;
    solve_system_return = Py_BuildValue("O", py_sol);
    Py_INCREF(solve_system_return);
    return solve_system_return;
}

//  *-*-*-*-*-*-*-*-*-*-   ULS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_uls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call uls_information
    uls_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = uls_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   ULS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_uls_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call uls_terminate
    uls_terminate(&data, &control, &inform);

    // Record that ULS must be reinitialised if called again
    init_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE ULS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* uls python module method table */
static PyMethodDef uls_module_methods[] = {
    {"initialize", (PyCFunction) py_uls_initialize, METH_VARARGS | METH_KEYWORDS, NULL},
    {"factorize_matrix", (PyCFunction) py_uls_factorize_matrix,
      METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_system", (PyCFunction) py_uls_solve_system, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_uls_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_uls_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* uls python module documentation */

PyDoc_STRVAR(uls_module_doc,

"The uls package solves dense or sparse symmetric systems of linear \n"
"equations using variants of Gaussian elimination. \n"
"Given a sparse matrix A = {a_ij}_mxn,  and an n-vector b,\n"
"this function solves the system Ax=b or its transpose A^Tx=b.\n"
"Both square (m=n) and rectangular (m/=n) matrices are handled;\n"
"one of an infinite class of solutions for consistent systems will\n"
"be returned whenever A is not of full rank.\n"
"\n"
"The method provides a common interface to a variety of well-known solvers \n"
"from HSL and elsewhere. Currently supported solvers include MA28/GLS, \n"
" and HSL_M48 from HSL, as well as GETR from LAPACK. Note\n"
"that, with the exception the Netlib reference LAPACK code, \n"
"the solvers themselves do not form part of this package and must be \n"
"obtained/linked to separately. Dummy instances are provided for solvers \n"
"that are unavailable. Also note that additional flexibility may be \n"
"obtained by calling the solvers directly rather that via this package.\n"
"\n"
"See $GALAHAD/html/Python/uls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* uls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "uls",               /* name of module */
   uls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   uls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_uls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
