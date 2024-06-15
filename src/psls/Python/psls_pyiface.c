//* \file psls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-20 AT 10:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_PSLS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. March 23rd 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_psls.h"

/* Nested SLS and HSL info/inform prototypes */
bool sls_update_control(struct sls_control_type *control,
                        PyObject *py_options);
PyObject* sls_make_options_dict(const struct sls_control_type *control);
PyObject* sls_make_inform_dict(const struct sls_inform_type *inform);

//bool mi28_update_control(struct mi28_control_type *control,
//                        PyObject *py_options);
//PyObject* mi28_make_options_dict(const struct mi28_control_type *control);
//PyObject* mi28_make_inform_dict(const struct mi28_info_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct psls_control_type control; // control struct
static struct psls_inform_type inform;   // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within TRU Python interface
bool psls_update_control(struct psls_control_type *control,
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
        if(strcmp(key_name, "preconditioner") == 0){
            if(!parse_int_option(value, "preconditioner",
                                  &control->preconditioner))
                return false;
            continue;
        }
        if(strcmp(key_name, "semi_bandwidth") == 0){
            if(!parse_int_option(value, "semi_bandwidth",
                                  &control->semi_bandwidth))
                return false;
            continue;
        }
        if(strcmp(key_name, "scaling") == 0){
            if(!parse_int_option(value, "scaling",
                                  &control->scaling))
                return false;
            continue;
        }
        if(strcmp(key_name, "ordering") == 0){
            if(!parse_int_option(value, "ordering",
                                  &control->ordering))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_col") == 0){
            if(!parse_int_option(value, "max_col",
                                  &control->max_col))
                return false;
            continue;
        }
        if(strcmp(key_name, "icfs_vectors") == 0){
            if(!parse_int_option(value, "icfs_vectors",
                                  &control->icfs_vectors))
                return false;
            continue;
        }
        if(strcmp(key_name, "mi28_lsize") == 0){
            if(!parse_int_option(value, "mi28_lsize",
                                  &control->mi28_lsize))
                return false;
            continue;
        }
        if(strcmp(key_name, "mi28_rsize") == 0){
            if(!parse_int_option(value, "mi28_rsize",
                                  &control->mi28_rsize))
                return false;
            continue;
        }
        if(strcmp(key_name, "min_diagonal") == 0){
            if(!parse_double_option(value, "min_diagonal",
                                  &control->min_diagonal))
                return false;
            continue;
        }
        if(strcmp(key_name, "new_structure") == 0){
            if(!parse_bool_option(value, "new_structure",
                                  &control->new_structure))
                return false;
            continue;
        }
        if(strcmp(key_name, "get_semi_bandwidth") == 0){
            if(!parse_bool_option(value, "get_semi_bandwidth",
                                  &control->get_semi_bandwidth))
                return false;
            continue;
        }
        if(strcmp(key_name, "get_norm_residual") == 0){
            if(!parse_bool_option(value, "get_norm_residual",
                                  &control->get_norm_residual))
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
        if(strcmp(key_name, "definite_linear_solver") == 0){
            if(!parse_char_option(value, "definite_linear_solver",
                                  control->definite_linear_solver,
                                  sizeof(control->definite_linear_solver)))
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
        if(strcmp(key_name, "sls_options") == 0){
            if(!sls_update_control(&control->sls_control, value))
                return false;
            continue;
        }
        //if(strcmp(key_name, "mi28_options") == 0){
        //    if(!mi28_update_control(&control->mi28_control, value))
        //        return false;
        //    continue;
        //}
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
PyObject* psls_make_options_dict(const struct psls_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "preconditioner",
                         PyLong_FromLong(control->preconditioner));
    PyDict_SetItemString(py_options, "semi_bandwidth",
                         PyLong_FromLong(control->semi_bandwidth));
    PyDict_SetItemString(py_options, "scaling",
                         PyLong_FromLong(control->scaling));
    PyDict_SetItemString(py_options, "ordering",
                         PyLong_FromLong(control->ordering));
    PyDict_SetItemString(py_options, "max_col",
                         PyLong_FromLong(control->max_col));
    PyDict_SetItemString(py_options, "icfs_vectors",
                         PyLong_FromLong(control->icfs_vectors));
    PyDict_SetItemString(py_options, "mi28_lsize",
                         PyLong_FromLong(control->mi28_lsize));
    PyDict_SetItemString(py_options, "mi28_rsize",
                         PyLong_FromLong(control->mi28_rsize));
    PyDict_SetItemString(py_options, "min_diagonal",
                         PyFloat_FromDouble(control->min_diagonal));
    PyDict_SetItemString(py_options, "new_structure",
                         PyBool_FromLong(control->new_structure));
    PyDict_SetItemString(py_options, "get_semi_bandwidth",
                         PyBool_FromLong(control->get_semi_bandwidth));
    PyDict_SetItemString(py_options, "get_norm_residual",
                         PyBool_FromLong(control->get_norm_residual));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "definite_linear_solver",
                         PyUnicode_FromString(control->definite_linear_solver));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "sls_options",
                         sls_make_options_dict(&control->sls_control));
    //PyDict_SetItemString(py_options, "mi28_options",
    //                     mi28_make_options_dict(&control->mi28_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* psls_make_time_dict(const struct psls_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries
    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "assemble",
                         PyFloat_FromDouble(time->assemble));
    PyDict_SetItemString(py_time, "analyse",
                         PyFloat_FromDouble(time->analyse));
    PyDict_SetItemString(py_time, "factorize",
                         PyFloat_FromDouble(time->factorize));
    PyDict_SetItemString(py_time, "solve",
                         PyFloat_FromDouble(time->solve));
    PyDict_SetItemString(py_time, "update",
                         PyFloat_FromDouble(time->update));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_assemble",
                         PyFloat_FromDouble(time->clock_assemble));
    PyDict_SetItemString(py_time, "clock_analyse",
                         PyFloat_FromDouble(time->clock_analyse));
    PyDict_SetItemString(py_time, "clock_factorize",
                         PyFloat_FromDouble(time->clock_factorize));
    PyDict_SetItemString(py_time, "clock_solve",
                         PyFloat_FromDouble(time->clock_solve));
    PyDict_SetItemString(py_time, "clock_update",
                         PyFloat_FromDouble(time->clock_update));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within TRU Python interface
PyObject* psls_make_inform_dict(const struct psls_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "analyse_status",
                         PyLong_FromLong(inform->analyse_status));
    PyDict_SetItemString(py_inform, "factorize_status",
                         PyLong_FromLong(inform->factorize_status));
    PyDict_SetItemString(py_inform, "solve_status",
                         PyLong_FromLong(inform->solve_status));
    PyDict_SetItemString(py_inform, "factorization_integer",
                         PyLong_FromLong(inform->factorization_integer));
    PyDict_SetItemString(py_inform, "factorization_real",
                         PyLong_FromLong(inform->factorization_real));
    PyDict_SetItemString(py_inform, "preconditioner",
                         PyLong_FromLong(inform->preconditioner));
    PyDict_SetItemString(py_inform, "semi_bandwidth",
                         PyLong_FromLong(inform->semi_bandwidth));
    PyDict_SetItemString(py_inform, "reordered_semi_bandwidth",
                         PyLong_FromLong(inform->reordered_semi_bandwidth));
    PyDict_SetItemString(py_inform, "out_of_range",
                         PyLong_FromLong(inform->out_of_range));
    PyDict_SetItemString(py_inform, "duplicates",
                         PyLong_FromLong(inform->duplicates));
    PyDict_SetItemString(py_inform, "upper",
                         PyLong_FromLong(inform->upper));
    PyDict_SetItemString(py_inform, "missing_diagonals",
                         PyLong_FromLong(inform->missing_diagonals));
    PyDict_SetItemString(py_inform, "semi_bandwidth_used",
                         PyLong_FromLong(inform->semi_bandwidth_used));
    PyDict_SetItemString(py_inform, "neg1",
                         PyLong_FromLong(inform->neg1));
    PyDict_SetItemString(py_inform, "neg2",
                         PyLong_FromLong(inform->neg2));
    PyDict_SetItemString(py_inform, "perturbed",
                         PyBool_FromLong(inform->perturbed));
    PyDict_SetItemString(py_inform, "fill_in_ratio",
                         PyFloat_FromDouble(inform->fill_in_ratio));
    PyDict_SetItemString(py_inform, "norm_residual",
                         PyFloat_FromDouble(inform->norm_residual));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    // PyDict_SetItemString(py_inform, "mc61_info",
    //                      PyLong_FromLong(inform->mc61_info));
    // PyDict_SetItemString(py_inform, "mc61_rinfo",
    //                      PyFloat_FromDouble(inform->mc61_rinfo));
    PyDict_SetItemString(py_inform, "sls_inform",
                         sls_make_inform_dict(&inform->sls_inform));
    // PyDict_SetItemString(py_inform, "mi28_info",
    //                     mi_make_inform_dict(&inform->mi28_info));
    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         psls_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   PSLS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_psls_initialize(PyObject *self){

    // Call psls_initialize
    psls_initialize(&data, &control, &status);

    // Record that PSLS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = psls_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   PSLS_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_psls_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
    PyObject *py_options = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    const char *A_type;
    int n, A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","A_type","A_ne","A_row","A_col","A_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "isiOOO|O", kwlist,
                                    &n, &A_type, &A_ne, &py_A_row,
                                    &py_A_col, &py_A_ptr, &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("A_row", py_A_row, A_ne) &&
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_int("A_ptr", py_A_ptr, n+1)
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

    // Reset control options
    psls_reset_control(&control, &data, &status);

    // Update PSLS control options
    if(!psls_update_control(&control, py_options))
        return NULL;

    // Call psls_analyse_matrix
    psls_import(&control, &data, &status, n, A_type, A_ne, A_row, A_col, A_ptr);

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

//  *-*-*-*-*-*-*-*-*-*-*-*-   PSLS_FORM_PRECONDITIONER    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_psls_form_preconditioner(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_A_val;
    double *A_val;
    int A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"A_ne","A_val",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iO",
                                    kwlist, &A_ne, &py_A_val))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(check_array_double("A_val", py_A_val, A_ne)))
        return NULL;

    // Get array data pointers
    A_val = (double *) PyArray_DATA(py_A_val);

    // Call psls_factorize_matrix
    psls_form_preconditioner(&data, &status, A_ne, A_val);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   PSLS_FORM_SUBSET_PRECONDITIONER    -*-*-*-*-*-*-*-*

static PyObject* py_psls_form_subset_preconditioner(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_A_val, *py_sub;
    int *sub = NULL;
    double *A_val;
    int A_ne, n_sub;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"A_ne","A_val","n_sub","sub",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iOiO",
                                    kwlist, &A_ne, &py_A_val, &n_sub, &py_sub))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(check_array_double("A_val", py_A_val, A_ne) &&
         check_array_int("sub", py_sub, n_sub)))
        return NULL;

    // Get array data pointers
    A_val = (double *) PyArray_DATA(py_A_val);
    if((PyObject *) py_sub != Py_None){
        sub = malloc(n_sub * sizeof(int));
        long int *sub_long = (long int *) PyArray_DATA(py_sub);
        for(int i = 0; i < n_sub; i++) sub[i] = (int) sub_long[i];
    }

    // Call psls_factorize_matrix
    psls_form_subset_preconditioner(&data, &status, A_ne, A_val, n_sub, sub);

    // Free allocated memory
    if(sub != NULL) free(sub);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   PSLS_APPLY_PRECONDITIONER  -*-*-*-*-*-*-*-*

static PyObject* py_psls_apply_preconditioner(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_sol;
    double *sol;
    int n;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "b", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iO", kwlist, &n, &py_sol))

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("b", py_sol, n))
        return NULL;

    // Get array data pointers
    sol = (double *) PyArray_DATA(py_sol);

    // Call psls_solve_direct
    psls_apply_preconditioner(&data, &status, n, sol);
    // for( int i = 0; i < n; i++) printf("x %f\n", sol[i]);

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

//  *-*-*-*-*-*-*-*-*-*-   PSLS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_psls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call psls_information
    psls_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = psls_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   PSLS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_psls_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call psls_terminate
    psls_terminate(&data, &control, &inform);

    // Record that PSLS must be reinitialised if called again
    init_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE PSLS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* psls python module method table */
static PyMethodDef psls_module_methods[] = {
    {"initialize", (PyCFunction) py_psls_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_psls_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"form_preconditioner",
      (PyCFunction) py_psls_form_preconditioner, METH_VARARGS | METH_KEYWORDS, NULL},
    {"form_subset_preconditioner",
      (PyCFunction) py_psls_form_subset_preconditioner, METH_VARARGS | METH_KEYWORDS, NULL},
    {"apply_preconditioner",
      (PyCFunction) py_psls_apply_preconditioner, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_psls_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_psls_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* psls python module documentation */

PyDoc_STRVAR(psls_module_doc,

"The psls package solves dense or sparse symmetric systems of linear \n"
"equations using variants of Gaussian elimination. \n"
"Given a sparse symmetric matrix \n"
"A = {a_ij}_nxn,  and an n-vector $b$ or a matrix B = {b_ij}_mxn,\n"
"this function builds a suitable symmetric, positive definite—or/n"
"diagonally dominant—preconditioner P of A, or a symmetric sub-matrix./n"
"thereof. The matrix A need not be definite. Facilities are provided./n"
"to apply the preconditioner to a given vector, and to remove rows and./n"
"columns (symmetrically) from the initial preconditioner without a./n"
"full re-factorization../n"
"\n"
"The method relies on a variety of well-known solvers \n"
"from HSL and elsewhere. Currently supported solvers include MA27/SILS, \n"
"HSL_MA57, HSL_MA77 , HSL_MA86, HSL_MA87 and HSL_MA97 from HSL, SSIDS \n"
"from SPRAL, MUMPS from Mumps Technologies, PARDISO both from the \n"
"Pardiso Project and Intel’s MKL, PaStiX from Inria, and WSMP from the \n"
"IBM alpha Works, as well as POTR, SYTR and SBTR from LAPACK. Note\n"
"that, with the exception of SSIDS and the Netlib reference LAPACK codes, \n"
"the solvers themselves do not form part of this package and must be \n"
"obtained/linked to separately. Dummy instances are provided for solvers \n"
"that are unavailable.\n"
"\n"
"See $GALAHAD/html/Python/psls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* psls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "psls",               /* name of module */
   psls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   psls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_psls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
