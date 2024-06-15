//* \file sbls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-20 AT 10:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SBLS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. March 27th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_sbls.h"

/* Nested HSL info/inform prototypes */
bool sls_update_control(struct sls_control_type *control,
                        PyObject *py_options);
PyObject* sls_make_options_dict(const struct sls_control_type *control);
PyObject* sls_make_inform_dict(const struct sls_inform_type *inform);
bool uls_update_control(struct uls_control_type *control,
                        PyObject *py_options);
PyObject* uls_make_options_dict(const struct uls_control_type *control);
PyObject* uls_make_inform_dict(const struct uls_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct sbls_control_type control;  // control struct
static struct sbls_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within SBLS Python interface
bool sbls_update_control(struct sbls_control_type *control,
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
        if(strcmp(key_name, "indmin") == 0){
            if(!parse_int_option(value, "indmin",
                                  &control->indmin))
                return false;
            continue;
        }
        if(strcmp(key_name, "valmin") == 0){
            if(!parse_int_option(value, "valmin",
                                  &control->valmin))
                return false;
            continue;
        }
        if(strcmp(key_name, "len_ulsmin") == 0){
            if(!parse_int_option(value, "len_ulsmin",
                                  &control->len_ulsmin))
                return false;
            continue;
        }
        if(strcmp(key_name, "itref_max") == 0){
            if(!parse_int_option(value, "itref_max",
                                  &control->itref_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "maxit_pcg") == 0){
            if(!parse_int_option(value, "maxit_pcg",
                                  &control->maxit_pcg))
                return false;
            continue;
        }
        if(strcmp(key_name, "new_a") == 0){
            if(!parse_int_option(value, "new_a",
                                  &control->new_a))
                return false;
            continue;
        }
        if(strcmp(key_name, "new_h") == 0){
            if(!parse_int_option(value, "new_h",
                                  &control->new_h))
                return false;
            continue;
        }
        if(strcmp(key_name, "new_c") == 0){
            if(!parse_int_option(value, "new_c",
                                  &control->new_c))
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
        if(strcmp(key_name, "factorization") == 0){
            if(!parse_int_option(value, "factorization",
                                  &control->factorization))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_col") == 0){
            if(!parse_int_option(value, "max_col",
                                  &control->max_col))
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
        if(strcmp(key_name, "pivot_tol") == 0){
            if(!parse_double_option(value, "pivot_tol",
                                  &control->pivot_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "pivot_tol_for_basis") == 0){
            if(!parse_double_option(value, "pivot_tol_for_basis",
                                  &control->pivot_tol_for_basis))
                return false;
            continue;
        }
        if(strcmp(key_name, "zero_pivot") == 0){
            if(!parse_double_option(value, "zero_pivot",
                                  &control->zero_pivot))
                return false;
            continue;
        }
        if(strcmp(key_name, "static_tolerance") == 0){
            if(!parse_double_option(value, "static_tolerance",
                                  &control->static_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "static_level") == 0){
            if(!parse_double_option(value, "static_level",
                                  &control->static_level))
                return false;
            continue;
        }
        if(strcmp(key_name, "min_diagonal") == 0){
            if(!parse_double_option(value, "min_diagonal",
                                  &control->min_diagonal))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_absolute") == 0){
            if(!parse_double_option(value, "stop_absolute",
                                  &control->stop_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_relative") == 0){
            if(!parse_double_option(value, "stop_relative",
                                  &control->stop_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "remove_dependencies") == 0){
            if(!parse_bool_option(value, "remove_dependencies",
                                  &control->remove_dependencies))
                return false;
            continue;
        }
        if(strcmp(key_name, "find_basis_by_transpose") == 0){
            if(!parse_bool_option(value, "find_basis_by_transpose",
                                  &control->find_basis_by_transpose))
                return false;
            continue;
        }
        if(strcmp(key_name, "affine") == 0){
            if(!parse_bool_option(value, "affine",
                                  &control->affine))
                return false;
            continue;
        }
        if(strcmp(key_name, "allow_singular") == 0){
            if(!parse_bool_option(value, "allow_singular",
                                  &control->allow_singular))
                return false;
            continue;
        }
        if(strcmp(key_name, "perturb_to_make_definite") == 0){
            if(!parse_bool_option(value, "perturb_to_make_definite",
                                  &control->perturb_to_make_definite))
                return false;
            continue;
        }
        if(strcmp(key_name, "get_norm_residual") == 0){
            if(!parse_bool_option(value, "get_norm_residual",
                                  &control->get_norm_residual))
                return false;
            continue;
        }
        if(strcmp(key_name, "check_basis") == 0){
            if(!parse_bool_option(value, "check_basis",
                                  &control->check_basis))
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
        if(strcmp(key_name, "symmetric_linear_solver") == 0){
            if(!parse_char_option(value, "symmetric_linear_solver",
                                  control->symmetric_linear_solver,
                                  sizeof(control->symmetric_linear_solver)))
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
        if(strcmp(key_name, "unsymmetric_linear_solver") == 0){
            if(!parse_char_option(value, "unsymmetric_linear_solver",
                                  control->unsymmetric_linear_solver,
                                  sizeof(control->unsymmetric_linear_solver)))
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
        if(strcmp(key_name, "uls_options") == 0){
            if(!uls_update_control(&control->uls_control, value))
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
// NB not static as it is used for nested inform within CQP Python interface
PyObject* sbls_make_options_dict(const struct sbls_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "indmin",
                         PyLong_FromLong(control->indmin));
    PyDict_SetItemString(py_options, "valmin",
                         PyLong_FromLong(control->valmin));
    PyDict_SetItemString(py_options, "len_ulsmin",
                         PyLong_FromLong(control->len_ulsmin));
    PyDict_SetItemString(py_options, "itref_max",
                         PyLong_FromLong(control->itref_max));
    PyDict_SetItemString(py_options, "maxit_pcg",
                         PyLong_FromLong(control->maxit_pcg));
    PyDict_SetItemString(py_options, "new_a",
                         PyLong_FromLong(control->new_a));
    PyDict_SetItemString(py_options, "new_h",
                         PyLong_FromLong(control->new_h));
    PyDict_SetItemString(py_options, "new_c",
                         PyLong_FromLong(control->new_c));
    PyDict_SetItemString(py_options, "preconditioner",
                         PyLong_FromLong(control->preconditioner));
    PyDict_SetItemString(py_options, "semi_bandwidth",
                         PyLong_FromLong(control->semi_bandwidth));
    PyDict_SetItemString(py_options, "factorization",
                         PyLong_FromLong(control->factorization));
    PyDict_SetItemString(py_options, "max_col",
                         PyLong_FromLong(control->max_col));
    PyDict_SetItemString(py_options, "scaling",
                         PyLong_FromLong(control->scaling));
    PyDict_SetItemString(py_options, "ordering",
                         PyLong_FromLong(control->ordering));
    PyDict_SetItemString(py_options, "pivot_tol",
                         PyFloat_FromDouble(control->pivot_tol));
    PyDict_SetItemString(py_options, "pivot_tol_for_basis",
                         PyFloat_FromDouble(control->pivot_tol_for_basis));
    PyDict_SetItemString(py_options, "zero_pivot",
                         PyFloat_FromDouble(control->zero_pivot));
    PyDict_SetItemString(py_options, "static_tolerance",
                         PyFloat_FromDouble(control->static_tolerance));
    PyDict_SetItemString(py_options, "static_level",
                         PyFloat_FromDouble(control->static_level));
    PyDict_SetItemString(py_options, "min_diagonal",
                         PyFloat_FromDouble(control->min_diagonal));
    PyDict_SetItemString(py_options, "stop_absolute",
                         PyFloat_FromDouble(control->stop_absolute));
    PyDict_SetItemString(py_options, "stop_relative",
                         PyFloat_FromDouble(control->stop_relative));
    PyDict_SetItemString(py_options, "remove_dependencies",
                         PyBool_FromLong(control->remove_dependencies));
    PyDict_SetItemString(py_options, "find_basis_by_transpose",
                         PyBool_FromLong(control->find_basis_by_transpose));
    PyDict_SetItemString(py_options, "affine",
                         PyBool_FromLong(control->affine));
    PyDict_SetItemString(py_options, "allow_singular",
                         PyBool_FromLong(control->allow_singular));
    PyDict_SetItemString(py_options, "perturb_to_make_definite",
                         PyBool_FromLong(control->perturb_to_make_definite));
    PyDict_SetItemString(py_options, "get_norm_residual",
                         PyBool_FromLong(control->get_norm_residual));
    PyDict_SetItemString(py_options, "check_basis",
                         PyBool_FromLong(control->check_basis));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "symmetric_linear_solver",
                         PyUnicode_FromString(control->symmetric_linear_solver));
    PyDict_SetItemString(py_options, "definite_linear_solver",
                         PyUnicode_FromString(control->definite_linear_solver));
    PyDict_SetItemString(py_options, "unsymmetric_linear_solver",
                         PyUnicode_FromString(control->unsymmetric_linear_solver));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "sls_options",
                         sls_make_options_dict(&control->sls_control));
    PyDict_SetItemString(py_options, "uls_options",
                         uls_make_options_dict(&control->uls_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* sbls_make_time_dict(const struct sbls_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries
    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "form",
                         PyFloat_FromDouble(time->form));
    PyDict_SetItemString(py_time, "factorize",
                         PyFloat_FromDouble(time->factorize));
    PyDict_SetItemString(py_time, "apply",
                         PyFloat_FromDouble(time->apply));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_form",
                         PyFloat_FromDouble(time->clock_form));
    PyDict_SetItemString(py_time, "clock_factorize",
                         PyFloat_FromDouble(time->clock_factorize));
    PyDict_SetItemString(py_time, "clock_apply",
                         PyFloat_FromDouble(time->clock_apply));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within SBLS Python interface
PyObject* sbls_make_inform_dict(const struct sbls_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "sort_status",
                         PyLong_FromLong(inform->sort_status));
    PyDict_SetItemString(py_inform, "factorization_integer",
                         PyLong_FromLong(inform->factorization_integer));
    PyDict_SetItemString(py_inform, "factorization_real",
                         PyLong_FromLong(inform->factorization_real));
    PyDict_SetItemString(py_inform, "preconditioner",
                         PyLong_FromLong(inform->preconditioner));
    PyDict_SetItemString(py_inform, "factorization",
                         PyLong_FromLong(inform->factorization));
    PyDict_SetItemString(py_inform, "d_plus",
                         PyLong_FromLong(inform->d_plus));
    PyDict_SetItemString(py_inform, "rank",
                         PyLong_FromLong(inform->rank));
    PyDict_SetItemString(py_inform, "rank_def",
                         PyBool_FromLong(inform->rank_def));
    PyDict_SetItemString(py_inform, "perturbed",
                         PyBool_FromLong(inform->perturbed));
    PyDict_SetItemString(py_inform, "iter_pcg",
                         PyLong_FromLong(inform->iter_pcg));
    PyDict_SetItemString(py_inform, "norm_residual",
                         PyFloat_FromDouble(inform->norm_residual));
    PyDict_SetItemString(py_inform, "alternative",
                         PyBool_FromLong(inform->alternative));
    PyDict_SetItemString(py_inform, "sls_inform",
                         sls_make_inform_dict(&inform->sls_inform));
    PyDict_SetItemString(py_inform, "uls_inform",
                         uls_make_inform_dict(&inform->uls_inform));
    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         sbls_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   SBLS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_sbls_initialize(PyObject *self){

    // Call sbls_initialize
    sbls_initialize(&data, &control, &status);

    // Record that SBLS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = sbls_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   SBLS_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_sbls_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_H_row, *py_H_col, *py_H_ptr;
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
    PyArrayObject *py_C_row, *py_C_col, *py_C_ptr;
    PyObject *py_options = NULL;
    int *H_row = NULL, *H_col = NULL, *H_ptr = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    int *C_row = NULL, *C_col = NULL, *C_ptr = NULL;
    const char *H_type, *A_type, *C_type;
    int n, m, H_ne, A_ne, C_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","m",
                             "H_type","H_ne","H_row","H_col","H_ptr",
                             "A_type","A_ne","A_row","A_col","A_ptr",
                             "C_type","C_ne","C_row","C_col","C_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOOsiOOOsiOOO|O",
                                    kwlist, &n, &m,
                                    &H_type, &H_ne, &py_H_row,
                                    &py_H_col, &py_H_ptr,
                                    &A_type, &A_ne, &py_A_row,
                                    &py_A_col, &py_A_ptr,
                                    &C_type, &C_ne, &py_C_row,
                                    &py_C_col, &py_C_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("H_row", py_H_row, H_ne) &&
        check_array_int("H_col", py_H_col, H_ne) &&
        check_array_int("H_ptr", py_H_ptr, n+1)  &&
        check_array_int("A_row", py_A_row, A_ne) &&
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_int("A_ptr", py_A_ptr, m+1)  &&
        check_array_int("C_row", py_C_row, C_ne) &&
        check_array_int("C_col", py_C_col, C_ne) &&
        check_array_int("C_ptr", py_C_ptr, m+1)
        ))
        return NULL;

    // Convert 64bit integer H_row array to 32bit
    if((PyObject *) py_H_row != Py_None){
        H_row = malloc(H_ne * sizeof(int));
        long int *H_row_long = (long int *) PyArray_DATA(py_H_row);
        for(int i = 0; i < H_ne; i++) H_row[i] = (int) H_row_long[i];
    }

    // Convert 64bit integer H_col array to 32bit
    if((PyObject *) py_H_col != Py_None){
        H_col = malloc(H_ne * sizeof(int));
        long int *H_col_long = (long int *) PyArray_DATA(py_H_col);
        for(int i = 0; i < H_ne; i++) H_col[i] = (int) H_col_long[i];
    }

    // Convert 64bit integer H_ptr array to 32bit
    if((PyObject *) py_H_ptr != Py_None){
        H_ptr = malloc((n+1) * sizeof(int));
        long int *H_ptr_long = (long int *) PyArray_DATA(py_H_ptr);
        for(int i = 0; i < n+1; i++) H_ptr[i] = (int) H_ptr_long[i];
    }

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
        for(int i = 0; i < m+1; i++) A_ptr[i] = (int) A_ptr_long[i];
    }

    // Convert 64bit integer C_row array to 32bit
    if((PyObject *) py_C_row != Py_None){
        C_row = malloc(C_ne * sizeof(int));
        long int *C_row_long = (long int *) PyArray_DATA(py_C_row);
        for(int i = 0; i < C_ne; i++) C_row[i] = (int) C_row_long[i];
    }

    // Convert 64bit integer C_col array to 32bit
    if((PyObject *) py_C_col != Py_None){
        C_col = malloc(C_ne * sizeof(int));
        long int *C_col_long = (long int *) PyArray_DATA(py_C_col);
        for(int i = 0; i < C_ne; i++) C_col[i] = (int) C_col_long[i];
    }

    // Convert 64bit integer C_ptr array to 32bit
    if((PyObject *) py_C_ptr != Py_None){
        C_ptr = malloc((n+1) * sizeof(int));
        long int *C_ptr_long = (long int *) PyArray_DATA(py_C_ptr);
        for(int i = 0; i < m+1; i++) C_ptr[i] = (int) C_ptr_long[i];
    }

    // Reset control options
    sbls_reset_control(&control, &data, &status);

    // Update SBLS control options
    if(!sbls_update_control(&control, py_options))
        return NULL;

    // Call sbls_analyse_matrix
    sbls_import(&control, &data, &status, n, m,
                H_type, H_ne, H_row, H_col, H_ptr,
                A_type, A_ne, A_row, A_col, A_ptr,
                C_type, C_ne, C_row, C_col, C_ptr);

    // Free allocated memory
    if(H_row != NULL) free(H_row);
    if(H_col != NULL) free(H_col);
    if(H_ptr != NULL) free(H_ptr);
    if(A_row != NULL) free(A_row);
    if(A_col != NULL) free(A_col);
    if(A_ptr != NULL) free(A_ptr);
    if(C_row != NULL) free(C_row);
    if(C_col != NULL) free(C_col);
    if(C_ptr != NULL) free(C_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}
//  *-*-*-*-*-*-*-*-*-*-*-*-   SBLS_FACTORIZE_MATRIX    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_sbls_factorize_matrix(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_H_val, *py_A_val, *py_C_val;
    PyArrayObject *py_D = NULL;
    double *H_val, *A_val, *C_val, *D;
    int n, m, H_ne, A_ne, C_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","m","H_ne","H_val","A_ne","A_val",
                             "C_ne","C_val","D",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiiOiOiO|O",
                                    kwlist, &n, &m, &H_ne, &py_H_val, &A_ne,
                                    &py_A_val, &C_ne, &py_C_val, &py_D ))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(check_array_double("H_val", py_H_val, H_ne) &&
         check_array_double("A_val", py_A_val, A_ne) &&
         check_array_double("C_val", py_C_val, C_ne)))
        return NULL;

    // Get array data pointers
    H_val = (double *) PyArray_DATA(py_H_val);
    A_val = (double *) PyArray_DATA(py_A_val);
    C_val = (double *) PyArray_DATA(py_C_val);
    if(py_D != NULL) D = (double *) PyArray_DATA(py_D);

    // Call sbls_factorize_matrix
    sbls_factorize_matrix(&data, &status, n, H_ne, H_val, A_ne, A_val,
                          C_ne, C_val, D );

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   SBLS_SOLVE_SYSTEM  -*-*-*-*-*-*-*-*

static PyObject* py_sbls_solve_system(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_sol;
    double *sol;
    int n, m;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "m", "rhs", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiO", kwlist, &n, &m, &py_sol))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("b", py_sol, n + m))
        return NULL;

    // Get array data pointers
    sol = (double *) PyArray_DATA(py_sol);

    // Call sbls_solve_direct
    sbls_solve_system(&data, &status, n, m, sol);
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

//  *-*-*-*-*-*-*-*-*-*-   SBLS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_sbls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call sbls_information
    sbls_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = sbls_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   SBLS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_sbls_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call sbls_terminate
    sbls_terminate(&data, &control, &inform);

    // Record that SBLS must be reinitialised if called again
    init_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE SBLS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* sbls python module method table */
static PyMethodDef sbls_module_methods[] = {
    {"initialize", (PyCFunction) py_sbls_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_sbls_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"factorize_matrix", (PyCFunction) py_sbls_factorize_matrix,
      METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_system", (PyCFunction) py_sbls_solve_system, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_sbls_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_sbls_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* sbls python module documentation */

PyDoc_STRVAR(sbls_module_doc,

"Given a block, symmetric matrix\n"
"  K_H =  ( H  A^T ),\n"
"         ( A  - C }\n"
"the sbls package constructs a variety of preconditioners of the form\n"
"  K_G =  ( G  A^T ).\n"
"         ( A  - C }\n"
"Here, the leading-block matrix G is a suitably-chosen\n"
"approximation to H; it may either be prescribed explicitly, in\n"
"which case a symmetric indefinite factorization of K_G\n"
"will be formed using the GALAHAD package SLS,\n"
"or implicitly, by requiring certain sub-blocks of G\n"
"be zero. In the latter case, a factorization of K_G will be\n"
"obtained implicitly (and more efficiently) without recourse to SLS.\n"
"In particular, for implicit preconditioners, a reordering\n"
"            ( G_{11}  G_{21}^T  A_1^T )\n"
"  K_{G} = P ( G_{21}  G_{22}    A_2^T ) P^T\n"
"            (  A_1     A_2      - C   )\n"
"involving a suitable permutation P will be found, for some\n"
"invertible sub-block (``basis'') A_1 of the columns of A;\n"
"the selection and factorization of A_1 uses the GALAHAD package ULS.\n"
"Once the preconditioner has been constructed,\n"
"solutions to the preconditioning system\n"
"   ( G  A^T ) ( x ) = ( a )\n"
"   ( A  - C } ( y )   ( b )\n"
"may be obtained by the package.\n"
"Full advantage is taken of any zero coefficients in the matrices H,\n"
"A and C.\n"
"\n"
"See $GALAHAD/html/Python/sbls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* sbls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "sbls",               /* name of module */
   sbls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   sbls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_sbls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
