//* \file eqp_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.0 - 2024-06-15 AT 11:50 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_EQP PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 5th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_eqp.h"

/* Nested FDC, SBLS and GLTR control and inform prototypes */
bool fdc_update_control(struct fdc_control_type *control,
                        PyObject *py_options);
PyObject* fdc_make_options_dict(const struct fdc_control_type *control);
PyObject* fdc_make_inform_dict(const struct fdc_inform_type *inform);
bool sbls_update_control(struct sbls_control_type *control,
                         PyObject *py_options);
PyObject* sbls_make_options_dict(const struct sbls_control_type *control);
PyObject* sbls_make_inform_dict(const struct sbls_inform_type *inform);
bool gltr_update_control(struct gltr_control_type *control,
                         PyObject *py_options);
PyObject* gltr_make_options_dict(const struct gltr_control_type *control);
PyObject* gltr_make_inform_dict(const struct gltr_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct eqp_control_type control;  // control struct
static struct eqp_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool eqp_update_control(struct eqp_control_type *control,
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
        if(strcmp(key_name, "cg_maxit") == 0){
            if(!parse_int_option(value, "cg_maxit",
                                  &control->cg_maxit))
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
        if(strcmp(key_name, "sif_file_device") == 0){
            if(!parse_int_option(value, "sif_file_device",
                                  &control->sif_file_device))
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
        if(strcmp(key_name, "inner_fraction_opt") == 0){
            if(!parse_double_option(value, "inner_fraction_opt",
                                  &control->inner_fraction_opt))
                return false;
            continue;
        }
        if(strcmp(key_name, "radius") == 0){
            if(!parse_double_option(value, "radius",
                                  &control->radius))
                return false;
            continue;
        }
        if(strcmp(key_name, "min_diagonal") == 0){
            if(!parse_double_option(value, "min_diagonal",
                                  &control->min_diagonal))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_infeasibility_relative") == 0){
            if(!parse_double_option(value, "max_infeasibility_relative",
                                  &control->max_infeasibility_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_infeasibility_absolute") == 0){
            if(!parse_double_option(value, "max_infeasibility_absolute",
                                  &control->max_infeasibility_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "inner_stop_relative") == 0){
            if(!parse_double_option(value, "inner_stop_relative",
                                  &control->inner_stop_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "inner_stop_absolute") == 0){
            if(!parse_double_option(value, "inner_stop_absolute",
                                  &control->inner_stop_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "inner_stop_inter") == 0){
            if(!parse_double_option(value, "inner_stop_inter",
                                  &control->inner_stop_inter))
                return false;
            continue;
        }
        if(strcmp(key_name, "find_basis_by_transpose") == 0){
            if(!parse_bool_option(value, "find_basis_by_transpose",
                                  &control->find_basis_by_transpose))
                return false;
            continue;
        }
        if(strcmp(key_name, "remove_dependencies") == 0){
            if(!parse_bool_option(value, "remove_dependencies",
                                  &control->remove_dependencies))
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
        if(strcmp(key_name, "generate_sif_file") == 0){
            if(!parse_bool_option(value, "generate_sif_file",
                                  &control->generate_sif_file))
                return false;
            continue;
        }
        if(strcmp(key_name, "sif_file_name") == 0){
            if(!parse_char_option(value, "sif_file_name",
                                  control->sif_file_name,
                                  sizeof(control->sif_file_name)))
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
        if(strcmp(key_name, "fdc_options") == 0){
            if(!fdc_update_control(&control->fdc_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "sbls_options") == 0){
            if(!sbls_update_control(&control->sbls_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "gltr_options") == 0){
            if(!gltr_update_control(&control->gltr_control, value))
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
PyObject* eqp_make_options_dict(const struct eqp_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "factorization",
                         PyLong_FromLong(control->factorization));
    PyDict_SetItemString(py_options, "max_col",
                         PyLong_FromLong(control->max_col));
    PyDict_SetItemString(py_options, "indmin",
                         PyLong_FromLong(control->indmin));
    PyDict_SetItemString(py_options, "valmin",
                         PyLong_FromLong(control->valmin));
    PyDict_SetItemString(py_options, "len_ulsmin",
                         PyLong_FromLong(control->len_ulsmin));
    PyDict_SetItemString(py_options, "itref_max",
                         PyLong_FromLong(control->itref_max));
    PyDict_SetItemString(py_options, "cg_maxit",
                         PyLong_FromLong(control->cg_maxit));
    PyDict_SetItemString(py_options, "preconditioner",
                         PyLong_FromLong(control->preconditioner));
    PyDict_SetItemString(py_options, "semi_bandwidth",
                         PyLong_FromLong(control->semi_bandwidth));
    PyDict_SetItemString(py_options, "new_a",
                         PyLong_FromLong(control->new_a));
    PyDict_SetItemString(py_options, "new_h",
                         PyLong_FromLong(control->new_h));
    PyDict_SetItemString(py_options, "sif_file_device",
                         PyLong_FromLong(control->sif_file_device));
    PyDict_SetItemString(py_options, "pivot_tol",
                         PyFloat_FromDouble(control->pivot_tol));
    PyDict_SetItemString(py_options, "pivot_tol_for_basis",
                         PyFloat_FromDouble(control->pivot_tol_for_basis));
    PyDict_SetItemString(py_options, "zero_pivot",
                         PyFloat_FromDouble(control->zero_pivot));
    PyDict_SetItemString(py_options, "inner_fraction_opt",
                         PyFloat_FromDouble(control->inner_fraction_opt));
    PyDict_SetItemString(py_options, "radius",
                         PyFloat_FromDouble(control->radius));
    PyDict_SetItemString(py_options, "min_diagonal",
                         PyFloat_FromDouble(control->min_diagonal));
    PyDict_SetItemString(py_options, "max_infeasibility_relative",
                         PyFloat_FromDouble(control->max_infeasibility_relative));
    PyDict_SetItemString(py_options, "max_infeasibility_absolute",
                         PyFloat_FromDouble(control->max_infeasibility_absolute));
    PyDict_SetItemString(py_options, "inner_stop_relative",
                         PyFloat_FromDouble(control->inner_stop_relative));
    PyDict_SetItemString(py_options, "inner_stop_absolute",
                         PyFloat_FromDouble(control->inner_stop_absolute));
    PyDict_SetItemString(py_options, "inner_stop_inter",
                         PyFloat_FromDouble(control->inner_stop_inter));
    PyDict_SetItemString(py_options, "find_basis_by_transpose",
                         PyBool_FromLong(control->find_basis_by_transpose));
    PyDict_SetItemString(py_options, "remove_dependencies",
                         PyBool_FromLong(control->remove_dependencies));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "generate_sif_file",
                         PyBool_FromLong(control->generate_sif_file));
    PyDict_SetItemString(py_options, "sif_file_name",
                         PyUnicode_FromString(control->sif_file_name));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "fdc_options",
                         fdc_make_options_dict(&control->fdc_control));
    PyDict_SetItemString(py_options, "sbls_options",
                         sbls_make_options_dict(&control->sbls_control));
    PyDict_SetItemString(py_options, "gltr_options",
                        gltr_make_options_dict(&control->gltr_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* eqp_make_time_dict(const struct eqp_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries

    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "find_dependent",
                         PyFloat_FromDouble(time->find_dependent));
    PyDict_SetItemString(py_time, "factorize",
                         PyFloat_FromDouble(time->factorize));
    PyDict_SetItemString(py_time, "solve",
                         PyFloat_FromDouble(time->solve));
    PyDict_SetItemString(py_time, "solve_inter",
                         PyFloat_FromDouble(time->solve_inter));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_find_dependent",
                         PyFloat_FromDouble(time->clock_find_dependent));
    PyDict_SetItemString(py_time, "clock_factorize",
                         PyFloat_FromDouble(time->clock_factorize));
    PyDict_SetItemString(py_time, "clock_solve",
                         PyFloat_FromDouble(time->clock_solve));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* eqp_make_inform_dict(const struct eqp_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "cg_iter",
                         PyLong_FromLong(inform->cg_iter));
    PyDict_SetItemString(py_inform, "cg_iter_inter",
                         PyLong_FromLong(inform->cg_iter_inter));
    PyDict_SetItemString(py_inform, "factorization_integer",
                         PyLong_FromLong(inform->factorization_integer));
    PyDict_SetItemString(py_inform, "factorization_real",
                         PyLong_FromLong(inform->factorization_real));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         eqp_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "fdc_inform",
                         fdc_make_inform_dict(&inform->fdc_inform));
    PyDict_SetItemString(py_inform, "sbls_inform",
                        sbls_make_inform_dict(&inform->sbls_inform));
    PyDict_SetItemString(py_inform, "gltr_inform",
                         gltr_make_inform_dict(&inform->gltr_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   EQP_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_eqp_initialize(PyObject *self){

    // Call eqp_initialize
    eqp_initialize(&data, &control, &status);

    // Record that EQP has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = eqp_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   EQP_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_eqp_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_H_row, *py_H_col, *py_H_ptr;
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
//    PyArrayObject *py_w;
    PyObject *py_options = NULL;
    int *H_row = NULL, *H_col = NULL, *H_ptr = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    const char *A_type, *H_type;
//    double *w;
    int n, m, A_ne, H_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","m",
                             "H_type","H_ne","H_row","H_col","H_ptr",
                             "A_type","A_ne","A_row","A_col","A_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOOsiOOO|O",
                                    kwlist, &n, &m,
                                    &H_type, &H_ne, &py_H_row,
                                    &py_H_col, &py_H_ptr,
                                    &A_type, &A_ne, &py_A_row,
                                    &py_A_col, &py_A_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("H_row", py_H_row, H_ne) &&
        check_array_int("H_col", py_H_col, H_ne) &&
        check_array_int("H_ptr", py_H_ptr, n+1)
        ))
        return NULL;
    if(!(
        check_array_int("A_row", py_A_row, A_ne) &&
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_int("A_ptr", py_A_ptr, m+1)
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
        A_ptr = malloc((m+1) * sizeof(int));
        long int *A_ptr_long = (long int *) PyArray_DATA(py_A_ptr);
        for(int i = 0; i < m+1; i++) A_ptr[i] = (int) A_ptr_long[i];
    }

    // Reset control options
    eqp_reset_control(&control, &data, &status);

    // Update EQP control options
    if(!eqp_update_control(&control, py_options))
        return NULL;

    // Call eqp_import
    eqp_import(&control, &data, &status, n, m,
               H_type, H_ne, H_row, H_col, H_ptr,
               A_type, A_ne, A_row, A_col, A_ptr);

    // Free allocated memory
    if(H_row != NULL) free(H_row);
    if(H_col != NULL) free(H_col);
    if(H_ptr != NULL) free(H_ptr);
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

//  *-*-*-*-*-*-*-*-*-*-   EQP_SOLVE_QP   -*-*-*-*-*-*-*-*

static PyObject* py_eqp_solve_qp(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_g, *py_H_val, *py_A_val;
    PyArrayObject *py_c, *py_x, *py_y;
    double *g, *H_val, *A_val, *c, *x, *y;
    int n, m, H_ne, A_ne;
    double f;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "m", "f", "g", "H_ne", "H_val", "A_ne", "A_val",
                             "c", "x", "y", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iidOiOiOOOO", kwlist, &n, &m, &f, &py_g,
                                    &H_ne, &py_H_val, &A_ne, &py_A_val,
                                    &py_c, &py_x, &py_y))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_g, n))
        return NULL;
    if(!check_array_double("H_val", py_H_val, H_ne))
        return NULL;
    if(!check_array_double("A_val", py_A_val, A_ne))
        return NULL;
    if(!check_array_double("c", py_c, m))
        return NULL;
    if(!check_array_double("x", py_x, n))
        return NULL;
    if(!check_array_double("y", py_y, m))
        return NULL;

    // Get array data pointer
    g = (double *) PyArray_DATA(py_g);
    H_val = (double *) PyArray_DATA(py_H_val);
    A_val = (double *) PyArray_DATA(py_A_val);
    c = (double *) PyArray_DATA(py_c);
    x = (double *) PyArray_DATA(py_x);
    y = (double *) PyArray_DATA(py_y);

    // Call eqp_solve_direct
    status = 1; // set status to 1 on entry
    eqp_solve_qp(&data, &status, n, m, H_ne, H_val, g, f, A_ne, A_val, c, x, y);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x and y
    PyObject *solve_qp_return;

    // solve_qp_return = Py_BuildValue("O", py_x);
    solve_qp_return = Py_BuildValue("OO", py_x, py_y);
    Py_INCREF(solve_qp_return);
    return solve_qp_return;

}
//  *-*-*-*-*-*-*-*-*-*-   EQP_SOLVE_SLDQP   -*-*-*-*-*-*-*-*

static PyObject* py_eqp_solve_sldqp(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_g, *py_w, *py_x0, *py_A_val;
    PyArrayObject *py_c, *py_x, *py_y;
    double *g, *w, *x0, *A_val, *c, *x, *y;
    int n, m, A_ne;
    double f;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "m", "f", "g", "w", "x0", "A_ne", "A_val",
                             "c", "x", "y", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iidOOOiOOOO", kwlist, &n, &m, &f, &py_g,
                                    &py_w, &py_x0, &A_ne, &py_A_val,
                                    &py_c, &py_x, &py_y))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_g, n))
        return NULL;
    if(!check_array_double("w", py_w, n))
        return NULL;
    if(!check_array_double("x0", py_x0, n))
        return NULL;
    if(!check_array_double("A_val", py_A_val, A_ne))
        return NULL;
    if(!check_array_double("c", py_c, m))
        return NULL;
    if(!check_array_double("x", py_x, n))
        return NULL;
    if(!check_array_double("y", py_y, m))
        return NULL;

    // Get array data pointer
    g = (double *) PyArray_DATA(py_g);
    w = (double *) PyArray_DATA(py_w);
    x0 = (double *) PyArray_DATA(py_x0);
    A_val = (double *) PyArray_DATA(py_A_val);
    c = (double *) PyArray_DATA(py_c);
    x = (double *) PyArray_DATA(py_x);
    y = (double *) PyArray_DATA(py_y);

    // Call eqp_solve_direct
    status = 1; // set status to 1 on entry
    eqp_solve_sldqp(&data, &status, n, m, w, x0, g, f, A_ne, A_val, c, x, y);

    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x, c, y, z, x_stat and c_stat
    PyObject *solve_sldqp_return;
    solve_sldqp_return = Py_BuildValue("OO", py_x, py_y);
    Py_INCREF(solve_sldqp_return);
    return solve_sldqp_return;
}

//  *-*-*-*-*-*-*-*-*-*-   EQP_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_eqp_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call eqp_information
    eqp_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = eqp_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   EQP_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_eqp_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call eqp_terminate
    eqp_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE EQP PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* eqp python module method table */
static PyMethodDef eqp_module_methods[] = {
    {"initialize", (PyCFunction) py_eqp_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_eqp_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_qp", (PyCFunction) py_eqp_solve_qp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_sldqp", (PyCFunction) py_eqp_solve_sldqp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_eqp_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_eqp_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* eqp python module documentation */

PyDoc_STRVAR(eqp_module_doc,

"The eqp package uses an iterative method to solve the a given\n"
"equality-constrained quadratic programming problem. The aim is to\n"
"minimize the quadratic objective\n"
"q(x) = 1/2 x^T H x + g^T x + f,\n"
"or the shifted least-distance objective\n"
"1/2 sum_{j=1}^n w_j^2 ( x_j - x_j^0 )^2,\n"
"subject to the general linear equality constraints\n"
"A x + c = 0,\n"
"where H and A are, respectively, given n by n symmetric and m by $n$\n"
"general matrices,  g, w, x^0 and c are vectors, and f is a scalar.\n"
"The method is most suitable for problems\n"
"involving a large number of unknowns x.\n"
"\n"
"See $GALAHAD/html/Python/eqp.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* eqp python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "eqp",               /* name of module */
   eqp_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   eqp_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_eqp(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
