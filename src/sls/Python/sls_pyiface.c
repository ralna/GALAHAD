//* \file sls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-20 AT 10:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SLS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_sls.h"

/* Nested HSL info/inform prototypes */
PyObject* sils_make_ainfo_dict(const struct sils_ainfo_type *inform);
PyObject* sils_make_finfo_dict(const struct sils_finfo_type *inform);
PyObject* sils_make_sinfo_dict(const struct sils_sinfo_type *inform);
//PyObject* ma57_make_ainfo_dict(const struct ma57_ainfo_type *inform);
//PyObject* ma57_make_finfo_dict(const struct ma57_finfo_type *inform);
//PyObject* ma57_make_sinfo_dict(const struct ma57_sinfo_type *inform);
//PyObject* ma77_make_inform_dict(const struct ma77_inform_type *inform);
//PyObject* ma86_make_inform_dict(const struct ma86_inform_type *inform);
//PyObject* ma87_make_inform_dict(const struct ma87_inform_type *inform);
//PyObject* ma97_make_inform_dict(const struct ma97_inform_type *inform);
//PyObject* ssids_make_inform_dict(const struct ssids_inform_type *inform);
//PyObject* mc64_make_inform_dict(const struct mc64_inform_type *inform);
//PyObject* mc68_make_inform_dict(const struct mc68_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct sls_control_type control;  // control struct
static struct sls_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within SBLS Python interface
bool sls_update_control(struct sls_control_type *control,
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
        if(strcmp(key_name, "statistics") == 0){
            if(!parse_int_option(value, "statistics",
                                  &control->statistics))
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
        if(strcmp(key_name, "bits") == 0){
            if(!parse_int_option(value, "bits",
                                  &control->bits))
                return false;
            continue;
        }
        if(strcmp(key_name, "block_size_kernel") == 0){
            if(!parse_int_option(value, "block_size_kernel",
                                  &control->block_size_kernel))
                return false;
            continue;
        }
        if(strcmp(key_name, "block_size_elimination") == 0){
            if(!parse_int_option(value, "block_size_elimination",
                                  &control->block_size_elimination))
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
        if(strcmp(key_name, "node_amalgamation") == 0){
            if(!parse_int_option(value, "node_amalgamation",
                                  &control->node_amalgamation))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_pool_size") == 0){
            if(!parse_int_option(value, "initial_pool_size",
                                  &control->initial_pool_size))
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
        if(strcmp(key_name, "max_real_factor_size") == 0){
            if(!parse_int64_t_option(value, "max_real_factor_size",
                                  &control->max_real_factor_size))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_integer_factor_size") == 0){
            if(!parse_int64_t_option(value, "max_integer_factor_size",
                                  &control->max_integer_factor_size))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_in_core_store") == 0){
            if(!parse_int64_t_option(value, "max_in_core_store",
                                  &control->max_in_core_store))
                return false;
            continue;
        }
        if(strcmp(key_name, "array_increase_factor") == 0){
            if(!parse_double_option(value, "array_increase_factor",
                                  &control->array_increase_factor))
                return false;
            continue;
        }
        if(strcmp(key_name, "array_decrease_factor") == 0){
            if(!parse_double_option(value, "array_decrease_factor",
                                  &control->array_decrease_factor))
                return false;
            continue;
        }
        if(strcmp(key_name, "pivot_control") == 0){
            if(!parse_int_option(value, "pivot_control",
                                  &control->pivot_control))
                return false;
            continue;
        }
        if(strcmp(key_name, "ordering") == 0){
            if(!parse_int_option(value, "ordering",
                                  &control->ordering))
                return false;
            continue;
        }
        if(strcmp(key_name, "full_row_threshold") == 0){
            if(!parse_int_option(value, "full_row_threshold",
                                  &control->full_row_threshold))
                return false;
            continue;
        }
        if(strcmp(key_name, "row_search_indefinite") == 0){
            if(!parse_int_option(value, "row_search_indefinite",
                                  &control->row_search_indefinite))
                return false;
            continue;
        }
        if(strcmp(key_name, "scaling") == 0){
            if(!parse_int_option(value, "scaling",
                                  &control->scaling))
                return false;
            continue;
        }
        if(strcmp(key_name, "scale_maxit") == 0){
            if(!parse_int_option(value, "scale_maxit",
                                  &control->scale_maxit))
                return false;
            continue;
        }
        if(strcmp(key_name, "scale_thresh") == 0){
            if(!parse_double_option(value, "scale_thresh",
                                  &control->scale_thresh))
                return false;
            continue;
        }
        if(strcmp(key_name, "relative_pivot_tolerance") == 0){
            if(!parse_double_option(value, "relative_pivot_tolerance",
                                  &control->relative_pivot_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "minimum_pivot_tolerance") == 0){
            if(!parse_double_option(value, "minimum_pivot_tolerance",
                                  &control->minimum_pivot_tolerance))
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
        if(strcmp(key_name, "zero_pivot_tolerance") == 0){
            if(!parse_double_option(value, "zero_pivot_tolerance",
                                  &control->zero_pivot_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "negative_pivot_tolerance") == 0){
            if(!parse_double_option(value, "negative_pivot_tolerance",
                                  &control->negative_pivot_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "static_pivot_tolerance") == 0){
            if(!parse_double_option(value, "static_pivot_tolerance",
                                  &control->static_pivot_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "static_level_switch") == 0){
            if(!parse_double_option(value, "static_level_switch",
                                  &control->static_level_switch))
                return false;
            continue;
        }
        if(strcmp(key_name, "consistency_tolerance") == 0){
            if(!parse_double_option(value, "consistency_tolerance",
                                  &control->consistency_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_iterative_refinements") == 0){
            if(!parse_int_option(value, "max_iterative_refinements",
                                  &control->max_iterative_refinements))
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
        if(strcmp(key_name, "multiple_rhs") == 0){
            if(!parse_bool_option(value, "multiple_rhs",
                                  &control->multiple_rhs))
                return false;
            continue;
        }
        if(strcmp(key_name, "generate_matrix_file") == 0){
            if(!parse_bool_option(value, "generate_matrix_file",
                                  &control->generate_matrix_file))
                return false;
            continue;
        }
        if(strcmp(key_name, "matrix_file_device") == 0){
            if(!parse_int_option(value, "matrix_file_device",
                                  &control->matrix_file_device))
                return false;
            continue;
        }
        if(strcmp(key_name, "matrix_file_name") == 0){
            if(!parse_char_option(value, "matrix_file_name",
                                  control->matrix_file_name,
                                  sizeof(control->matrix_file_name)))
                return false;
            continue;
        }
        if(strcmp(key_name, "out_of_core_directory") == 0){
            if(!parse_char_option(value, "out_of_core_directory",
                                  control->out_of_core_directory,
                                  sizeof(control->out_of_core_directory)))
                return false;
            continue;
        }
        if(strcmp(key_name, "out_of_core_integer_factor_file") == 0){
            if(!parse_char_option(value, "out_of_core_integer_factor_file",
                                  control->out_of_core_integer_factor_file,
                                  sizeof(control->out_of_core_integer_factor_file)))
                return false;
            continue;
        }
        if(strcmp(key_name, "out_of_core_real_factor_file") == 0){
            if(!parse_char_option(value, "out_of_core_real_factor_file",
                                  control->out_of_core_real_factor_file,
                                  sizeof(control->out_of_core_real_factor_file)))
                return false;
            continue;
        }
        if(strcmp(key_name, "out_of_core_real_work_file") == 0){
            if(!parse_char_option(value, "out_of_core_real_work_file",
                                  control->out_of_core_real_work_file,
                                  sizeof(control->out_of_core_real_work_file)))
                return false;
            continue;
        }
        if(strcmp(key_name, "out_of_core_indefinite_file") == 0){
            if(!parse_char_option(value, "out_of_core_indefinite_file",
                                  control->out_of_core_indefinite_file,
                                  sizeof(control->out_of_core_indefinite_file)))
                return false;
            continue;
        }
        if(strcmp(key_name, "out_of_core_restart_file") == 0){
            if(!parse_char_option(value, "out_of_core_restart_file",
                                  control->out_of_core_restart_file,
                                  sizeof(control->out_of_core_restart_file)))
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
PyObject* sls_make_options_dict(const struct sls_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "warning",
                         PyLong_FromLong(control->warning));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "statistics",
                         PyLong_FromLong(control->statistics));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "print_level_solver",
                         PyLong_FromLong(control->print_level_solver));
    PyDict_SetItemString(py_options, "bits",
                         PyLong_FromLong(control->bits));
    PyDict_SetItemString(py_options, "block_size_kernel",
                         PyLong_FromLong(control->block_size_kernel));
    PyDict_SetItemString(py_options, "block_size_elimination",
                         PyLong_FromLong(control->block_size_elimination));
    PyDict_SetItemString(py_options, "blas_block_size_factorize",
                         PyLong_FromLong(control->blas_block_size_factorize));
    PyDict_SetItemString(py_options, "blas_block_size_solve",
                         PyLong_FromLong(control->blas_block_size_solve));
    PyDict_SetItemString(py_options, "node_amalgamation",
                         PyLong_FromLong(control->node_amalgamation));
    PyDict_SetItemString(py_options, "initial_pool_size",
                         PyLong_FromLong(control->initial_pool_size));
    PyDict_SetItemString(py_options, "min_real_factor_size",
                         PyLong_FromLong(control->min_real_factor_size));
    PyDict_SetItemString(py_options, "min_integer_factor_size",
                         PyLong_FromLong(control->min_integer_factor_size));
    PyDict_SetItemString(py_options, "max_real_factor_size",
                         PyLong_FromLong(control->max_real_factor_size));
    PyDict_SetItemString(py_options, "max_integer_factor_size",
                         PyLong_FromLong(control->max_integer_factor_size));
    PyDict_SetItemString(py_options, "max_in_core_store",
                         PyLong_FromLong(control->max_in_core_store));
    PyDict_SetItemString(py_options, "array_increase_factor",
                         PyFloat_FromDouble(control->array_increase_factor));
    PyDict_SetItemString(py_options, "array_decrease_factor",
                         PyFloat_FromDouble(control->array_decrease_factor));
    PyDict_SetItemString(py_options, "pivot_control",
                         PyLong_FromLong(control->pivot_control));
    PyDict_SetItemString(py_options, "ordering",
                         PyLong_FromLong(control->ordering));
    PyDict_SetItemString(py_options, "full_row_threshold",
                         PyLong_FromLong(control->full_row_threshold));
    PyDict_SetItemString(py_options, "row_search_indefinite",
                         PyLong_FromLong(control->row_search_indefinite));
    PyDict_SetItemString(py_options, "scaling",
                         PyLong_FromLong(control->scaling));
    PyDict_SetItemString(py_options, "scale_maxit",
                         PyLong_FromLong(control->scale_maxit));
    PyDict_SetItemString(py_options, "scale_thresh",
                         PyFloat_FromDouble(control->scale_thresh));
    PyDict_SetItemString(py_options, "relative_pivot_tolerance",
                         PyFloat_FromDouble(control->relative_pivot_tolerance));
    PyDict_SetItemString(py_options, "minimum_pivot_tolerance",
                         PyFloat_FromDouble(control->minimum_pivot_tolerance));
    PyDict_SetItemString(py_options, "absolute_pivot_tolerance",
                         PyFloat_FromDouble(control->absolute_pivot_tolerance));
    PyDict_SetItemString(py_options, "zero_tolerance",
                         PyFloat_FromDouble(control->zero_tolerance));
    PyDict_SetItemString(py_options, "zero_pivot_tolerance",
                         PyFloat_FromDouble(control->zero_pivot_tolerance));
    PyDict_SetItemString(py_options, "negative_pivot_tolerance",
                         PyFloat_FromDouble(control->negative_pivot_tolerance));
    PyDict_SetItemString(py_options, "static_pivot_tolerance",
                         PyFloat_FromDouble(control->static_pivot_tolerance));
    PyDict_SetItemString(py_options, "static_level_switch",
                         PyFloat_FromDouble(control->static_level_switch));
    PyDict_SetItemString(py_options, "consistency_tolerance",
                         PyFloat_FromDouble(control->consistency_tolerance));
    PyDict_SetItemString(py_options, "max_iterative_refinements",
                         PyLong_FromLong(control->max_iterative_refinements));
    PyDict_SetItemString(py_options, "acceptable_residual_relative",
                         PyFloat_FromDouble(control->acceptable_residual_relative));
    PyDict_SetItemString(py_options, "acceptable_residual_absolute",
                         PyFloat_FromDouble(control->acceptable_residual_absolute));
    PyDict_SetItemString(py_options, "multiple_rhs",
                         PyBool_FromLong(control->multiple_rhs));
    PyDict_SetItemString(py_options, "generate_matrix_file",
                         PyBool_FromLong(control->generate_matrix_file));
    PyDict_SetItemString(py_options, "matrix_file_device",
                         PyLong_FromLong(control->matrix_file_device));
    PyDict_SetItemString(py_options, "matrix_file_name",
                         PyUnicode_FromString(control->matrix_file_name));
    PyDict_SetItemString(py_options, "out_of_core_directory",
                         PyUnicode_FromString(control->out_of_core_directory));
    PyDict_SetItemString(py_options, "out_of_core_integer_factor_file",
                         PyUnicode_FromString(control->out_of_core_integer_factor_file));
    PyDict_SetItemString(py_options, "out_of_core_real_factor_file",
                         PyUnicode_FromString(control->out_of_core_real_factor_file));
    PyDict_SetItemString(py_options, "out_of_core_real_work_file",
                         PyUnicode_FromString(control->out_of_core_real_work_file));
    PyDict_SetItemString(py_options, "out_of_core_indefinite_file",
                         PyUnicode_FromString(control->out_of_core_indefinite_file));
    PyDict_SetItemString(py_options, "out_of_core_restart_file",
                         PyUnicode_FromString(control->out_of_core_restart_file));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* sls_make_time_dict(const struct sls_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries
    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "analyse",
                         PyFloat_FromDouble(time->analyse));
    PyDict_SetItemString(py_time, "factorize",
                         PyFloat_FromDouble(time->factorize));
    PyDict_SetItemString(py_time, "solve",
                         PyFloat_FromDouble(time->solve));
    PyDict_SetItemString(py_time, "order_external",
                         PyFloat_FromDouble(time->order_external));
    PyDict_SetItemString(py_time, "analyse_external",
                         PyFloat_FromDouble(time->analyse_external));
    PyDict_SetItemString(py_time, "factorize_external",
                         PyFloat_FromDouble(time->factorize_external));
    PyDict_SetItemString(py_time, "solve_external",
                         PyFloat_FromDouble(time->solve_external));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_analyse",
                         PyFloat_FromDouble(time->clock_analyse));
    PyDict_SetItemString(py_time, "clock_factorize",
                         PyFloat_FromDouble(time->clock_factorize));
    PyDict_SetItemString(py_time, "clock_solve",
                         PyFloat_FromDouble(time->clock_solve));
    PyDict_SetItemString(py_time, "clock_order_external",
                         PyFloat_FromDouble(time->clock_order_external));
    PyDict_SetItemString(py_time, "clock_analyse_external",
                         PyFloat_FromDouble(time->clock_analyse_external));
    PyDict_SetItemString(py_time, "clock_factorize_external",
                         PyFloat_FromDouble(time->clock_factorize_external));
    PyDict_SetItemString(py_time, "clock_solve_external",
                         PyFloat_FromDouble(time->clock_solve_external));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within SBLS Python interface
PyObject* sls_make_inform_dict(const struct sls_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "more_info",
                         PyLong_FromLong(inform->more_info));
    PyDict_SetItemString(py_inform, "entries",
                         PyLong_FromLong(inform->entries));
    PyDict_SetItemString(py_inform, "out_of_range",
                         PyLong_FromLong(inform->out_of_range));
    PyDict_SetItemString(py_inform, "duplicates",
                         PyLong_FromLong(inform->duplicates));
    PyDict_SetItemString(py_inform, "upper",
                         PyLong_FromLong(inform->upper));
    PyDict_SetItemString(py_inform, "missing_diagonals",
                         PyLong_FromLong(inform->missing_diagonals));
    PyDict_SetItemString(py_inform, "max_depth_assembly_tree",
                         PyLong_FromLong(inform->max_depth_assembly_tree));
    PyDict_SetItemString(py_inform, "nodes_assembly_tree",
                         PyLong_FromLong(inform->nodes_assembly_tree));
    PyDict_SetItemString(py_inform, "real_size_desirable",
                         PyLong_FromLong(inform->real_size_desirable));
    PyDict_SetItemString(py_inform, "integer_size_desirable",
                         PyLong_FromLong(inform->integer_size_desirable));
    PyDict_SetItemString(py_inform, "real_size_necessary",
                         PyLong_FromLong(inform->real_size_necessary));
    PyDict_SetItemString(py_inform, "integer_size_necessary",
                         PyLong_FromLong(inform->integer_size_necessary));
    PyDict_SetItemString(py_inform, "real_size_factors",
                         PyLong_FromLong(inform->real_size_factors));
    PyDict_SetItemString(py_inform, "integer_size_factors",
                         PyLong_FromLong(inform->integer_size_factors));
    PyDict_SetItemString(py_inform, "entries_in_factors",
                         PyLong_FromLong(inform->entries_in_factors));
    PyDict_SetItemString(py_inform, "max_task_pool_size",
                         PyLong_FromLong(inform->max_task_pool_size));
    PyDict_SetItemString(py_inform, "max_front_size",
                         PyLong_FromLong(inform->max_front_size));
    PyDict_SetItemString(py_inform, "compresses_real",
                         PyLong_FromLong(inform->compresses_real));
    PyDict_SetItemString(py_inform, "compresses_integer",
                         PyLong_FromLong(inform->compresses_integer));
    PyDict_SetItemString(py_inform, "two_by_two_pivots",
                         PyLong_FromLong(inform->two_by_two_pivots));
    PyDict_SetItemString(py_inform, "semi_bandwidth",
                         PyLong_FromLong(inform->semi_bandwidth));
    PyDict_SetItemString(py_inform, "delayed_pivots",
                         PyLong_FromLong(inform->delayed_pivots));
    PyDict_SetItemString(py_inform, "pivot_sign_changes",
                         PyLong_FromLong(inform->pivot_sign_changes));
    PyDict_SetItemString(py_inform, "static_pivots",
                         PyLong_FromLong(inform->static_pivots));
    PyDict_SetItemString(py_inform, "first_modified_pivot",
                         PyLong_FromLong(inform->first_modified_pivot));
    PyDict_SetItemString(py_inform, "rank",
                         PyLong_FromLong(inform->rank));
    PyDict_SetItemString(py_inform, "negative_eigenvalues",
                         PyLong_FromLong(inform->negative_eigenvalues));
    PyDict_SetItemString(py_inform, "num_zero",
                         PyLong_FromLong(inform->num_zero));
    PyDict_SetItemString(py_inform, "iterative_refinements",
                         PyLong_FromLong(inform->iterative_refinements));
    PyDict_SetItemString(py_inform, "flops_assembly",
                         PyLong_FromLong(inform->flops_assembly));
    PyDict_SetItemString(py_inform, "flops_elimination",
                         PyLong_FromLong(inform->flops_elimination));
    PyDict_SetItemString(py_inform, "flops_blas",
                         PyLong_FromLong(inform->flops_blas));
    PyDict_SetItemString(py_inform, "largest_modified_pivot",
                         PyFloat_FromDouble(inform->largest_modified_pivot));
    PyDict_SetItemString(py_inform, "minimum_scaling_factor",
                         PyFloat_FromDouble(inform->minimum_scaling_factor));
    PyDict_SetItemString(py_inform, "maximum_scaling_factor",
                         PyFloat_FromDouble(inform->maximum_scaling_factor));
    PyDict_SetItemString(py_inform, "condition_number_1",
                         PyFloat_FromDouble(inform->condition_number_1));
    PyDict_SetItemString(py_inform, "condition_number_2",
                         PyFloat_FromDouble(inform->condition_number_2));
    PyDict_SetItemString(py_inform, "backward_error_1",
                         PyFloat_FromDouble(inform->backward_error_1));
    PyDict_SetItemString(py_inform, "backward_error_2",
                         PyFloat_FromDouble(inform->backward_error_2));
    PyDict_SetItemString(py_inform, "forward_error",
                         PyFloat_FromDouble(inform->forward_error));
    PyDict_SetItemString(py_inform, "alternative",
                         PyBool_FromLong(inform->alternative));
    PyDict_SetItemString(py_inform, "sils_ainfo",
                         sils_make_ainfo_dict(&inform->sils_ainfo));
    PyDict_SetItemString(py_inform, "sils_finfo",
                         sils_make_finfo_dict(&inform->sils_finfo));
    PyDict_SetItemString(py_inform, "sils_sinfo",
                         sils_make_sinfo_dict(&inform->sils_sinfo));
    //PyDict_SetItemString(py_inform, "ma57_ainfo",
    //                     ma57_make_ainfo_dict(&inform->ma57_ainfo));
    //PyDict_SetItemString(py_inform, "ma57_finfo",
    //                     ma57_make_finfo_dict(&inform->ma57_finfo));
    //PyDict_SetItemString(py_inform, "ma57_sinfo",
    //                     ma57_make_sinfo_dict(&inform->ma57_sinfo));
    //PyDict_SetItemString(py_inform, "ma77_inform",
    //                     ma77_make_inform_dict(&inform->ma77_inform));
    //PyDict_SetItemString(py_inform, "ma86_inform",
    //                     ma86_make_inform_dict(&inform->ma86_inform));
    //PyDict_SetItemString(py_inform, "ma87_inform",
    //                     ma87_make_inform_dict(&inform->ma87_inform));
    //PyDict_SetItemString(py_inform, "ma97_inform",
    //                     ma97_make_inform_dict(&inform->ma97_inform));
    //PyDict_SetItemString(py_inform, "ssids_inform",
    //                     ssids_make_inform_dict(&inform->ssids_inform));
    //PyDict_SetItemString(py_inform, "mc61_info",
    //                     PyLong_FromLong(inform->mc61_info));
    //PyDict_SetItemString(py_inform, "mc61_rinfo",
    //                     PyFloat_FromDouble(inform->mc61_rinfo));
    //PyDict_SetItemString(py_inform, "mc64_inform",
    //                     mc64_make_inform_dict(&inform->mc64_inform));
    //PyDict_SetItemString(py_inform, "mc68_inform",
    //                     mc68_make_inform_dict(&inform->mc68_inform));
    //PyDict_SetItemString(py_inform, "mc77_info",
    //                     PyLong_FromLong(inform->mc77_info));
    //PyDict_SetItemString(py_inform, "mc77_rinfo",
    //                     PyFloat_FromDouble(inform->mc77_rinfo));
    //PyDict_SetItemString(py_inform, "mumps_error",
    //                     PyLong_FromLong(inform->mumps_error));
    //PyDict_SetItemString(py_inform, "mumps_info",
    //                     PyLong_FromLong(inform->mumps_imfo));
    //PyDict_SetItemString(py_inform, "mumps_rinfo",
    //                     PyFloat_FromDouble(inform->mumps_rinfo));
    //PyDict_SetItemString(py_inform, "pardiso_error",
    //                     PyLong_FromLong(inform->pardiso_error));
    //PyDict_SetItemString(py_inform, "pardiso_IPARM",
    //                     PyLong_FromLong(inform->pardiso_IPARM));
    //PyDict_SetItemString(py_inform, "pardiso_DPARM",
    //                     PyFloat_FromDouble(inform->pardiso_DPARM));
    //PyDict_SetItemString(py_inform, "mkl_pardiso_error",
    //                     PyLong_FromLong(inform->mkl_pardiso_error));
    //PyDict_SetItemString(py_inform, "mkl_pardiso_IPARM",
    //                     PyLong_FromLong(inform->mkl_pardiso_IPARM));
    //PyDict_SetItemString(py_inform, "pastix_error",
    //                     PyLong_FromLong(inform->pastix_error));
    //PyDict_SetItemString(py_inform, "wsmp_error",
    //                     PyLong_FromLong(inform->wsmp_error));
    //PyDict_SetItemString(py_inform, "wsmp_iparm",
    //                     PyLong_FromLong(inform->wsmp_iparm));
    //PyDict_SetItemString(py_inform, "wsmp_dparm",
    //                     PyFloat_FromDouble(inform->wsmp_dparm));
    //PyDict_SetItemString(py_inform, "mpi_ierr",
    //                     PyLong_FromLong(inform->mpi_ierr));
    PyDict_SetItemString(py_inform, "lapack_error",
                         PyLong_FromLong(inform->lapack_error));
    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         sls_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   SLS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_sls_initialize(PyObject *self, PyObject *args, PyObject *keywds){
    const char *solver;

    static char *kwlist[] = {"solver", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "s", kwlist, &solver))
        return NULL;

    // Call sls_initialize
    sls_initialize(solver, &data, &control, &status);

    // Record that SLS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = sls_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   SLS_ANALYSE_MATRIX    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_sls_analyse_matrix(PyObject *self, PyObject *args, PyObject *keywds){
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
    sls_reset_control(&control, &data, &status);

    // Update SLS control options
    if(!sls_update_control(&control, py_options))
        return NULL;

    // Call sls_analyse_matrix
    sls_analyse_matrix(&control, &data, &status, n,
                       A_type, A_ne, A_row, A_col, A_ptr);

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
//  *-*-*-*-*-*-*-*-*-*-*-*-   SLS_FACTORIZE_MATRIX    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_sls_factorize_matrix(PyObject *self, PyObject *args, PyObject *keywds){
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

    // Call sls_factorize_matrix
    sls_factorize_matrix(&data, &status, A_ne, A_val);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   SLS_SOLVE_SYSTEM  -*-*-*-*-*-*-*-*

static PyObject* py_sls_solve_system(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_sol;
    double *sol;
    int n;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "b", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iO", kwlist, &n, &py_sol))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("b", py_sol, n))
        return NULL;

    // Get array data pointers
    sol = (double *) PyArray_DATA(py_sol);

    // Call sls_solve_direct
    sls_solve_system(&data, &status, n, sol);
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

//  *-*-*-*-*-*-*-*-*-*-   SLS_PARTIAL_SOLVE_SYSTEM  -*-*-*-*-*-*-*-*

static PyObject* py_sls_partial_solve_system(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_b;
    const char part;
    double *sol;
    int n;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"part", "n", "b", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "siO", kwlist, &part, &n, &py_b))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_b, n))
        return NULL;

    // Get array data pointer
    sol = (double *) PyArray_DATA(py_b);

    // Call sls_solve_direct
    sls_partial_solve_system(&part, &data, &status, n, sol);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Wrap C arrays as NumPy arrays
    npy_intp ndim[] = {n}; // size of x
    PyObject *py_x = PyArray_SimpleNewFromData(1, ndim,
                        NPY_DOUBLE, (void *) sol); // create NumPy x array

    // Return x
    PyObject *solve_system_return;
    solve_system_return = Py_BuildValue("O", py_x);
    Py_INCREF(solve_system_return);
    return solve_system_return;
}

//  *-*-*-*-*-*-*-*-*-*-   SLS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_sls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call sls_information
    sls_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = sls_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   SLS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_sls_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call sls_terminate
    sls_terminate(&data, &control, &inform);

    // Record that SLS must be reinitialised if called again
    init_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE SLS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* sls python module method table */
static PyMethodDef sls_module_methods[] = {
    {"initialize", (PyCFunction) py_sls_initialize, METH_VARARGS | METH_KEYWORDS, NULL},
    {"analyse_matrix", (PyCFunction) py_sls_analyse_matrix, METH_VARARGS | METH_KEYWORDS, NULL},
    {"factorize_matrix", (PyCFunction) py_sls_factorize_matrix,
      METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_system", (PyCFunction) py_sls_solve_system, METH_VARARGS | METH_KEYWORDS, NULL},
    {"partial_solve_system", (PyCFunction) py_sls_partial_solve_system,
      METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_sls_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_sls_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* sls python module documentation */

PyDoc_STRVAR(sls_module_doc,

"The sls package solves dense or sparse symmetric systems of linear \n"
"equations using variants of Gaussian elimination. \n"
"Given a sparse symmetric matrix \n"
"A = {a_ij}_nxn,  and an n-vector b or a matrix B = {b_ij}_mxn,\n"
"this function solves the system Ax=b or the system AX=B.\n"
"The matrix A need not be definite.\n"
"\n"
"The method provides a common interface to a variety of well-known solvers \n"
"from HSL and elsewhere. Currently supported solvers include MA27/SILS, \n"
"HSL_MA57, HSL_MA77 , HSL_MA86, HSL_MA87 and HSL_MA97 from HSL, SSIDS \n"
"from SPRAL, MUMPS from Mumps Technologies, PARDISO both from the \n"
"Pardiso Project and Intel’s MKL, PaStiX from Inria, and WSMP from the \n"
"IBM alpha Works, as well as POTR, SYTR and SBTR from LAPACK. Note\n"
"that, with the exception of SSIDS and the Netlib reference LAPACK codes, \n"
"the solvers themselves do not form part of this package and must be \n"
"obtained/linked to separately. Dummy instances are provided for solvers \n"
"that are unavailable. Also note that additional flexibility may be \n"
"obtained by calling the solvers directly rather that via this package.\n"
"\n"
"See $GALAHAD/html/Python/sls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* sls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "sls",               /* name of module */
   sls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   sls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_sls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
