/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 4.3 - 2024-02-03 AT 15:00 GMT
 */
#pragma once

#include <vector>
#include "ssids_rip.hxx"

namespace spral { namespace ssids { namespace cpu {

/** Symbolic representation of a node */
struct SymbolicNode {
   bool insmallleaf;
   ipc_ idx; //< Index of node
   ipc_ nrow; //< Number of rows
   ipc_ ncol; //< Number of columns
   SymbolicNode* first_child; //< Pointer to first child in linked list
   SymbolicNode* next_child; //< Pointer to second child in linked list
   ipc_ const* rlist; //< Pointer to row lists
   ipc_ num_a; //< Number of entries mapped from A to L
   longc_ const* amap; //< Pointer to map from A to L locations
   ipc_ parent; //< index of parent node
   std::vector<ipc_> contrib; //< index of expected contribution(s)
};

}}} /* end of namespace spral::ssids::cpu */
