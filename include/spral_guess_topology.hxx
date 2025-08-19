/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 09:00 GMT
 *
 * \brief
 * Defines NumaRegion struct.
 */
#pragma once

namespace galahad {
/** \brief Hardware topology module */
namespace hw_topology {

struct NumaRegion {
   int nproc;
   int ngpu;
   int *gpus;
};

extern "C"
void spral_hw_topology_guess(int* nregions, NumaRegion** regions);
extern "C"
void spral_hw_topology_free(int nregions, NumaRegion* regions);

}} /* namespace galahad::hw_topology */
