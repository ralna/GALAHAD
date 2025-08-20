/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg and Florent Lopez
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 09:00 GMT
 */

#include "ssids_routines.h"
#include "galahad_precision.h"
#include "ssids_profile.hxx"

#ifdef PROFILE
struct timespec galahad::ssids::Profile::tstart;
#endif

using namespace galahad::ssids;

extern "C"
void galahad_ssids_profile_begin(ipc_ nregions, void const* regions) {
   Profile::init(nregions, (galahad::hw_topology::NumaRegion*)regions);
}

extern "C"
void galahad_ssids_profile_end() {
   Profile::end();
}

extern "C"
Profile::Task* galahad_ssids_profile_create_task(char const* name, 
                                                 ipc_ thread) {
   // We interpret negative thread values as absent
   if(thread >= 0) {
      return new Profile::Task(name, thread);
   } else {
      return new Profile::Task(name);
   }
}

extern "C"
void galahad_ssids_profile_end_task(Profile::Task* task) {
   task->done();
   delete task;
}

extern "C"
void galahad_ssids_profile_set_state(char const* container, char const* type,
      char const* name) {
   Profile::setState(container, type, name);
}

extern "C"
void galahad_ssids_profile_add_event(
      char const* type, char const*val, ipc_ thread) {
   Profile::addEvent(type, val, thread);
}
