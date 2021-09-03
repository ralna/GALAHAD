/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \author    Florent Lopez
 */
#include "ssids/profile.hxx"

#ifdef PROFILE
struct timespec spral::ssids::Profile::tstart;
#endif

using namespace spral::ssids;

extern "C"
void spral_ssids_profile_begin(int nregions, void const* regions) {
   Profile::init(nregions, (spral::hw_topology::NumaRegion*)regions);
}

extern "C"
void spral_ssids_profile_end() {
   Profile::end();
}

extern "C"
Profile::Task* spral_ssids_profile_create_task(char const* name, int thread) {
   // We interpret negative thread values as absent
   if(thread >= 0) {
      return new Profile::Task(name, thread);
   } else {
      return new Profile::Task(name);
   }
}

extern "C"
void spral_ssids_profile_end_task(Profile::Task* task) {
   task->done();
   delete task;
}

extern "C"
void spral_ssids_profile_set_state(char const* container, char const* type,
      char const* name) {
   Profile::setState(container, type, name);
}

extern "C"
void spral_ssids_profile_add_event(
      char const* type, char const*val, int thread) {
   Profile::addEvent(type, val, thread);
}
