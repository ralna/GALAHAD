/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#include "config.h"
#include <time.h>

//#define PROFILE

#if defined(PROFILE) && !defined(HAVE_GTG)
#error "Cannot enable profiling without GTG library"
#endif

#include <cstdio>

#ifdef HAVE_GTG
extern "C" {
#include <GTG.h>
}
#endif /* HAVE_GTG */
#ifdef HAVE_SCHED_GETCPU
#include <sched.h>
#endif /* HAVE_SCHED_GETCPU */

#include "hw_topology/guess_topology.hxx"
#include "omp.hxx"

namespace spral { namespace ssids {

/**
 * \brief Provide easy to use wrapper around GTG to record execution traces.
 *
 * Whilst GTG nicely wraps the file format, this class also builds in knowledge
 * of the SSIDS application (e.g. what tasks we have) and handles timing and
 * topology as well.
 *
 * \note If PROFILE is not defined (by ./configure --enable-profile) most
 *       of these calls are no-ops.
 */
class Profile {
public:
   /**
    * \brief Represents a single Task that begins upon constructions and ends
    *        when done() is called.
    */
   class Task {
   public:
      /**
       * \brief Constructor. Starts task timer.
       *
       * \note Task state change not acutally written until done() is called.
       *
       * \param name Predefined name of task, as setup in Profile::init().
       * \param thread Optional thread number, otherwise use best guess.
       */
      Task(char const* name, int thread=Profile::guess_core())
      : name(name), thread(thread), t1(Profile::now())
      {}

      /**
       * \brief Stop task timer and write event out to profile.
       */
      void done() {
#if defined(PROFILE) && defined(HAVE_GTG)
         double t2 = Profile::now();
         ::setState(t1, "ST_TASK", Profile::get_thread_name(thread), name);
         ::setState(t2, "ST_TASK", Profile::get_thread_name(thread), "0");
#endif
      }

   private:
      char const* name; //< Name of task, one defined in Profile::init().
      int thread; //< Thread of task.
      double t1; //< Start time of task.
   };

   /**
    * \brief Set a particular state (e.g. in a particular task)
    * \param name Name of state.
    * \param thread Optional thread number, otherwise use best guess.
    */
   static
   void setState(char const* name, int thread=Profile::guess_core()) {
#if defined(PROFILE) && defined(HAVE_GTG)
      double t = Profile::now();
      ::setState(t, "ST_TASK", Profile::get_thread_name(thread), name);
#endif
   }

   /**
    * \brief Set a particular state (e.g. in a particular task)
    * \param name Name of state.
    * \param container Container on which to set state
    */
   static
   void setState(char const* container, char const* type, char const* name) {
#if defined(PROFILE) && defined(HAVE_GTG)
      double t = Profile::now();
      ::setState(t, type, container, name);
#endif
   }

   /**
    * \brief Set state as "0", signifying not executing.
    * \param thread Optional thread number, otherwise use best guess.
    */
   static
   void setNullState(int thread=Profile::guess_core()) {
      setState("0", thread);
   }

   /**
    * \brief Add an event at current time.
    * \param type Type of event, one of those predefined in Profile::init().
    * \param val A value to associated with the event ie specific description.
    * \param thread Optional thread number, otherwise use best guess.
    */
   static
   void addEvent(char const* type, char const*val,
         int thread=Profile::guess_core()) {
#if defined(PROFILE) && defined(HAVE_GTG)
      ::addEvent(now(), type, get_thread_name(thread), val);
#endif
   };

   /**
    * \brief Open trace file, initialise topoology and define states and events.
    *
    * \note Times are all measured from the end of this subroutine.
    */
   static
   // void init(int nregions, spral::hw_topology::NumaRegion* regions) {
   void init(int nnodes, spral::hw_topology::NumaRegion* nodes) {
#if defined(PROFILE) && defined(HAVE_GTG)
      // Initialise profiling
      setTraceType(PAJE);
      initTrace("ssids", 0, GTG_FLAG_NONE);
      // Define containers (i.e. nodes/threads)
      addContType("CT_NODE", "0", "Node");
      addContType("CT_THREAD", "CT_NODE", "Thread");
      addContType("CT_GPU", "CT_NODE", "GPU");
      // int nnodes = 0;
      // spral::hw_topology::NumaRegion* nodes;
      if (!nodes) spral_hw_topology_guess(&nnodes, &nodes);
      int core_idx=0;
      for(int node=0; node<nnodes; ++node) {
         char node_id[100], node_name[100];
         snprintf(node_id, 100, "C_Node%d", node);
         snprintf(node_name, 100, "Node %d", node);
         addContainer(0.0, node_id, "CT_NODE", "0", node_name, "0");
         for(int i=0; i<nodes[node].nproc; ++i) {
            char core_name[100];
            snprintf(core_name, 100, "Core %d", core_idx);
            addContainer(0.0, get_thread_name(core_idx), "CT_THREAD", node_id,
                  core_name, "0");
            core_idx++;
         }
         for(int gpu=0; gpu<nodes[node].ngpu; ++gpu) {
            char gpu_name[100], gpu_id[100];
            snprintf(gpu_id, 100, "C_GPU%d", nodes[node].gpus[gpu]);
            snprintf(gpu_name, 100, "GPU %d", nodes[node].gpus[gpu]);
            addContainer(0.0, gpu_id, "CT_GPU", node_id, gpu_name, "0");
         }
      }
      // Define states (i.e. task types)
      // GTG_WHITE, GTG_BLACK, GTG_DARKGREY,
      // GTG_LIGHTBROWN, GTG_LIGHTGREY, GTG_DARKBLUE, GTG_DARKPINK
      // GTG_LIGHTPINK
      // CPU tasks
      addStateType("ST_TASK", "CT_THREAD", "Task");
      addEntityValue("TA_SUBTREE", "ST_TASK", "Subtree", GTG_RED);
      addEntityValue("TA_ASSEMBLE", "ST_TASK", "Assemble", GTG_GREEN);
      addEntityValue("TA_CHOL_DIAG", "ST_TASK", "CholDiag", GTG_PURPLE);
      addEntityValue("TA_CHOL_TRSM", "ST_TASK", "CholTrsm", GTG_PINK);
      addEntityValue("TA_CHOL_UPD", "ST_TASK", "CholUpd", GTG_SEABLUE);
      addEntityValue("TA_LDLT_DIAG", "ST_TASK", "LDLTDiag", GTG_PURPLE);
      addEntityValue("TA_LDLT_APPLY", "ST_TASK", "LDLTTrsm", GTG_PINK);
      addEntityValue("TA_LDLT_ADJUST", "ST_TASK", "LDLTTrsm", GTG_GRENAT);
      addEntityValue("TA_LDLT_UPDA", "ST_TASK", "LDLT Upd A", GTG_SEABLUE);
      addEntityValue("TA_LDLT_UPDC", "ST_TASK", "LDLT Upd C", GTG_ORANGE);
      addEntityValue("TA_LDLT_POST", "ST_TASK", "LDLT TPP", GTG_YELLOW);
      addEntityValue("TA_LDLT_TPP", "ST_TASK", "LDLT TPP", GTG_BLUE);
      addEntityValue("TA_ASM_PRE", "ST_TASK", "Assembly Pre", GTG_TEAL);
      addEntityValue("TA_ASM_POST", "ST_TASK", "Assembly Post", GTG_MAUVE);
      addEntityValue("TA_MISC1", "ST_TASK", "Misc 1", GTG_KAKI);
      addEntityValue("TA_MISC2", "ST_TASK", "Misc 2", GTG_REDBLOOD);
      // GPU tasks
      addStateType("ST_GPU_TASK", "CT_GPU", "GPU exec");
      addEntityValue("GT_FACTOR", "ST_GPU_TASK", "Factor", GTG_RED);
      // Define events
      addEventType("EV_AGG_FAIL", "CT_THREAD", "Aggressive pivot fail");
      addEventType("EV_ALL_REGIONS", "CT_THREAD", "All regions subtree");
      // Initialise start time
      clock_gettime(CLOCK_REALTIME, &tstart);
#endif
   }

   /**
    * \brief Close trace file.
    */
   static
   void end(void) {
#if defined(PROFILE) && defined(HAVE_GTG)
      endTrace();
#endif
   }

   /**
    * \brief Return time since end of call to Profile::init().
    */
   static
   double now() {
#ifdef PROFILE
      struct timespec t;
      clock_gettime(CLOCK_REALTIME, &t);
      return tdiff(tstart, t);
#else
      return 0;
#endif
   }

private:
   /** \brief Convert thread index to character string */
   static
   char const* get_thread_name(int thread) {
      char const* thread_name[] = {
         "Thread0", "Thread1", "Thread2", "Thread3",
         "Thread4", "Thread5", "Thread6", "Thread7",
         "Thread8", "Thread9", "Thread10","Thread11",
         "Thread12", "Thread13", "Thread14", "Thread15",
         "Thread16", "Thread17", "Thread18", "Thread19",
         "Thread20", "Thread21", "Thread22", "Thread23",
         "Thread24", "Thread25", "Thread26", "Thread27",
         "Thread28", "Thread29", "Thread30", "Thread31"
      };
      return thread_name[thread];
   }

#ifdef PROFILE
   /** \brief Return difference in seconds between t1 and t2. */
   static
   double tdiff(struct timespec t1, struct timespec t2) {
      return (t2.tv_sec - t1.tv_sec) + 1e-9*(t2.tv_nsec - t1.tv_nsec);
   }
#endif

   /** \brief Return best guess at processor id. */
   static
   int guess_core() {
#ifdef HAVE_SCHED_GETCPU
      return sched_getcpu();
#else /* HAVE_SCHED_GETCPU */
      return omp::get_global_thread_num();
#endif /* HAVE_SCHED_GETCPU */
   }

#ifdef PROFILE
   static struct timespec tstart; //< The time at end of Profile::init().
#endif /* PROFILE */
};


}} /* namespace spral::ssids */
