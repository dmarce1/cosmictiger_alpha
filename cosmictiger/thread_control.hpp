#pragma once

#include <cosmictiger/hpx.hpp>

#include <stack>

class thread_control {
   struct stack_type {
      std::shared_ptr<hpx::lcos::local::promise<void>> promise;
      int thread_cnt;
   };
   int thread_cnt;
   int priority;
   bool released;
   static std::stack<stack_type> stack;
   static std::stack<stack_type> priority_stack;
   static bool initialized;
   static hpx::lcos::local::mutex mtx;
   static std::atomic<int> avail;
   static std::atomic<size_t> total_threads;
   static std::atomic<size_t> total_priority;
public:
   static void init();
   static void check_stacks();

   thread_control();
//   thread_control(const thread_control&) = delete;
//   thread_control(thread_control&&) = default;
//   thread_control& operator=(const thread_control&) = delete;
   thread_control& operator=(thread_control&&);
    thread_control(int thread_cnt, int priority);
   ~thread_control();
   void release();
   void acquire();
   bool try_acquire();

};
