#include <cosmictiger/thread_control.hpp>
#include <cosmictiger/hpx.hpp>

std::stack<thread_control::stack_type> thread_control::stack;
std::stack<thread_control::stack_type> thread_control::priority_stack;
bool thread_control::initialized = false;
hpx::lcos::local::mutex thread_control::mtx;
std::atomic<int> thread_control::avail(0);
std::atomic<size_t> thread_control::total_threads;
std::atomic<size_t> thread_control::total_priority;

void thread_control::init() {
   if (!initialized) {
      std::lock_guard < hpx::lcos::local::mutex > lock(mtx);
      initialized = true;
      avail = hpx::thread::hardware_concurrency();
   }
}

void thread_control::check_stacks() {
   stack_type entry;
   {
      std::lock_guard < hpx::lcos::local::mutex > lock(mtx);
      bool found_priority = false;
      if (!priority_stack.empty()) {
         const auto top = priority_stack.top();
         if (avail >= top.thread_cnt) {
            avail -= top.thread_cnt;
            entry = std::move(top);
            ;
            priority_stack.pop();
            found_priority = true;
         }
      }
      if (!stack.empty()) {
         const auto top = stack.top();
         if (avail >= top.thread_cnt) {
            avail -= top.thread_cnt;
            entry = std::move(top);

            stack.pop();
         }
      }
   }
   if (entry.promise) {
      entry.promise->set_value();
   }
}

thread_control::thread_control(int number, int p) {
   init();
   total_threads++;
   total_priority += priority;
   thread_cnt = number;
   priority = p;
   acquire();
}

void thread_control::release() {
   assert(!released);
   released = true;
   avail += thread_cnt;
   check_stacks();
}

void thread_control::acquire() {
   assert(!released);
   released = false;
   hpx::future<void> fut;
   std::shared_ptr<hpx::lcos::local::promise<void>> promise;
   bool thread_avail = false;
   {
      std::lock_guard < hpx::lcos::local::mutex > lock(mtx);
      const double avg_priority = total_priority / total_threads;
      if (((priority > avg_priority || priority_stack.empty()) && avail >= thread_cnt)) {
         avail -= thread_cnt;
         thread_avail = true;
      } else {
         promise = std::make_shared<hpx::lcos::local::promise<void>>();
         stack_type entry;
         entry.promise = promise;
         entry.thread_cnt = thread_cnt;
         if (priority > avg_priority) {
            priority_stack.push(std::move(entry));
         } else {
            stack.push(std::move(entry));
         }
      }
   }
   if (thread_avail) {
      check_stacks();
   }
   if (!thread_avail) {
      fut = promise->get_future();
      fut.get();
      check_stacks();
   }
}

thread_control::~thread_control() {
   if (!released) {
      avail += thread_cnt;
   }
   check_stacks();
}
