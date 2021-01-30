#include <cosmictiger/thread_control.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/defs.hpp>

std::stack<thread_control::stack_type> thread_control::stack;
std::stack<thread_control::stack_type> thread_control::priority_stack;
bool thread_control::initialized = false;
hpx::lcos::local::spinlock thread_control::mtx;
std::atomic<int> thread_control::avail(0);
int thread_control::max_avail;
std::atomic<size_t> thread_control::total_threads;
std::atomic<size_t> thread_control::total_priority;

void thread_control::init() {
   if (!initialized) {
      std::lock_guard < hpx::lcos::local::spinlock > lock(mtx);
      initialized = true;
      max_avail = hpx::thread::hardware_concurrency() * OVERSUBSCRIPTION;
      avail = max_avail;
   }
}

void thread_control::check_stacks() {
   assert(avail >= 0);
   assert(avail <= max_avail);
   stack_type entry;
   {
      std::lock_guard < hpx::lcos::local::spinlock > lock(mtx);
      bool found_priority = false;
      if (!priority_stack.empty()) {
         const auto top = priority_stack.top();
         if (avail >= top.thread_cnt) {
            avail -= top.thread_cnt;
            entry = std::move(top);
            priority_stack.pop();
            found_priority = true;
         }
      }
      if (!stack.empty() && !found_priority) {
         const auto top = stack.top();
         if (avail >= top.thread_cnt) {
            avail -= top.thread_cnt;
            entry = std::move(top);
            stack.pop();
         }
      }
   }
   if (entry.promise) {
      *entry.active = true;
      entry.promise->set_value();
   }
   assert(avail >= 0);
   assert(avail <= max_avail);

}
//
//thread_control::thread_control() {
//   assert(avail >= 0);
//   assert(avail <= max_avail);
//   thread_cnt = 1;
//   priority = 0;
//   active = std::make_shared<bool>(false);
//}

thread_control& thread_control::operator=(thread_control &&other) {
   assert(avail >= 0);
   assert(avail <= max_avail);
   priority = other.priority;
   thread_cnt = other.thread_cnt;
   active = other.active;
   return *this;
}

thread_control::thread_control(int number, int p) {
   init();
   assert(avail >= 0);
   assert(avail <= max_avail);
   active = std::make_shared<bool>(false);
   total_threads++;
   total_priority += priority;
   thread_cnt = number;
   priority = p;
}

void thread_control::release() {
   assert(*active);
   *active = false;
   avail += thread_cnt;
   check_stacks();
   assert(avail <= max_avail);
   assert(avail >= 0);
}

void thread_control::release_some(int n) {
   avail += n;
   thread_cnt -= n;
   check_stacks();
   assert(thread_cnt > 0);
   assert(avail >= 0);
   assert(avail <= max_avail);
}

void thread_control::acquire() {
   if (!try_acquire()) {
      auto promise = std::make_shared<hpx::lcos::local::promise<void>>();
      const double avg_priority = total_priority / total_threads;
      std::lock_guard < hpx::lcos::local::spinlock > lock(mtx);
      stack_type entry;
      entry.promise = promise;
      entry.thread_cnt = thread_cnt;
      entry.active = active;
      if (priority > avg_priority) {
         priority_stack.push(std::move(entry));
      } else {
         stack.push(std::move(entry));
      }
   }
//   check_stacks();
}

bool thread_control::try_acquire() {
//   assert(!released);
 //  printf("%i %i\n", (int) avail, max_avail);
    assert(avail <= max_avail);
   hpx::future<void> fut;
   std::shared_ptr<hpx::lcos::local::promise<void>> promise;
   bool thread_avail = false;
   {
      std::lock_guard < hpx::lcos::local::spinlock > lock(mtx);
      const double avg_priority = total_priority / total_threads;
      if (((priority > avg_priority || priority_stack.empty()) && avail >= thread_cnt)) {
         avail -= thread_cnt;
         *active = true;
         thread_avail = true;
      }
   }
 //  check_stacks();
   return thread_avail;
}

thread_control::~thread_control() {
   if (active && *active) {
      avail += thread_cnt;
      check_stacks();
   }

   assert(avail >= 0);
   assert(avail <= max_avail);
   assert(thread_cnt > 0);

}
