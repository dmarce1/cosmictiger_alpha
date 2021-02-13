/*
 * lockfree_queue.hpp
 *
 *  Created on: Feb 7, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_LOCKFREE_QUEUE_HPP_
#define COSMICTIGER_LOCKFREE_QUEUE_HPP_
#ifndef __CUDACC__
#include <cosmictiger/hpx.hpp>

#include <atomic>

template<class T, int N>
class lockfree_queue {
private:
   T data[N];
   size_t head;
   size_t tail;
   size_t count;
   mutable mutex_type mtx;
public:
   size_t size() const {
      std::lock_guard<mutex_type> lock(mtx);
      return count;
   }
   lockfree_queue() : head(0), tail(0), count(0) {
   }
   void push(T&& d) {
      std::lock_guard<mutex_type> lock(mtx);
      data[(tail++) % N] = std::move(d);
      count++;
      if( count >= N) {
         printf( "lock free queue overrun!\n");
         abort();
      }
   }
   void push(const T& d) {
      std::lock_guard<mutex_type> lock(mtx);
      data[(tail++) % N] = d;
      count++;
      if( count >= N) {
         printf( "lock free queue overrun!\n");
         abort();
      }
   }
   T pop() {
      std::lock_guard<mutex_type> lock(mtx);
      assert((int) count);
      count--;
      return std::move(data[head++ % N]);
   }
};

#endif

#endif /* COSMICTIGER_LOCKFREE_QUEUE_HPP_ */
