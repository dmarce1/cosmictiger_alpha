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
   std::atomic<size_t> head;
   std::atomic<size_t> tail;
   std::atomic<size_t> count;
public:
   size_t size() const {
      return count;
   }
   lockfree_queue() : head(0), tail(0), count(0) {
   }
   void push(T&& d) {
      data[(tail++) % N] = std::move(d);
      count++;
      if( count >= N) {
         PRINT( "lock free queue overrun!\n");
         abort();
      }
   }
   void push(const T& d) {
      data[(tail++) % N] = d;
      count++;
      if( count >= N) {
         PRINT( "lock free queue overrun!\n");
         abort();
      }
   }
   T pop() {
      assert((int) count);
      count--;
      return std::move(data[head++ % N]);
   }
};

#endif

#endif /* COSMICTIGER_LOCKFREE_QUEUE_HPP_ */
