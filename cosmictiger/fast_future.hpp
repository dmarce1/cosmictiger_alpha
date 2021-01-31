#pragma once

#include <cosmictiger/hpx.hpp>

template<class T>
class fast_future {
   T data;
   hpx::future<T> fut;
   bool has_data;
public:
   fast_future() {
      has_data = false;
   }
   fast_future(const fast_future<T>&) = delete;
   fast_future(fast_future<T>&&) = default;
   fast_future& operator=(const fast_future<T>&) = delete;
   fast_future& operator=(fast_future<T>&&) = default;
   fast_future& operator=(T&& data_) {
      has_data = true;
      data = std::move(data_);
      return *this;
   }
   inline fast_future(T&& data_) {
      has_data = true;
      data = std::move(data_);
   }
   fast_future& operator=(hpx::future<T> &&fut_) {
      has_data = false;
      fut = std::move(fut_);
      return *this;
   }
   fast_future(hpx::future<T> &&fut_) {
      has_data = false;
      fut = std::move(fut_);
   }
   void set_value(T&& data) {
      has_data = true;
      data = std::move(data);
   }
   bool valid() const {
      return( has_data || fut.valid() );
   }
   T get() {
      if( has_data ) {
         return std::move(data);
      } else {
         return fut.get();
      }
   }


};
