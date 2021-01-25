#pragma once

#include <time.h>

class timer {
   double time;
   double this_time;
   constexpr static double cpsinv = 1.0 / CLOCKS_PER_SEC;
public:
   inline timer() {
      time = 0.0;
   }
   inline void stop() {
      time += clock() * cpsinv - this_time;
   }
   inline void start() {
      this_time = clock() * cpsinv;
   }
   inline double read() {
      return time;
   }
};
