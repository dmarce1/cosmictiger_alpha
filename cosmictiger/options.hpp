/*
 * options.hpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_OPTIONS_HPP_
#define COSMICTIGER_OPTIONS_HPP_

#include <string>

struct options {
   std::string config;
   std::string test;
   size_t bucket_size;
   size_t nparts;
   float hsoft;
   float theta;
   template<class A>
   void serialize(A&& arc, unsigned) {
      arc & config;
      arc & test;
      arc & nparts;
   }
};

bool process_options(int argc, char *argv[], options &opts);                                          //

#endif /* COSMICTIGER_OPTIONS_HPP_ */
