/*
 * options.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/options.hpp>
#include <cosmictiger/hpx.hpp>
#include <fstream>
#include <boost/program_options.hpp>

bool process_options(int argc, char *argv[], options &opts) {
   namespace po = boost::program_options;
   bool rc;
   po::options_description command_opts("options");

   command_opts.add_options()                                                                       //
   ("help", "produce help message")                                                                 //
   ("config", po::value < std::string > (&(opts.config))->default_value(""), "configuration file") //
   ("parts_dim", po::value < size_t > (&(opts.parts_dim))->default_value(16), "number of particles = parts_dim^3") //
   ("test", po::value < std::string > (&(opts.test))->default_value(""), "test problem") //
         ;

   boost::program_options::variables_map vm;
   po::store(po::parse_command_line(argc, argv, command_opts), vm);
   po::notify(vm);
   if (vm.count("help")) {
      std::cout << command_opts << "\n";
      rc = false;
   } else {
      if (!opts.config.empty()) {
         std::ifstream cfg_fs { vm["config"].as<std::string>() };
         if (cfg_fs) {
            po::store(po::parse_config_file(cfg_fs, command_opts), vm);
            rc = true;
         } else {
            printf("Configuration file %s not found!\n", opts.config.c_str());
            rc = false;
         }
      } else {
         rc = true;
      }
   }
   if (rc) {
      po::notify(vm);
      opts.nparts = opts.parts_dim * opts.parts_dim * opts.parts_dim;
      {
#define SHOW( opt ) std::cout << std::string( #opt ) << " = " << std::to_string(opts.opt) << '\n';
#define SHOW_STRING( opt ) std::cout << std::string( #opt ) << " = " << opts.opt << '\n';
         SHOW_STRING(config);
         SHOW_STRING(test);
         SHOW(nparts);
      }
   }
   opts.hsoft = 1.0 / pow(opts.nparts, 1.0 / 3.0) / 50.0;
   opts.theta = 0.4;
   return rc;
}

