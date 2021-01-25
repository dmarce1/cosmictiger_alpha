#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/global.hpp>

int hpx_main(int argc, char *argv[]) {
   options opts;
   if (process_options(argc, argv, opts)) {
      hpx_init();
      global_init(opts);
      particle_set::create();
      if (opts.test != "") {

      } else {

      }

      particle_set::destroy();
   }

   return hpx::finalize();
}

int main(int argc, char *argv[]) {
   std::vector < std::string > cfg = { "hpx.commandline.allow_unknown=1" };
   hpx::init(argc, argv, cfg);
}
