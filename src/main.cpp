#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tests.hpp>

int hpx_main(int argc, char *argv[]) {
   options opts;
   if (process_options(argc, argv, opts)) {
      hpx_init();
      const auto cuda = cuda_init();
      global_init(opts, cuda);
      particle_set::create();
      if (opts.test != "") {
         test_run(opts.test);
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
