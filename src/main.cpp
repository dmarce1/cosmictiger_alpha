#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/options.hpp>

int hpx_main(int argc, char *argv[]) {
   if (process_options(argc, argv)) {
      hpx_init();
      particle_set::create();

      particle_set::destroy();
   }
   return hpx::finalize();
}

int main(int argc, char *argv[]) {
   std::vector < std::string > cfg = { "hpx.commandline.allow_unknown=1" };
   hpx::init(argc, argv, cfg);
}
