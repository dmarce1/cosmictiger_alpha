#include <cosmictiger/particle_sets.hpp>

#include <silo.h>

void silo_out(particle_sets partsets, const char* filename) {
	printf("outputing in SILO format...\n");
	auto db = DBCreate(filename, DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
	const int npart_types = global().opts.sph ? 2 : 1;
	const std::string set_names[2] = {"cdm", "baryon"};
	for (int pi = 0; pi < npart_types; pi++) {
		auto& parts = *partsets.sets[pi];
		const auto size_ = parts.size();
		const auto mesh_name = (std::string("points_") + set_names[pi]);
		printf("positions\n");
		{
			std::vector<float> x(size_), y(size_), z(size_);
			for (int i = 0; i < size_; i++) {
				x[i] = parts.pos(0, i).to_float();
				y[i] = parts.pos(1, i).to_float();
				z[i] = parts.pos(2, i).to_float();
			}
			float* coords[NDIM] = { x.data(), y.data(), z.data() };
			DBPutPointmesh(db, mesh_name.c_str(), NDIM, coords, size_, DB_FLOAT, NULL);
		}
		printf("velocities\n");
		{
			std::vector<float> v(size_);
			for (int dim = 0; dim < NDIM; dim++) {
				for (int i = 0; i < size_; i++) {
					v[i] = parts.vel(dim, i);
				}
				std::string name = "v_" + set_names[pi] + "_";
				name.push_back(char('x' + dim));
				DBPutPointvar1(db, name.c_str(), mesh_name.c_str(), v.data(), size_, DB_FLOAT, NULL);
			}
		}
		printf("rungs\n");
		{
			std::vector<short> r(size_);
			for (int i = 0; i < size_; i++) {
				r[i] = parts.rung(i);
			}
			DBPutPointvar1(db, (std::string("rung_") + set_names[pi]).c_str(), mesh_name.c_str(), r.data(), size_, DB_SHORT, NULL);
		}
		printf("groups\n");
		{
			std::vector<long long> g(size_);
			for (int i = 0; i < size_; i++) {
				g[i] = parts.group(i);
			}
			DBPutPointvar1(db, (std::string("group_") + set_names[pi]).c_str(), mesh_name.c_str(), g.data(), size_, DB_LONG_LONG, NULL);
		}
#ifdef TEST_FORCE
		printf( "forces\n");
		{
			std::vector<float> g(size_);
			for (int dim = 0; dim < NDIM; dim++) {
				for (int i = 0; i < size_; i++) {
					g[i] = parts.force(dim, i);
				}
				std::string name = "g_" + set_names[pi] + "_";
				name.push_back(char('x' + dim));
				DBPutPointvar1(db, name.c_str(), mesh_name.c_str(), g.data(), size_, DB_FLOAT, NULL);
			}
		}
		printf( "potential\n");
		{
			std::vector<float> p(size_);
			for (int i = 0; i < size_; i++) {
				p[i] = parts.pot(i);
			}
			DBPutPointvar1(db, (std::string("phi_") + set_names[pi]).c_str(), mesh_name.c_str(), p.data(), size_, DB_FLOAT, NULL);
		}
#endif
	}
	DBClose(db);

}

