#include <cosmictiger/direct.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/rand.hpp>

struct test_particle {
	std::array<fixed32, NDIM> x;
	gforce f;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & x;
		arc & f;
	}
};

#ifdef TEST_FORCE

std::vector<test_particle> direct_get_test_particles(int cnt) {
	particle_server pserv;
	const auto& parts = pserv.get_particle_set();
	std::vector<test_particle> testparts(cnt);
	std::set<int> used;
	for (int i = 0; i < cnt; i++) {
		int index;
		do {
			index = rand() % parts.size();
		} while (used.find(index) != used.end());
		used.insert(index);
		for (int dim = 0; dim < NDIM; dim++) {
			testparts[i].x[dim] = parts.pos(dim, index);
			testparts[i].f.f[dim] = parts.force(dim, index);
		}
		testparts[i].f.phi = parts.pot(index);
	}
	return testparts;
}

HPX_PLAIN_ACTION (direct_get_test_particles);
HPX_PLAIN_ACTION (cuda_direct);

void direct_force_test() {
	particle_server pserv;
	std::vector<int> counts(hpx_size());
	for (int i = 0; i < N_TEST_PARTS; i++) {
		array<fixed32, NDIM> x;
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = rand_fixed32();
		}
		counts[pserv.get_domain_bounds().find_proc(x)]++;
	}
	std::vector<hpx::future<std::vector<test_particle>>>futs1;
	for (int i = 0; i < hpx_size(); i++) {
		futs1.push_back(hpx::async<direct_get_test_particles_action>(hpx_localities()[i], counts[i]));
	}
	std::vector<test_particle> test_parts;
	for (auto& f : futs1) {
		auto tmp = f.get();
		test_parts.insert(test_parts.end(), tmp.begin(), tmp.end());
	}
	std::vector<std::array<fixed32, NDIM>> pts;
	for (int i = 0; i < test_parts.size(); i++) {
		pts.push_back(test_parts[i].x);
	}
	std::vector<hpx::future<std::vector<gforce>>>futs2;
	for (int i = 0; i < hpx_size(); i++) {
		futs2.push_back(hpx::async<cuda_direct_action>(hpx_localities()[i], pts));
	}

	std::vector<gforce> total(pts.size());
	for (int i = 0; i < pts.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			total[i].f[dim] = 0.0;
		}
		total[i].phi = 0.0;
	}
	for (auto& f : futs2) {
		auto tmp = f.get();
		for (int i = 0; i < pts.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				total[i].f[dim] += tmp[i].f[dim];
			}
			total[i].phi += tmp[i].phi;
		}
	}
	double gerr = 0.0, phierr = 0.0;
	double gerr_max = 0.0, phierr_max = 0.0;
	double gnorm = 0.0, phinorm = 0.0;
	for (int i = 0; i < pts.size(); i++) {
		double gnum = 0.0, gdir = 0.0;
		double phinum, phidir;
		for (int dim = 0; dim < NDIM; dim++) {
			gdir += sqr(total[i].f[dim]);
			gnum += sqr(test_parts[i].f.f[dim]);
		}
		gdir = std::sqrt(gdir);
		gnum = std::sqrt(gnum);
		phinum = test_parts[i].f.phi;
		phidir = total[i].phi - PHI0/global().opts.hsoft * global().opts.G * global().opts.M;
		gnorm += gdir;
		phinorm += std::abs(phidir);
		gerr += std::abs(gdir - gnum);
		phierr += std::abs(phidir - phinum);
		gerr_max = std::max(gerr_max, std::abs(gdir-gnum));
		phierr_max = std::max(phierr_max,std::abs(phidir-phinum));
		PRINT( "%e %e %e %e\n", gdir, gnum, phidir, phinum);
	}
	gerr /= gnorm;
	phierr /= phinorm;
	gerr_max /= gnorm;
	phierr_max /= phinorm;
	gerr_max *= pts.size();
	phierr_max *= pts.size();
	PRINT("gerr = %e\n", gerr);
	PRINT("gerr_max = %e\n\n", gerr_max);
	PRINT("phierr = %e\n", phierr);
	PRINT("phierr_max = %e\n", phierr_max);
}

#endif
