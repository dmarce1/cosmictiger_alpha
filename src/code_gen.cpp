#include <cosmictiger/defs.hpp>
#include <cosmictiger/tensor.hpp>

int ntab = 0;

void indent() {
	ntab++;
}

void deindent() {
	ntab--;
}

template<class ...Args>
void tprint(const char* str, Args&&...args) {
	for (int i = 0; i < ntab; i++) {
		printf("\t");
	}
	printf(str, std::forward<Args>(args)...);
}

void compute_dx(int P) {
	array<int, NDIM> n;
	tprint("const real x0y0z0 = real(1);\n");
	tprint("const real& x1y0z0 = X[0];\n");
	tprint("const real& x0y1z0 = X[1];\n");
	tprint("const real& x0y0z1 = X[2];\n");
	for (int n0 = 2; n0 < P; n0++) {
		for (n[0] = 0; n[0] < n0; n[0]++) {
			for (n[1] = 0; n[1] < n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				array<int, NDIM> j = n;
				int j0 = n0;
				const int jmin = std::max(1, n0 / 2);
				while (j0 >= jmin) {
					for (int dim = 0; dim < NDIM; dim++ && j0 >= jmin) {
						if (j[dim] > 0) {
							j[dim]--;
							j0--;
						}
					}
				}
				array<int, NDIM> k;
				k = n - j;
				tprint("const real x%i%i%i = x%i%i%i * x%i%i%i;\n", n[0], n[1], n[2], k[0], k[1], k[2], j[0], j[1], j[2]);
			}
		}
	}
}

int acc(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	print("%s%i%i%i += %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int dec(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	print("%s%i%i%i -= %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int eqp(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	print("%s%i%i%i = %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 0;
}

int eqn(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	print("%s%i%i%i = -%s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int mul(std::string a, array<int, NDIM> n, double b, std::string c, array<int, NDIM> j) {
	print("%s%i%i%i = %e * %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], b, c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int fma(std::string a, array<int, NDIM> n, double b, std::string c, array<int, NDIM> j) {
	print("%s%i%i%i = fma(%e, %s%i%i%i, %s%i%i%i);\n", a.c_str(), n[0], n[1], n[2], b, c.c_str(), j[0], j[1], j[2],
			a.c_str(), n[0], n[1], n[2]);
	return 2;
}

bool close21(double a) {
	return std::abs(1.0 - a) < 1.0e-20;
}

template<int P>
void compute_detraceF(std::string iname, std::string oname) {
	array<int, NDIM> m;
	array<int, NDIM> k;
	array<int, NDIM> n;
	tensor_sym<int, P> first_use;
	first_use = 1;
	int flops = 0;
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
			for (n[2] = 0; n[2] < nzmax; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
					for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
						for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
							const int m0 = m[0] + m[1] + m[2];
							double num = double(n1pow(m0) * dfactorial(2 * n0 - 2 * m0 - 1) * vfactorial(n));
							double den = double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
							const double fnm = num / den;
							for (k[0] = 0; k[0] <= m0; k[0]++) {
								for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
									k[2] = m0 - k[0] - k[1];
									const auto p = n - (m) * 2 + (k) * 2;
									num = factorial(m0);
									den = vfactorial(k);
									const double number = fnm * num / den;
									if (first_use(n)) {
										if (close21(number)) {
											flops += eqp(oname, n, iname, p);
										} else if (close21(-number)) {
											flops += eqn(oname, n, iname, p);
										} else {
											flops += mul(oname, n, number, iname, p);
										}
										first_use(n) = 0;
									} else {
										if (close21(number)) {
											flops += acc(oname, n, iname, p);
										} else if (close21(-number)) {
											flops += dec(oname, n, iname, p);
										} else {
											flops += fma(oname, n, number, iname, p);
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	print("/* FLOPS = %i */\n", flops);
}

int main() {
	indent();
	compute_detraceF<5>("x", "A");
	deindent();
}
