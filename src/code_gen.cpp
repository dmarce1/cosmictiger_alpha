#include <cosmictiger/defs.hpp>
#define CODE_GEN_CPP
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

int compute_dx(int P, const char* name = "X") {
	array<int, NDIM> n;
	tprint("const T x000 = T(1);\n");
	tprint("const T& x100 = %s[0];\n", name);
	tprint("const T& x010 = %s[1];\n", name);
	tprint("const T& x001 = %s[2];\n", name);
	int flops = 0;
	for (int n0 = 2; n0 < P; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				array<int, NDIM> j = n;
				int j0 = n0;
				const int jmin = std::max(1, n0 / 2);
				while (j0 > jmin) {
					for (int dim = 0; dim < NDIM && j0 > jmin; dim++) {
						if (j[dim] > 0) {
							j[dim]--;
							j0--;
						}
					}
				}
				array<int, NDIM> k;
				k = n - j;
				tprint("const T x%i%i%i = x%i%i%i * x%i%i%i;\n", n[0], n[1], n[2], k[0], k[1], k[2], j[0], j[1], j[2]);
				flops++;
			}
		}
	}
	return flops;
}

int trless_index(int l, int m, int n, int Q) {
	return (l + m) * ((l + m) + 1) / 2 + (m) + (Q * (Q + 1) / 2) * (n == 1) + (Q * Q) * (n == 2);
}

int sym_index(int l, int m, int n) {
	return (l + m + n) * (l + m + n + 1) * ((l + m + n) + 2) / 6 + (m + n) * ((m + n) + 1) / 2 + n;
}

int compute_dx_tensor(int P, const char* name = "X") {
	array<int, NDIM> n;
	tprint("tensor_sym<T,%i> dx;\n", P);
	tprint("dx[0] = T(1);\n");
	tprint("dx[%i] = %s[0];\n", sym_index(1, 0, 0), name);
	tprint("dx[%i] = %s[1];\n", sym_index(0, 1, 0), name);
	tprint("dx[%i] = %s[2];\n", sym_index(0, 0, 1), name);
	int flops = 0;
	for (int n0 = 2; n0 < P; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				array<int, NDIM> j = n;
				int j0 = n0;
				const int jmin = std::max(1, n0 / 2);
				while (j0 > jmin) {
					for (int dim = 0; dim < NDIM && j0 > jmin; dim++) {
						if (j[dim] > 0) {
							j[dim]--;
							j0--;
						}
					}
				}
				array<int, NDIM> k;
				k = n - j;
				tprint("dx[%i]= dx[%i] * dx[%i];\n", sym_index(n[0], n[1], n[2]), sym_index(k[0], k[1], k[2]),
						sym_index(j[0], j[1], j[2]));
				flops++;
			}
		}
	}
	return flops;
}

int acc(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i += %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int dec(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i -= %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int eqp(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i = %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 0;
}

int eqn(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i = -%s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int mul(std::string a, array<int, NDIM> n, double b, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i = T(%.8e) * %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], b, c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int fma(std::string a, array<int, NDIM> n, double b, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i = fmaf(T(%.8e), %s%i%i%i, %s%i%i%i);\n", a.c_str(), n[0], n[1], n[2], b, c.c_str(), j[0], j[1],
			j[2], a.c_str(), n[0], n[1], n[2]);
	return 2;
}

bool close21(double a) {
	return std::abs(1.0 - a) < 1.0e-20;
}

#define ASPRINTF(...) if( asprintf(__VA_ARGS__) == 0 ) printf( "asprintf failed\n")

template<int P>
int compute_detrace(std::string iname, std::string oname, char type = 'f') {
	array<int, NDIM> m;
	array<int, NDIM> k;
	array<int, NDIM> n;

	struct entry_type {
		array<int, NDIM> src;
		array<int, NDIM> dest;
		double factor;
	};
	std::vector<entry_type> entries;
	int flops = 0;

	std::vector<std::string> asn;
	std::vector<std::string> op;
	for (int m0 = 1; m0 <= P / 2; m0++) {
		array<int, NDIM> j;
		const int Q = P - 2 * m0;
		for (j[0] = 0; j[0] < Q; j[0]++) {
			for (j[1] = 0; j[1] < Q - j[0]; j[1]++) {
				for (j[2] = 0; j[2] < Q - j[0] - j[1]; j[2]++) {
					const int j0 = j[0] + j[1] + j[2];
					int n0 = j0 + 2 * m0;
					bool first = true;
					array<int, NDIM> k;
					for (k[0] = 0; k[0] <= m0; k[0]++) {
						for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
							k[2] = m0 - k[0] - k[1];
							const double num = factorial(m0);
							const double den = vfactorial(k);
							const double factor = num / den;
							const auto p = j + k * 2;
							char* str;
							if (first) {
								if (close21(factor)) {
									ASPRINTF(&str, "T %s_%i_%i_%i%i%i = %s%i%i%i;\n", iname.c_str(), n0, m0, j[0], j[1], j[2],
											iname.c_str(), p[0], p[1], p[2]);
								} else if (close21(-factor)) {
									ASPRINTF(&str, "T %s_%i_%i_%i%i%i = -%s%i%i%i;\n", iname.c_str(), n0, m0, j[0], j[1], j[2],
											iname.c_str(), p[0], p[1], p[2]);
									flops++;
								} else {
									ASPRINTF(&str, "T %s_%i_%i_%i%i%i = T(%.8e) * %s%i%i%i;\n", iname.c_str(), n0, m0, j[0], j[1],
											j[2], factor, iname.c_str(), p[0], p[1], p[2]);
									flops++;
								}
								asn.push_back(str);
								first = false;
								free(str);
							} else {
								if (close21(factor)) {
									ASPRINTF(&str, "%s_%i_%i_%i%i%i += %s%i%i%i;\n", iname.c_str(), n0, m0, j[0], j[1], j[2],
											iname.c_str(), p[0], p[1], p[2]);
									flops++;
								} else if (close21(-factor)) {
									ASPRINTF(&str, "%s_%i_%i_%i%i%i -= %s%i%i%i;\n", iname.c_str(), n0, m0, j[0], j[1], j[2],
											iname.c_str(), p[0], p[1], p[2]);
									flops++;
								} else {
									ASPRINTF(&str, "%s_%i_%i_%i%i%i = fmaf(T(%.8e), %s%i%i%i, %s_%i_%i_%i%i%i);\n", iname.c_str(),
											n0, m0, j[0], j[1], j[2], factor, iname.c_str(), p[0], p[1], p[2], iname.c_str(), n0,
											m0, j[0], j[1], j[2]);
									flops += 2;
								}
								op.push_back(str);
								free(str);
							}
						}
					}
				}
			}
		}
	}
	int maxop = (op.size() + 1) / 2;
	for (int i = 0; i < asn.size(); i++) {
		tprint("%s", asn[i].c_str());
	}
	for (int i = 0; i < maxop; i++) {
		tprint("%s", op[i].c_str());
		if (i + maxop < op.size()) {
			tprint("%s", op[i + maxop].c_str());
		}
	}
	op.resize(0);
	asn.resize(0);
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
			for (n[2] = 0; n[2] < nzmax; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				bool first = true;
				for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
					for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
						for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
							const int m0 = m[0] + m[1] + m[2];
							if (type == 'd' && ((n0 == 2 && (n[0] == 2 || n[1] == 2 || n[2] == 2)) && m0 == 1)) {
								continue;
							}
							double num = double(n1pow(m0) * dfactorial(2 * n0 - 2 * m0 - 1) * vfactorial(n));
							double den = double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
							double factor = num / den;
							if (type == 'd') {
								factor *= 1.0 / dfactorial(2 * n0 - 1);
							}
							const auto p = n - m * 2;
							char* str;
							if (first) {
								if (m0 > 0) {
									if (close21(factor)) {
										ASPRINTF(&str, "%s%i%i%i = %s_%i_%i_%i%i%i;\n", oname.c_str(), n[0], n[1], n[2],
												iname.c_str(), n0, m0, p[0], p[1], p[2]);
									} else if (close21(-factor)) {
										ASPRINTF(&str, "%s%i%i%i = -%s_%i_%i_%i%i%i;\n", oname.c_str(), n[0], n[1], n[2],
												iname.c_str(), n0, m0, p[0], p[1], p[2]);
										flops++;
									} else {
										ASPRINTF(&str, "%s%i%i%i = T(%.8e) * %s_%i_%i_%i%i%i;\n", oname.c_str(), n[0], n[1], n[2],
												factor, iname.c_str(), n0, m0, p[0], p[1], p[2]);
										flops++;
									}
								} else {
									if (close21(factor)) {
										ASPRINTF(&str, "%s%i%i%i = %s%i%i%i;\n", oname.c_str(), n[0], n[1], n[2], iname.c_str(),
												p[0], p[1], p[2]);
									} else if (close21(-factor)) {
										ASPRINTF(&str, "%s%i%i%i = -%s%i%i%i;\n", oname.c_str(), n[0], n[1], n[2], iname.c_str(),
												p[0], p[1], p[2]);
										flops++;
									} else {
										ASPRINTF(&str, "%s%i%i%i = T(%.8e) * %s%i%i%i;\n", oname.c_str(), n[0], n[1], n[2], factor,
												iname.c_str(), p[0], p[1], p[2]);
										flops++;
									}
								}
								asn.push_back(str);
								free(str);
								first = false;
							} else {
								if (close21(factor)) {
									if (m0 > 0) {
										ASPRINTF(&str, "%s%i%i%i += %s_%i_%i_%i%i%i;\n", oname.c_str(), n[0], n[1], n[2],
												iname.c_str(), n0, m0, p[0], p[1], p[2]);
										flops++;
									} else {
										ASPRINTF(&str, "%s%i%i%i += %s%i%i%i;\n", oname.c_str(), n[0], n[1], n[2], iname.c_str(),
												p[0], p[1], p[2]);
										flops++;
									}
								} else if (close21(-factor)) {
									if (m0 > 0) {
										ASPRINTF(&str, "%s%i%i%i -= %s_%i_%i_%i%i%i;\n", oname.c_str(), n[0], n[1], n[2],
												iname.c_str(), n0, m0, p[0], p[1], p[2]);
										flops++;
									} else {
										ASPRINTF(&str, "%s%i%i%i -= %s%i%i%i;\n", oname.c_str(), n[0], n[1], n[2], iname.c_str(),
												p[0], p[1], p[2]);
										flops++;
									}
								} else {
									if (m0 > 0) {
										ASPRINTF(&str, "%s%i%i%i = fmaf(T(%.8e), %s_%i_%i_%i%i%i, %s%i%i%i);\n", oname.c_str(), n[0],
												n[1], n[2], factor, iname.c_str(), n0, m0, p[0], p[1], p[2], oname.c_str(), n[0],
												n[1], n[2]);
										flops += 2;
									} else {
										ASPRINTF(&str, "%s%i%i%i = fmaf(T(%.8e), %s%i%i%i, %s%i%i%i);\n", oname.c_str(), n[0], n[1],
												n[2], factor, iname.c_str(), p[0], p[1], p[2], oname.c_str(), n[0], n[1], n[2]);
										flops += 2;
									}
								}
								op.push_back(str);
								free(str);
							}
						}
					}
				}
			}
		}
	}
	maxop = (op.size() + 1) / 2;
	for (int i = 0; i < asn.size(); i++) {
		tprint("%s", asn[i].c_str());
	}
	for (int i = 0; i < maxop; i++) {
		tprint("%s", op[i].c_str());
		if (i + maxop < op.size()) {
			tprint("%s", op[i + maxop].c_str());
		}
	}

	return flops;
}

template<int P>
int compute_detrace_ewald(std::string iname, std::string oname) {
	array<int, NDIM> m;
	array<int, NDIM> k;
	array<int, NDIM> n;

	struct entry_type {
		array<int, NDIM> src;
		array<int, NDIM> dest;
		double factor;
	};
	std::vector<entry_type> entries;
	int flops = 0;

	std::vector<std::string> asn;
	std::vector<std::string> op;
	for (int m0 = 1; m0 <= P / 2; m0++) {
		array<int, NDIM> j;
		const int Q = P - 2 * m0;
		for (j[0] = 0; j[0] < Q; j[0]++) {
			for (j[1] = 0; j[1] < Q - j[0]; j[1]++) {
				for (j[2] = 0; j[2] < Q - j[0] - j[1]; j[2]++) {
					const int j0 = j[0] + j[1] + j[2];
					int n0 = j0 + 2 * m0;
					bool first = true;
					array<int, NDIM> k;
					for (k[0] = 0; k[0] <= m0; k[0]++) {
						for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
							k[2] = m0 - k[0] - k[1];
							const double num = factorial(m0);
							const double den = vfactorial(k);
							const double factor = num / den;
							const auto p = j + k * 2;
							char* str;
							if (first) {
								if (close21(factor)) {
									ASPRINTF(&str, "T %s_%i_%i_%i%i%i = %s%i%i%i;\n", iname.c_str(), n0, m0, j[0], j[1], j[2],
											iname.c_str(), p[0], p[1], p[2]);
								} else if (close21(-factor)) {
									ASPRINTF(&str, "T %s_%i_%i_%i%i%i = -%s%i%i%i;\n", iname.c_str(), n0, m0, j[0], j[1], j[2],
											iname.c_str(), p[0], p[1], p[2]);
									flops += 1;
								} else {
									ASPRINTF(&str, "T %s_%i_%i_%i%i%i = T(%.8e) * %s%i%i%i;\n", iname.c_str(), n0, m0, j[0], j[1],
											j[2], factor, iname.c_str(), p[0], p[1], p[2]);
									flops++;
								}
								asn.push_back(str);
								first = false;
								free(str);
							} else {
								if (close21(factor)) {
									ASPRINTF(&str, "%s_%i_%i_%i%i%i += %s%i%i%i;\n", iname.c_str(), n0, m0, j[0], j[1], j[2],
											iname.c_str(), p[0], p[1], p[2]);
									flops += 1;
								} else if (close21(-factor)) {
									ASPRINTF(&str, "%s_%i_%i_%i%i%i -= %s%i%i%i;\n", iname.c_str(), n0, m0, j[0], j[1], j[2],
											iname.c_str(), p[0], p[1], p[2]);
									flops += 1;
								} else {
									ASPRINTF(&str, "%s_%i_%i_%i%i%i = fmaf(T(%.8e), %s%i%i%i, %s_%i_%i_%i%i%i);\n", iname.c_str(),
											n0, m0, j[0], j[1], j[2], factor, iname.c_str(), p[0], p[1], p[2], iname.c_str(), n0,
											m0, j[0], j[1], j[2]);
									flops += 2;
								}
								op.push_back(str);
								free(str);
							}
						}
					}
				}
			}
		}
	}
	int maxop = (op.size() + 1) / 2;
	for (int i = 0; i < asn.size(); i++) {
		tprint("%s", asn[i].c_str());
	}
	for (int i = 0; i < maxop; i++) {
		tprint("%s", op[i].c_str());
		if (i + maxop < op.size()) {
			tprint("%s", op[i + maxop].c_str());
		}
	}
	for (int m0 = 1; m0 <= P / 2; m0++) {
		array<int, NDIM> j;
		const int Q = P - 2 * m0;
		for (j[0] = 0; j[0] < Q; j[0]++) {
			for (j[1] = 0; j[1] < Q - j[0]; j[1]++) {
				for (j[2] = 0; j[2] < Q - j[0] - j[1]; j[2]++) {
					const int j0 = j[0] + j[1] + j[2];
					int n0 = j0 + 2 * m0;
					tprint("%s_%i_%i_%i%i%i *= Drinvpow_%i_%i;\n", iname.c_str(), n0, m0, j[0], j[1], j[2], n0 - m0, m0);
					flops++;
				}
			}
		}
	}

	op.resize(0);
	asn.resize(0);
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			for (n[2] = 0; n[2] < P - n[0] - n[1]; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				bool first = true;
				for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
					for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
						for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
							const int m0 = m[0] + m[1] + m[2];
							double num = double(vfactorial(n));
							double den = double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
							double factor = num / den;
							const auto p = n - m * 2;
							char* str;
							if (m0 > 0) {
								if (close21(factor)) {
									ASPRINTF(&str, "%s[%i] += %s_%i_%i_%i%i%i;\n", oname.c_str(), sym_index(n[0], n[1], n[2]),
											iname.c_str(), n0, m0, p[0], p[1], p[2]);
									flops += 1;
								} else if (close21(-factor)) {
									ASPRINTF(&str, "%s[%i] -= %s_%i_%i_%i%i%i;\n", oname.c_str(), sym_index(n[0], n[1], n[2]),
											iname.c_str(), n0, m0, p[0], p[1], p[2]);
									flops += 1;
								} else {
									ASPRINTF(&str, "%s[%i] = fmaf(T(%.8e), %s_%i_%i_%i%i%i, %s[%i]);\n", oname.c_str(),
											sym_index(n[0], n[1], n[2]), factor, iname.c_str(), n0, m0, p[0], p[1], p[2],
											oname.c_str(), sym_index(n[0], n[1], n[2]));
									flops += 2;
								}
							} else {
								if (close21(factor)) {
									ASPRINTF(&str, "%s[%i] = fmaf(%s%i%i%i, Drinvpow_%i_%i, %s[%i]);\n", oname.c_str(),
											sym_index(n[0], n[1], n[2]), iname.c_str(), p[0], p[1], p[2], n0 - m0, m0,
											oname.c_str(), sym_index(n[0], n[1], n[2]));
									flops += 2;
								} else if (close21(-factor)) {
									ASPRINTF(&str, "%s[%i] -= %s%i%i%i * Drinvpow_%i_%i;\n", oname.c_str(),
											sym_index(n[0], n[1], n[2]), iname.c_str(), p[0], p[1], p[2], n0 - m0, m0);
									flops += 2;
								} else {
									ASPRINTF(&str, "%s[%i] = fmaf(T(%.8e), %s%i%i%i*Drinvpow_%i_%i, %s[%i]);\n", oname.c_str(),
											sym_index(n[0], n[1], n[2]), factor, iname.c_str(), p[0], p[1], p[2], n0 - m0, m0,
											oname.c_str(), sym_index(n[0], n[1], n[2]));

									flops += 3;
								}
							}
							op.push_back(str);
							free(str);
						}
					}
				}
			}
		}
	}
	std::sort(op.begin(), op.end(), [](std::string a, std::string b) {
		return atoi(a.c_str()+6) < atoi(b.c_str()+6);
	});
	maxop = (op.size() + 1) / 2;
	for (int i = 0; i < maxop; i++) {
		tprint("%s", op[i].c_str());
		if (i + maxop < op.size()) {
			tprint("%s", op[i + maxop].c_str());
		}
	}

	return flops;
}

#define P ORDER
template<int Q>
void const_ref_compute(int sign, tensor_trless_sym<int, Q>& counts, tensor_trless_sym<int, Q>& signs,
		array<int, NDIM> n) {
	int flops = 0;
	if (n[2] >= 2 && !(n[0] == 0 && n[1] == 0 && n[2] == 2)) {
		auto n1 = n;
		auto n2 = n;
		n1[2] -= 2;
		n2[2] -= 2;
		n1[0] += 2;
		n2[1] += 2;
		const_ref_compute(-sign, counts, signs, n1);
		const_ref_compute(-sign, counts, signs, n2);
	} else {
		counts(n)++;signs
		(n) = sign;
	}
}

template<int Q>
int print_const_ref(std::string name, std::string& cmd, const tensor_trless_sym<int, Q>& counts,
		const tensor_trless_sym<int, Q>& signs, array<int, NDIM> n) {
	int flops = 0;
	array<int, NDIM> k;
	array<int, NDIM> last_index;
	for (k[0] = 0; k[0] < Q; k[0]++) {
		for (k[1] = 0; k[1] < Q - k[0]; k[1]++) {
			for (k[2] = 0; k[2] < Q - k[0] - k[1]; k[2]++) {
				if (k[2] < 2 || (k[0] == 0 && k[1] == 0 && k[2] == 2)) {
					if (counts(k)) {
						last_index = k;
					}
				}
			}
		}
	}
	cmd += signs(last_index) == 1 ? "+" : "-";
	if (signs(last_index) != 1) {
		flops++;
	}
	int opened = 0;
	bool fma = false;
	for (int l = 0; l < Q * Q + 1; l++) {
		if (counts[l]) {
			if (opened && !fma) {
				cmd += "+";
			} else if (fma) {
				cmd += ",";
			}
			flops++;
			opened++;
			if (counts[l] == 1) {
				cmd += "(";
				cmd += name;
				cmd += "[";
				cmd += std::to_string(l);
				cmd += "]";
			} else {
				flops++;
				if (n[0] != last_index[0] || n[1] != last_index[1] || n[2] != last_index[2]) {
					fma = true;
				} else {
					fma = false;
				}
				if (fma) {
					cmd += "fmaf(";
				} else {
					cmd += "(";
				}
				cmd += "T(";
				cmd += std::to_string(counts[l]);
				if (fma) {
					cmd += "),";
				} else {
					cmd += ")*";
				}
				cmd += name;
				cmd += "[";
				cmd += std::to_string(l);
				cmd += "]";
			}
		}
	}
	for (int l = 0; l < opened; l++) {
		cmd += ")";
	}
	return flops;
}

template<int Q>
int const_reference_trless(std::string name) {
	array<int, NDIM> n;
	int flops = 0;
	for (n[0] = 0; n[0] < Q; n[0]++) {
		for (n[1] = 0; n[1] < Q - n[0]; n[1]++) {
			for (n[2] = 0; n[2] < Q - n[0] - n[1]; n[2]++) {
				if (!(n[2] >= 2 && !(n[0] == 0 && n[1] == 0 && n[2] == 2))) {
					tprint("const T& %s%i%i%i = ", name.c_str(), n[0], n[1], n[2]);
				} else {
					tprint("const T %s%i%i%i = ", name.c_str(), n[0], n[1], n[2]);
				}
				tensor_trless_sym<int, Q> counts;
				tensor_trless_sym<int, Q> signs;
				counts = 0;
				signs = 1;
				const_ref_compute(+1, counts, signs, n);
				std::string cmd;
				flops += print_const_ref(name, cmd, counts, signs, n);
				if (cmd[0] == '+') {
					flops--;
					cmd[0] = ' ';
				}
				printf("%s;\n", cmd.c_str());
			}
		}
	}
	return flops;
}

template<int Q>
int const_reference_trless_tensor(std::string name, std::string oname) {
	array<int, NDIM> n;
	int flops = 0;
	for (n[0] = 0; n[0] < Q; n[0]++) {
		for (n[1] = 0; n[1] < Q - n[0]; n[1]++) {
			for (n[2] = 0; n[2] < Q - n[0] - n[1]; n[2]++) {
				tprint("%s[%i] = ", oname.c_str(), sym_index(n[0], n[1], n[2]));
				tensor_trless_sym<int, Q> counts;
				tensor_trless_sym<int, Q> signs;
				counts = 0;
				signs = 1;
				const_ref_compute(+1, counts, signs, n);
				std::string cmd;
				flops += print_const_ref(name, cmd, counts, signs, n);
				if (cmd[0] == '+') {
					flops--;
					cmd[0] = ' ';
				}
				printf("%s;\n", cmd.c_str());
			}
		}
	}
	return flops;
}

void reference_trless(std::string name, int Q) {
	for (int l = 0; l < Q; l++) {
		for (int m = 0; m < Q - l; m++) {
			for (int n = 0; n < Q - l - m; n++) {
				if (n > 1 && !(l == 0 && m == 0 && n == 2)) {
					continue;
				}
				const int index = trless_index(l, m, n, Q);
				tprint("T& %s%i%i%i = %s[%i];\n", name.c_str(), l, m, n, name.c_str(), index);
			}
		}
	}
}

void reference_sym(std::string name, int Q) {
	for (int l = 0; l < Q; l++) {
		for (int m = 0; m < Q - l; m++) {
			for (int n = 0; n < Q - l - m; n++) {
				const int index = sym_index(l, m, n);
				tprint("const T& %s%i%i%i = %s[%i];\n", name.c_str(), l, m, n, name.c_str(), index);
			}
		}
	}
}

template<int Q>
void do_expansion(bool two) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT\n");
	if (two) {
		tprint(
				"tensor_trless_sym<T, %i> expansion_translate2(const tensor_trless_sym<T, %i>& La, const array<T, NDIM>& X, bool do_phi) {\n",
				Q, P);
	} else {
		tprint(
				"tensor_trless_sym<T, %i> expansion_translate(const tensor_trless_sym<T, %i>& La, const array<T, NDIM>& X, bool do_phi) {\n",
				Q, P);

	}
	indent();
	tprint("tensor_trless_sym<T, %i> Lb;\n//", Q);
	flops += compute_dx(P);
	array<int, NDIM> n;
	array<int, NDIM> k;
	flops += const_reference_trless<P>("La");
	tprint("Lb = La;\n");
	struct entry {
		int Ldest;
		array<int, NDIM> p;
		array<int, NDIM> k;
		double factor;
	};
	std::vector<entry> entries;
	for (int n0 = 0; n0 < Q; n0++) {
		if (n0 == 0) {
			tprint("if( do_phi ) {\n");
			indent();
		}
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[1] - n[0];
				if (n[2] <= 1 || (n[0] == 0 && n[1] == 0 && n[2] == 2)) {
					const int n0 = n[0] + n[1] + n[2];
					for (k[0] = 0; k[0] < P - n0; k[0]++) {
						for (k[1] = 0; k[1] < P - n0 - k[0]; k[1]++) {
							for (k[2] = 0; k[2] < P - n0 - k[0] - k[1]; k[2]++) {
								const auto factor = double(1) / double(vfactorial(k));
								const auto p = n + k;
								const int p0 = p[0] + p[1] + p[2];
								if (n != p) {
									if (n0 == 0) {
										if (close21(factor)) {
											tprint("Lb[%i] = fmaf( x%i%i%i, La%i%i%i, Lb[%i]);\n",
													trless_index(n[0], n[1], n[2], Q), k[0], k[1], k[2], p[0], p[1], p[2],
													trless_index(n[0], n[1], n[2], Q));
											flops += 2;
										} else {
											tprint("Lb[%i] = fmaf(T(%.8e) * x%i%i%i, La%i%i%i, Lb[%i]);\n",
													trless_index(n[0], n[1], n[2], Q), factor, k[0], k[1], k[2], p[0], p[1], p[2],
													trless_index(n[0], n[1], n[2], Q));
											flops += 3;
										}
									} else {
										entry e;
										e.Ldest = trless_index(n[0], n[1], n[2], Q);
										e.factor = factor;
										e.k = k;
										e.p = p;
										entries.push_back(e);
									}
//			L2(n) += factor * delta_x(k) * L1(p);
								}
							}
						}
					}
				}
			}
		}
		if (n0 == 0) {
			deindent();
			tprint("}\n");
		}
	}
	std::sort(entries.begin(), entries.end(), [](entry a, entry b) {
		if(a.Ldest < b.Ldest) {
			return true;
		} else if( a.Ldest > b.Ldest) {
			return false;
		} else {
			return sym_index(a.k[0], a.k[1], a.k[2]) < sym_index(b.k[0], b.k[1], b.k[2]);
		}
	});
	std::vector<entry> entries1, entries2;
	for (int i = 0; i < entries.size(); i++) {
		if (i < (entries.size() + 1) / 2) {
			entries1.push_back(entries[i]);
		} else {
			entries2.push_back(entries[i]);
		}
	}
	for (int i = 0; i < entries1.size(); i++) {
		{
			const auto factor = entries1[i].factor;
			const auto p = entries1[i].p;
			const auto k = entries1[i].k;
			const auto index = entries1[i].Ldest;
			if (close21(factor)) {
				tprint("Lb[%i] = fmaf( x%i%i%i, La%i%i%i, Lb[%i]);\n", index, k[0], k[1], k[2], p[0], p[1], p[2], index);
				flops += 2;
			} else {
				tprint("Lb[%i] = fmaf(T(%.8e) * x%i%i%i, La%i%i%i, Lb[%i]);\n", index, factor, k[0], k[1], k[2], p[0], p[1],
						p[2], index);
				flops += 3;
			}

		}
		if (entries2.size() > i) {
			const auto factor = entries2[i].factor;
			const auto p = entries2[i].p;
			const auto k = entries2[i].k;
			const auto index = entries2[i].Ldest;
			if (close21(factor)) {
				tprint("Lb[%i] = fmaf( x%i%i%i, La%i%i%i, Lb[%i]);\n", index, k[0], k[1], k[2], p[0], p[1], p[2], index);
				flops += 2;
			} else {
				tprint("Lb[%i] = fmaf(T(%.8e) * x%i%i%i, La%i%i%i, Lb[%i]);\n", index, factor, k[0], k[1], k[2], p[0], p[1],
						p[2], index);
				flops += 3;
			}

		}
	}
	tprint("return Lb;\n");
	printf("/* FLOPS = %i*/\n", flops);
	deindent();
	tprint("}\n");
}

template<int Q>
void do_expansion_cuda() {
	int flops = 0;
	struct entry {
		int Lsource;
		int Ldest;
		int xsource;
		float factor;
	};
	array<int, NDIM> n;
	array<int, NDIM> k;
	std::vector<entry> entries;
	std::vector<entry> phi_entries;
	for (int n0 = 0; n0 < Q; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[1] - n[0];
				if (n[2] <= 1 || (n[0] == 0 && n[1] == 0 && n[2] == 2)) {
					const int n0 = n[0] + n[1] + n[2];
					for (k[0] = 0; k[0] < P - n0; k[0]++) {
						for (k[1] = 0; k[1] < P - n0 - k[0]; k[1]++) {
							for (k[2] = 0; k[2] < P - n0 - k[0] - k[1]; k[2]++) {
								const auto factor = double(1) / double(vfactorial(k));
								const auto p = n + k;
								const int p0 = p[0] + p[1] + p[2];
								if (n != p) {
									entry e;
									e.Ldest = trless_index(n[0], n[1], n[2], P);
									e.xsource = sym_index(k[0], k[1], k[2]);
									e.Lsource = sym_index(p[0], p[1], p[2]);
									e.factor = factor;
									if (n0 == 0) {
										phi_entries.push_back(e);
									} else {
										entries.push_back(e);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	std::sort(entries.begin(), entries.end(), [](entry a, entry b) {
		if( a.Ldest < b.Ldest) {
			return true;
		} else if( a.Ldest > b.Ldest) {
			return false;
		} else {
			return a.Lsource < b.Lsource;
		}
	});
	tprint("#ifdef EXPANSION_CU\n");
	std::vector<entry> entries1, entries2;
	for (int i = 0; i < entries.size(); i++) {
		if (i < (entries.size() + 1) / 2) {
			entries1.push_back(entries[i]);
		} else {
			entries2.push_back(entries[i]);
		}
	}

	tprint("__managed__ char Ldest1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("%i", entries1[i].Ldest);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("__managed__ float factor1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("%.8e", entries1[i].factor);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("__managed__ char xsrc1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("%i", entries1[i].xsource);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("__managed__ char Lsrc1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("%i", entries1[i].Lsource);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");

	tprint("__managed__ char Ldest2[%i] = { ", entries2.size());
	for (int i = 0; i < entries2.size(); i++) {
		printf("%i", entries2[i].Ldest);
		if (i != entries2.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("__managed__ float factor2[%i] = { ", entries2.size());
	for (int i = 0; i < entries2.size(); i++) {
		printf("%.8e", entries2[i].factor);
		if (i != entries2.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("__managed__ char xsrc2[%i] = { ", entries2.size());
	for (int i = 0; i < entries2.size(); i++) {
		printf("%i", entries2[i].xsource);
		if (i != entries2.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("__managed__ char Lsrc2[%i] = { ", entries2.size());
	for (int i = 0; i < entries2.size(); i++) {
		printf("%i", entries2[i].Lsource);
		if (i != entries2.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");

	tprint("__managed__ float phi_factor[%i] = { ", phi_entries.size());
	for (int i = 0; i < phi_entries.size(); i++) {
		printf("%.8e", phi_entries[i].factor);
		if (i != phi_entries.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("__managed__ char phi_Lsrc[%i] = { ", phi_entries.size());
	for (int i = 0; i < phi_entries.size(); i++) {
		printf("%i", phi_entries[i].Lsource);
		if (i != phi_entries.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");

	tprint("#ifdef __CUDA_ARCH__\n");
	tprint("template<class T>\n");
	tprint("__device__\n");
	tprint(
			"tensor_trless_sym<T, %i> expansion_translate_cuda(const tensor_trless_sym<T, %i>& La, const array<T, NDIM>& X, bool do_phi) {\n",
			Q, P);

	indent();
	tprint("const int tid = threadIdx.x;\n");
	tprint("tensor_trless_sym<T, %i> Lb;\n", Q);
	tprint("tensor_sym<T, %i> Lc;\n", Q);
	tprint("Lb = 0.0f;\n");
	tprint("for( int i = tid; i < LP; i += KICK_BLOCK_SIZE ) {\n");
	indent();
	tprint("Lb[i] = La[i];\n");
	deindent();
	tprint("}\n");
	flops += compute_dx_tensor(P);
	flops += const_reference_trless_tensor<P>("La", "Lc");
	flops += 4 * entries.size();
	tprint("for( int i = tid; i < %i; i+=KICK_BLOCK_SIZE) {\n",
			entries1.size() - 1 + (entries1.size() == entries2.size() ? 1 : 0));
	indent();
	tprint("Lb[Ldest1[i]] = fmaf(factor1[i] * dx[xsrc1[i]], Lc[Lsrc1[i]], Lb[Ldest1[i]]);\n");
	tprint("Lb[Ldest2[i]] = fmaf(factor2[i] * dx[xsrc2[i]], Lc[Lsrc2[i]], Lb[Ldest2[i]]);\n");
	deindent();
	tprint("}\n");
	if (entries1.size() != entries2.size()) {
		tprint("Lb[Ldest1[%i]] = fmaf(factor1[%i] * dx[xsrc1[%i]], Lc[Lsrc1[%i]], Lb[Ldest1[%i]]);\n", entries1.size(),
				entries1.size(), entries1.size(), entries1.size(), entries1.size());
	}
	tprint("if( do_phi ) {\n");
	indent();
	tprint("for( int i = tid; i < %i; i+=KICK_BLOCK_SIZE) {\n", phi_entries.size());
	indent();
	tprint("Lb[0] = fmaf(phi_factor[i] * dx[phi_Lsrc[i]], Lc[phi_Lsrc[i]], Lb[0]);\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");

	tprint("for (int P = warpSize / 2; P >= 1; P /= 2) {\n");
	indent();
	tprint("for (int i = 0; i < LP; i++) {\n");
	indent();
	tprint("Lb[i] += __shfl_xor_sync(0xffffffff, Lb[i], P);\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");

	tprint("return Lb;\n");
	printf("/* FLOPS = %i + do_phi * %i*/\n", flops, (int) (4 * phi_entries.size()));
	deindent();
	tprint("}\n");
	tprint("#endif\n");
	tprint("#endif\n");
}

void ewald(int direct_flops) {
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT int ewald_greens_function(tensor_trless_sym<T,%i> &D, array<T, NDIM> X) {\n", P);
	indent();
	tprint("ewald_const econst;\n");
	tprint("int flops = %i;\n", 7);
	tprint("T r = SQRT(FMA(X[0], X[0], FMA(X[1], X[1], sqr(X[2]))));\n"); // 6
	tprint("const T fouroversqrtpi = T(%.8e);\n", 4.0 / sqrt(M_PI));
	tprint("tensor_sym<T, %i> Dreal;\n", P);
	tprint("tensor_trless_sym<T,%i> Dfour;\n", P);
	tprint("Dreal = 0.0f;\n");
	tprint("Dfour = 0.0f;\n");
	tprint("D = 0.0f;\n");
	tprint("const auto realsz = econst.nreal();\n");
	tprint("const T zero_mask = r > T(0);\n");                            // 1
	tprint("int icnt = 0;\n");
	tprint("for (int i = 0; i < realsz; i++) {\n");
	indent();
	tprint("const auto n = econst.real_index(i);\n");
	tprint("array<T, NDIM> dx;\n");
	tprint("for (int dim = 0; dim < NDIM; dim++) {\n");
	indent();
	tprint("dx[dim] = X[dim] - n[dim];\n");                                // 3
	deindent();
	tprint("}\n");
	tprint("T r2 = FMA(dx[0], dx[0], FMA(dx[1], dx[1], sqr(dx[2])));\n");    // 5
	tprint("if (anytrue(r2 < (EWALD_REAL_CUTOFF2))) {\n");                   // 1
	indent();
	tprint("icnt++;\n");                                       // 1
	tprint("const T r = SQRT(r2);\n");                                       // 1
	tprint("const T n8r = T(-8) * r;\n");                                       // 1
	tprint("const T rinv = (r > T(0)) / max(r, 1.0e-20);\n");                // 2
	tprint("T exp0;\n");
	tprint("T erfc0 = erfcexp(2.f * r, &exp0);\n");                          // 20
	tprint("const T expfactor = fouroversqrtpi * exp0;\n");                  // 1
	tprint("T e0 = expfactor * rinv;\n");                                   // 1
	tprint("const T rinv0 = T(1);\n");                                           // 2
	tprint("const T rinv1 = rinv;\n");                                           // 2
	for (int l = 2; l < P; l++) {
		const int i = l / 2;
		const int j = l - i;
		tprint("const T rinv%i = rinv%i * rinv%i;\n", l, i, j);                      // (P-2)
	}
	tprint("const T d0 = -erfc0 * rinv;\n");                                       // 2
	for (int l = 1; l < P; l++) {
		tprint("const T d%i = FMA(T(%i) * d%i, rinv, e0);\n", l, -2 * l + 1, l - 1);            // (P-1)*4
		if (l != P - 1) {
			tprint("e0 *= n8r;\n");                                                // (P-1)
		}
	}
	for (int l = 0; l < P; l++) {
		for (int m = 0; m <= l; m++) {
			tprint("const T Drinvpow_%i_%i = d%i * rinv%i;\n", l, m, l, m);
		}
	}
	tprint("array<T,NDIM> dxrinv;\n");
	tprint("dxrinv[0] = dx[0] * rinv;\n");
	tprint("dxrinv[1] = dx[1] * rinv;\n");
	tprint("dxrinv[2] = dx[2] * rinv;\n");
	int these_flops = compute_dx(P, "dxrinv");
	array<int, NDIM> m;
	array<int, NDIM> k;
	array<int, NDIM> n;
	these_flops += 37 + 7 * (P - 1) + P * (P + 1) / 2 + P - 2;

	these_flops += compute_detrace_ewald<P>("x", "Dreal");

	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("flops += icnt * %i;\n", these_flops);

	tprint("const auto foursz = econst.nfour();\n");
	tprint("for (int i = 0; i < foursz; i++) {\n");
	indent();
	tprint("const auto &h = econst.four_index(i);\n");
	tprint("const auto& D0 = econst.four_expansion(i);\n");
	tprint("const T hdotx = FMA(h[0], X[0], FMA(h[1], X[1], h[2] * X[2]));\n");
	tprint("T cn, sn;\n");
	tprint("sincos(T(2.0 * M_PI) * hdotx, &sn, &cn);\n");
	these_flops = 0;
	bool iscos[P * P + 1];
	for (k[0] = 0; k[0] < P; k[0]++) {
		for (k[1] = 0; k[1] < P - k[0]; k[1]++) {
			const int zmax = (k[0] == 0 && k[1] == 0) ? intmin(3, P) : intmin(P - k[0] - k[1], 2);
			for (k[2] = 0; k[2] < zmax; k[2]++) {
				const int k0 = k[0] + k[1] + k[2];
				iscos[trless_index(k[0], k[1], k[2], P)] = (k0 % 2 == 0);
			}
		}
	}
	int maxi = (P * P + 2) / 2;
	for (int i = 0; i < maxi; i++) {
		int j = i + maxi;
		tprint("Dfour[%i] = fmaf(%cn, D0[%i], Dfour[%i]);\n", i, iscos[i] ? 'c' : 's', i, i);
		these_flops += 2;
		if (j < P * P + 1) {
			tprint("Dfour[%i] = fmaf(%cn, D0[%i], Dfour[%i]);\n", j, iscos[j] ? 'c' : 's', j, j);
			these_flops += 2;
		}

	}

	deindent();
	tprint("}\n");
	reference_sym("Dreal", P);
	reference_trless("D", P);
	int those_flops = compute_detrace<P>("Dreal", "D", 'd');
	those_flops += 16 + 3 * (P * P + 1);
	tprint("flops += %i * foursz + %i;\n", these_flops, those_flops + P * P + 1);
	tprint("D = D + Dfour;\n");                                    // P*P+1
	tprint("expansion<T> D1;\n");
	tprint("direct_greens_function(D1,X);\n");
	tprint("D(0, 0, 0) = T(%.8e) + D(0, 0, 0); \n", M_PI / 4.0);                                    // 1
	tprint("for (int i = 0; i < LP; i++) {\n");
	tprint("D[i] -= D1[i];\n");                                    // 2*(P*P+1)
	indent();
	tprint("D[i] *= zero_mask;\n");
	deindent();
	tprint("}\n");
	tprint("D[0] += 2.837291e+00 * (T(1) - zero_mask);\n");                                    // 3
	tprint("if ( LORDER > 2) {\n");
	indent();
	tprint("D[3] += T(%.8e) * (T(1) - zero_mask);\n", -4.0 / 3.0 * M_PI);                                    // 3
	tprint("D[5] += T(%.8e) * (T(1) - zero_mask);\n", -4.0 / 3.0 * M_PI);                                    // 3
	tprint("D[LP - 1] += T(%.8e) * (T(1) - zero_mask);\n", -4.0 / 3.0 * M_PI);                                 // 3
	deindent();
	tprint("}\n");
	tprint("return flops;\n");
	deindent();
	tprint("}\n");

}

int main() {

	int flops = 0;

	tprint("#pragma once\n");

	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("inline int direct_greens_function(tensor_trless_sym<T, %i>& D, array<T, NDIM> X) {\n", P);
	flops = 0;
	indent();
	tprint("auto r2 = sqr(X[0], X[1], X[2]);\n");
	tprint("r2 = sqr(X[0], X[1], X[2]);\n");
	tprint("const T r = SQRT(r2);\n");
	tprint("const T rinv1 = -(r > T(0)) / max(r, T(1e-20));\n");
	for (int i = 1; i < P; i++) {
		const int j = i / 2;
		const int k = i - j;
		tprint("const T rinv%i = -rinv%i * rinv%i;\n", i + 1, j + 1, k);
	}
//	tprint("const T scale = T(1) / max(SQRT(SQRT(r2)),T(1e-4));\n");
//	tprint("const T sw = (r2 < T(25e-8));\n");
//	tprint("constexpr float scale = 837.0f;\n", 38.0 / (2 * LORDER - 1));
//	tprint("X[0] *= scale;\n");
//	tprint("X[1] *= scale;\n");
//	tprint("X[2] *= scale;\n");
	tprint("X[0] *= rinv1;\n");
	tprint("X[1] *= rinv1;\n");
	tprint("X[2] *= rinv1;\n");
	flops += 12;
	flops += compute_dx(P);
	reference_trless("D", P);
	flops += compute_detrace<P>("x", "D");
	flops += 11 + (P - 1) * 2;
	array<int, NDIM> k;
//	tprint("constexpr float scale0 = scale;\n");
//	for (int l = 1; l < P; l++) {
//		tprint("constexpr float scale%i = scale%i * scale;\n", l, l - 1);
//	}
	/*	for (int l = 0; l < P; l++) {
	 tprint("rinv_pow[%i] *= scale%i;\n", l, l);
	 }
	 flops += P;*/
	for (k[0] = 0; k[0] < P; k[0]++) {
		for (k[1] = 0; k[1] < P - k[0]; k[1]++) {
			const int zmax = (k[0] == 0 && k[1] == 0) ? intmin(3, P) : intmin(P - k[0] - k[1], 2);
			for (k[2] = 0; k[2] < zmax; k[2]++) {
				const int k0 = k[0] + k[1] + k[2];
				tprint("D%i%i%i *= rinv%i;\n", k[0], k[1], k[2], k0 + 1);
				flops++;
			}
		}
	}
	tprint("return %i;\n", flops);
	deindent();
	tprint("}\n");

	ewald(flops);

	array<int, NDIM> n;
	array<int, NDIM> m;
	for (int Pmax = 2; Pmax <= P; Pmax += P - 2) {
		flops = 0;
		tprint("\n\ntemplate<class T>\n");
		tprint("CUDA_EXPORT\n");
		tprint(
				"inline int interaction(tensor_trless_sym<T, %i>& L, const tensor_trless_sym<T, %i>& M, const tensor_trless_sym<T, %i>& D, bool do_phi) {\n",
				Pmax, P - 1, P);
		indent();
		int phi_flops = 0;
		flops += const_reference_trless<P - 1>("M");
		flops += const_reference_trless<P>("D");
		n[0] = n[1] = n[2] = 0;
		const int n0 = 0;
		const int q0 = intmin(P - n0, P - 1);
		struct entry_type {
			array<int, NDIM> n;
			array<int, NDIM> m;
			double coeff;
		};
		std::vector<entry_type> entries0;
		std::vector<entry_type> entries;
		for (n[0] = 0; n[0] < Pmax; n[0]++) {
			for (n[1] = 0; n[1] < Pmax - n[0]; n[1]++) {
				const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, Pmax) : intmin(Pmax - n[0] - n[1], 2);
				for (n[2] = 0; n[2] < nzmax; n[2]++) {
					const int n0 = n[0] + n[1] + n[2];
					const int q0 = intmin(P - n0, P - 1);
					for (m[0] = 0; m[0] < q0; m[0]++) {
						for (m[1] = 0; m[1] < q0 - m[0]; m[1]++) {
							for (m[2] = 0; m[2] < q0 - m[0] - m[1]; m[2]++) {
								const double coeff = 1.0 / vfactorial(m);
								entry_type e;
								e.coeff = coeff;
								e.n = n;
								e.m = m;
								if (n0 == 0) {
									entries0.push_back(e);
								} else {
									entries.push_back(e);
								}
							}
						}
					}
				}
			}
		}
		const auto sort_func = [Pmax](entry_type a, entry_type b) {
			if( trless_index(a.n[0], a.n[1], a.n[2], Pmax) < trless_index(b.n[0], b.n[1], b.n[2], Pmax)) {
				return true;
			}
			return false;
		};
		std::sort(entries.begin(), entries.end(), sort_func);
		std::vector<entry_type> entries1, entries2;
		for (int i = 0; i < entries.size(); i++) {
			if (i < (entries.size() + 1) / 2) {
				entries1.push_back(entries[i]);
			} else {
				entries2.push_back(entries[i]);
			}
		}
		tprint("if( do_phi ) {\n");
		indent();
		for (int i = 0; i < entries0.size(); i++) {
			const auto coeff = entries0[i].coeff;
			const auto n = entries0[i].n;
			const auto m = entries0[i].m;
			if (close21(coeff)) {
				tprint("L[0] = fmaf(M%i%i%i, D%i%i%i, L[0]);\n", m[0], m[1], m[2], n[0] + m[0], n[1] + m[1], n[2] + m[2]);
				phi_flops += 2;
			} else {
				phi_flops += 3;
				tprint("L[0] = fmaf(T(%.8e) * M%i%i%i, D%i%i%i, L[0]);\n", coeff, m[0], m[1], m[2], n[0] + m[0],
						n[1] + m[1], n[2] + m[2]);
			}
		}
		deindent();
		tprint("}\n");
		for (int i = 0; i < entries1.size(); i++) {
			{
				const auto coeff = entries1[i].coeff;
				const auto n = entries1[i].n;
				const auto m = entries1[i].m;
				if (close21(coeff)) {
					tprint("L[%i] = fmaf(M%i%i%i, D%i%i%i, L[%i]);\n", trless_index(n[0], n[1], n[2], Pmax), m[0], m[1],
							m[2], n[0] + m[0], n[1] + m[1], n[2] + m[2], trless_index(n[0], n[1], n[2], Pmax));
					flops += 2;
				} else {
					flops += 3;
					tprint("L[%i] = fmaf(T(%.8e) * M%i%i%i, D%i%i%i, L[%i]);\n", trless_index(n[0], n[1], n[2], Pmax), coeff,
							m[0], m[1], m[2], n[0] + m[0], n[1] + m[1], n[2] + m[2], trless_index(n[0], n[1], n[2], Pmax));
				}
			}
			if (i < entries2.size()) {
				const auto coeff = entries2[i].coeff;
				const auto n = entries2[i].n;
				const auto m = entries2[i].m;
				if (close21(coeff)) {
					tprint("L[%i] = fmaf(M%i%i%i, D%i%i%i, L[%i]);\n", trless_index(n[0], n[1], n[2], Pmax), m[0], m[1],
							m[2], n[0] + m[0], n[1] + m[1], n[2] + m[2], trless_index(n[0], n[1], n[2], Pmax));
					flops += 2;
				} else {
					flops += 3;
					tprint("L[%i] = fmaf(T(%.8e) * M%i%i%i, D%i%i%i, L[%i]);\n", trless_index(n[0], n[1], n[2], Pmax), coeff,
							m[0], m[1], m[2], n[0] + m[0], n[1] + m[1], n[2] + m[2], trless_index(n[0], n[1], n[2], Pmax));
				}
			}
		}
		tprint("return %i + do_phi * %i;\n", flops, phi_flops);
		deindent();
		tprint("}\n");
	}

	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("tensor_trless_sym<T, %i> monopole_translate(array<T, NDIM>& X) {\n", P - 1);
	flops = 0;
	indent();
	tprint("tensor_trless_sym<T, %i> M;\n", P - 1);
	tprint("X[0] = -X[0];\n");
	tprint("X[1] = -X[1];\n");
	tprint("X[2] = -X[2];\n");
	reference_trless("M", P - 1);
	flops += 3;
	flops += compute_dx(P - 1);
	flops += compute_detrace<P - 1>("x", "M", 'd');
	tprint("return M;\n");
	printf("/* FLOPS = %i*/\n", flops);
	deindent();
	tprint("}\n");

	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("tensor_trless_sym<T, %i> multipole_translate(const tensor_trless_sym<T,%i>& Ma, array<T, NDIM>& X) {\n",
	P - 1, P - 1);
	flops = 0;
	indent();
	tprint("tensor_sym<T, %i> Mb;\n", P - 1);
	tprint("tensor_trless_sym<T, %i> Mc;\n", P - 1);
	tprint("X[0] = -X[0];\n");
	tprint("X[1] = -X[1];\n");
	tprint("X[2] = -X[2];\n");
	flops += const_reference_trless<P - 1>("Ma");
	reference_sym("Mb", P - 1);
	reference_trless("Mc", P - 1);
	flops += compute_dx(P - 1);

	for (int i = 0; i < (P - 1) * P * (P + 1) / 6; i++) {
		for (n[0] = 0; n[0] < P - 1; n[0]++) {
			for (n[1] = 0; n[1] < P - n[0] - 1; n[1]++) {
				for (n[2] = 0; n[2] < P - n[0] - n[1] - 1; n[2]++) {
					if (i == sym_index(n[0], n[1], n[2])) {
						tprint("Mb[%i] = Ma%i%i%i;\n", i, n[0], n[1], n[2]);
					}
				}
			}
		}
	}
	struct mentry {
		array<int, NDIM> n;
		array<int, NDIM> k;
		double factor;
	};
	std::vector<mentry> mentries;
	for (int n0 = 0; n0 <= P - 2; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				for (k[0] = 0; k[0] <= intmin(n0, n[0]); k[0]++) {
					for (k[1] = 0; k[1] <= intmin(n0 - k[0], n[1]); k[1]++) {
						for (k[2] = 0; k[2] <= intmin(n0 - k[0] - k[1], n[2]); k[2]++) {
							const auto factor = (vfactorial(n)) / double(vfactorial(k) * vfactorial(n - k));
							if (n != k) {
								mentry e;
								e.n = n;
								e.k = k;
								e.factor = factor;
								mentries.push_back(e);
							}
						}
					}
				}
			}
		}
	}
	std::sort(mentries.begin(), mentries.end(), [](mentry a, mentry b) {
		if( sym_index(a.n[0], a.n[1], a.n[2]) < sym_index(b.n[0], b.n[1], b.n[2])) {
			return true;
		}
		return false;
	});
	std::vector<mentry> entries1, entries2;
	for (int i = 0; i < mentries.size(); i++) {
		if (i < (mentries.size() + 1) / 2) {
			entries1.push_back(mentries[i]);
		} else {
			entries2.push_back(mentries[i]);
		}
	}
	for (int i = 0; i < entries1.size(); i++) {
		for (int pass = 0; pass < 2; pass++) {
			if (pass == 1 && i >= entries2.size()) {
				continue;
			}
			const auto n = pass == 0 ? entries1[i].n : entries2[i].n;
			const auto k = pass == 0 ? entries1[i].k : entries2[i].k;
			const auto factor = pass == 0 ? entries1[i].factor : entries2[i].factor;
			if (close21(factor)) {
				tprint("Mb[%i] = fmaf( x%i%i%i, Ma%i%i%i, Mb[%i]);\n", sym_index(n[0], n[1], n[2]), n[0] - k[0],
						n[1] - k[1], n[2] - k[2], k[0], k[1], k[2], sym_index(n[0], n[1], n[2]));
				flops += 2;
			} else {
				tprint("Mb[%i] = fmaf(T(%.8e) * x%i%i%i, Ma%i%i%i, Mb[%i]);\n", sym_index(n[0], n[1], n[2]), factor,
						n[0] - k[0], n[1] - k[1], n[2] - k[2], k[0], k[1], k[2], sym_index(n[0], n[1], n[2]));
				flops += 3;
			}
		}

	}

	flops += compute_detrace<P - 1>("Mb", "Mc", 'd');

	tprint("return Mc;\n");
	printf("/* FLOPS = %i*/\n", flops);
	deindent();
	tprint("}\n");

	do_expansion<P>(false);

	do_expansion<2>(true);

	do_expansion_cuda<P>();

}
