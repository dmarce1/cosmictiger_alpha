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
	tprint("%s%i%i%i = %.8e * %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], b, c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int fma(std::string a, array<int, NDIM> n, double b, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i = fmaf(%.8e, %s%i%i%i, %s%i%i%i);\n", a.c_str(), n[0], n[1], n[2], b, c.c_str(), j[0], j[1], j[2],
			a.c_str(), n[0], n[1], n[2]);
	return 2;
}

bool close21(double a) {
	return std::abs(1.0 - a) < 1.0e-20;
}

template<int P>
int compute_detrace(std::string iname, std::string oname, char type = 'f') {
	array<int, NDIM> m;
	array<int, NDIM> k;
	array<int, NDIM> n;
	tensor_sym<int, P> first_use;
	first_use = 1;
	int flops = 0;
	for (int pass = 0; pass < 3; pass++) {
		for (n[0] = 0; n[0] < P; n[0]++) {
			for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
				const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
				for (n[2] = 0; n[2] < nzmax; n[2]++) {
					const int n0 = n[0] + n[1] + n[2];
					for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
						for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
							for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
								const int m0 = m[0] + m[1] + m[2];
								if (type == 'd' && ((n0 == 2 && (n[0] == 2 || n[1] == 2 || n[2] == 2)) && m0 == 1)) {
									continue;
								}
								double num = double(n1pow(m0) * dfactorial(2 * n0 - 2 * m0 - 1) * vfactorial(n));
								double den = double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
								const double fnm = num / den;
								for (k[0] = 0; k[0] <= m0; k[0]++) {
									for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
										k[2] = m0 - k[0] - k[1];
										const auto p = n - (m) * 2 + (k) * 2;
										num = factorial(m0);
										den = vfactorial(k);
										double number = fnm * num / den;
										if (type == 'd') {
											number *= 1.0 / dfactorial(2 * n0 - 1);
										}
										if ((number < 0 && pass == 0) || (number > 0 && pass == 1)) {
											number = std::abs(number);
											if (first_use(n)) {
												if (close21(number)) {
													flops += eqp(oname, n, iname, p);
												} else {
													flops += mul(oname, n, number, iname, p);
												}
												first_use(n) = 0;
											} else {
												if (close21(number)) {
													flops += acc(oname, n, iname, p);
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
		}
		if (pass == 0) {
			for (n[0] = 0; n[0] < P; n[0]++) {
				for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
					const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
					for (n[2] = 0; n[2] < nzmax; n[2]++) {
						const int n0 = n[0] + n[1] + n[2];
						if (!first_use(n)) {
							tprint("%s%i%i%i = -%s%i%i%i;\n", oname.c_str(), n[0], n[1], n[2], oname.c_str(), n[0], n[1],
									n[2]);
							flops++;
						}
					}
				}
			}
		}
	}
	return flops;
}

#define P ORDER

int trless_index(int l, int m, int n, int Q) {
	return (l + m) * ((l + m) + 1) / 2 + (m) + (Q * (Q + 1) / 2) * (n == 1) + (Q * Q) * (n == 2);
}

int sym_index(int l, int m, int n) {
	return (l + m + n) * (l + m + n + 1) * ((l + m + n) + 2) / 6 + (m + n) * ((m + n) + 1) / 2 + n;
}

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

int interaction_code() {
	int flops = 0;
	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint(
			"inline tensor_trless_sym<T, %i> interaction(const tensor_trless_sym<T, %i>& M, const tensor_trless_sym<T, %i>& D ) {\n",
			P, P - 1, P);
	indent();
	tprint("tensor_trless_sym<T, %i> L;\n", P);
	tensor_sym<int, P> first_use;
	first_use = 1;
	array<int, NDIM> n;
	array<int, NDIM> m;
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
			for (n[2] = 0; n[2] < nzmax; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				const int q0 = P + 1 - n0;
				for (m[0] = 0; m[0] < q0; m[0]++) {
					for (m[1] = 0; m[1] < q0 - m[0]; m[1]++) {
						for (m[2] = 0; m[2] < q0 - m[0] - m[1]; m[2]++) {
							const double number = 1.0 / vfactorial(m);
							if (first_use(n)) {
								first_use(n) = 0;
							} else {
								//		L(n) += M(m) * D(n + m) * number;
							}
						}
					}
				}
			}
		}
	}
	printf("/* FLOPS = %i*/\n", flops);
	deindent();
	tprint("}\n");
	return flops;
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
				"tensor_trless_sym<T, %i> expansion_translate2(const tensor_trless_sym<T, %i>& La, const array<T, NDIM>& X) {\n",
				Q, P);
	} else {
		tprint(
				"tensor_trless_sym<T, %i> expansion_translate(const tensor_trless_sym<T, %i>& La, const array<T, NDIM>& X) {\n",
				Q, P);

	}
	indent();
	tprint("tensor_trless_sym<T, %i> Lb;\n", Q);
	flops += compute_dx(P);
	array<int, NDIM> n;
	array<int, NDIM> k;
	flops += const_reference_trless<P>("La");
	tprint("Lb = La;\n");
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
									if (close21(factor)) {
										tprint("Lb[%i] = fmaf( x%i%i%i, La%i%i%i, Lb[%i]);\n", trless_index(n[0], n[1], n[2], Q),
												k[0], k[1], k[2], p[0], p[1], p[2], trless_index(n[0], n[1], n[2], Q));
										flops += 2;
									} else {
										tprint("Lb[%i] = fmaf(T(%.8e) * x%i%i%i, La%i%i%i, Lb[%i]);\n",
												trless_index(n[0], n[1], n[2], Q), factor, k[0], k[1], k[2], p[0], p[1], p[2],
												trless_index(n[0], n[1], n[2], Q));
										flops += 3;
									}

									//			L2(n) += factor * delta_x(k) * L1(p);
								}
							}
						}
					}
				}
			}
		}
	}
	tprint("return Lb;\n");
	printf("/* FLOPS = %i*/\n", flops);
	deindent();
	tprint("}\n");
}

void ewald() {
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT int ewald_greens_function(tensor_trless_sym<T,%i> &D, array<T, NDIM> X) {\n", P);
	indent();
	tprint("ewald_const econst;\n");
	tprint("T r = SQRT(FMA(X[0], X[0], FMA(X[1], X[1], sqr(X[2]))));\n");
	tprint("const T fouroversqrtpi = T(4.0 / SQRT(M_PI));\n");
	tprint("tensor_sym<T, %i> Dreal;\n", P);
	tprint("tensor_trless_sym<T,%i> Dfour;\n", P);
	tprint("Dreal = 0.0f;\n");
	tprint("Dfour = 0.0f;\n");
	tprint("D = 0.0f;\n");
	tprint("const auto realsz = econst.nreal();\n");
	tprint("const T zero_mask = (sqr(X[0], X[1], X[2]) > T(0));\n");
	tprint("T tmp;\n");
	tprint("for (int i = 0; i < realsz; i++) {\n");
	indent();
	tprint("const auto n = econst.real_index(i);\n");
	tprint("array<T, NDIM> dx;\n");
	tprint("for (int dim = 0; dim < NDIM; dim++) {\n");
	indent();
	tprint("dx[dim] = X[dim] - n[dim];\n");
	deindent();
	tprint("}");
	tprint("T r2 = FMA(dx[0], dx[0], FMA(dx[1], dx[1], sqr(dx[2])));\n");
	tprint("if (anytrue(r2 < (EWALD_REAL_CUTOFF2))) {\n");
	indent();
	tprint("const T r = SQRT(r2);\n");                                                         // FLOP_SQRT
	tprint("const T rinv = (r > T(0)) / max(r, 1.0e-20);\n");
	tprint("const T r2inv = rinv * rinv;\n");
	tprint("T exp0;\n");
	tprint("T erfc0 = erfcexp(2.f * r, &exp0);\n");                                           // 18 + FLOP_DIV + FLOP_EXP
	tprint("const T expfactor = fouroversqrtpi * exp0;\n");                                // 2
	tprint("T e1 = expfactor * r2inv;\n");                                               // 1
	tprint("array<T, LORDER> d;\n");
	tprint("d[0] = -erfc0 * rinv;\n");
	tprint("for (int l = 1; l < LORDER; l++) {\n");
	indent();
	tprint("d[l] = FMA(T(-2 * l + 1) * d[l - 1], r2inv, e1);\n");
	tprint("e1 *= T(-8);\n");
	deindent();
	tprint("}\n");
	tprint("array<T, LORDER> rinv2pow;\n");
	tprint("rinv2pow[0] = T(1);\n");                                               // 2
	tprint("for (int l = 1; l < LORDER; l++) {\n");
	indent();
	tprint("	rinv2pow[l] = rinv2pow[l - 1] * rinv * rinv;\n");
	deindent();
	tprint("}\n");
	compute_dx(P, "dx");
	array<int, NDIM> m;
	array<int, NDIM> k;
	array<int, NDIM> n;
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			for (n[2] = 0; n[2] < P - n[0] - n[1]; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
					for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
						for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
							const int m0 = m[0] + m[1] + m[2];
							double num = double(vfactorial(n));
							double den = double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
							const double fnm = num / den;
							tprint("tmp =  d[%i] * rinv2pow[%i];\n", n0 - m0, m0);
							for (k[0] = 0; k[0] <= m0; k[0]++) {
								for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
									k[2] = m0 - k[0] - k[1];
									const auto p = n - (m) * 2 + (k) * 2;
									num = factorial(m0);
									den = vfactorial(k);
									const double number = fnm * num / den;
									if (close21(number)) {
										tprint("Dreal[%i] = fmaf(x%i%i%i, tmp, Dreal[%i]);\n", sym_index(n[0], n[1], n[2]), p[0],
												p[1], p[2], sym_index(n[0], n[1], n[2]));
									} else {
										tprint("Dreal[%i] = fmaf(%.8e * x%i%i%i, tmp, Dreal[%i]);\n", sym_index(n[0], n[1], n[2]),
												number, p[0], p[1], p[2], sym_index(n[0], n[1], n[2]));
									}
								}
							}
						}
					}
				}
			}
		}
	}
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");

	tprint("const auto foursz = econst.nfour();\n");
	tprint("for (int i = 0; i < foursz; i++) {\n");
	indent();
	tprint("const auto &h = econst.four_index(i);\n");
	tprint("const auto& D0 = econst.four_expansion(i);\n");
	tprint("const T hdotx = FMA(h[0], X[0], FMA(h[1], X[1], h[2] * X[2]));\n");
	tprint("T co, so;\n");
	tprint("sincos(T(2.0 * M_PI) * hdotx, &so, &co);\n");
	for (k[0] = 0; k[0] < P; k[0]++) {
		for (k[1] = 0; k[1] < P - k[0]; k[1]++) {
			const int zmax = (k[0] == 0 && k[1] == 0) ? intmin(3, P) : intmin(P - k[0] - k[1], 2);
			for (k[2] = 0; k[2] < zmax; k[2]++) {
				const int k0 = k[0] + k[1] + k[2];
				tprint("Dfour[%i] = fmaf(%co, D0[%i], Dfour[%i]);\n", trless_index(k[0], k[1], k[2], P), k0 % 2 ? 's' : 'c',
						trless_index(k[0], k[1], k[2], P), trless_index(k[0], k[1], k[2], P));
			}
		}
	}
	deindent();
	printf("}\n");
	reference_sym("Dreal", P);
	reference_trless("D", P);
	compute_detrace<P>("Dreal", "D", 'd');
	tprint("D = D + Dfour;\n");
	tprint("expansion<T> D1 = direct_greens_function(X);\n");
	tprint("D(0, 0, 0) = T(M_PI / 4.0) + D(0, 0, 0); \n");
	tprint("for (int i = 0; i < LP; i++) {\n");
	tprint("D[i] -= D1[i];\n");
	indent();
	tprint("D[i] *= zero_mask;\n");
	deindent();
	tprint("}\n");
	tprint("D[0] += 2.837291e+00 * (T(1) - zero_mask);\n");
	tprint("if ( LORDER > 2) {\n");
	indent();
	tprint("D[3] += -4.0 / 3.0 * M_PI * (T(1) - zero_mask);\n");
	tprint("D[5] += -4.0 / 3.0 * M_PI * (T(1) - zero_mask);\n");
	tprint("D[LP - 1] += -4.0 / 3.0 * M_PI * (T(1) - zero_mask);\n");
	deindent();
	tprint("}\n");
	tprint("return 0;\n");
	deindent();
	tprint("}\n");

}

int main() {

	int flops = 0;

	tprint("#pragma once\n");

	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("inline tensor_trless_sym<T, %i> direct_greens_function(array<T, NDIM> X) {\n", P);
	flops = 0;
	indent();
	tprint("auto r2 = sqr(X[0], X[1], X[2]);\n");
//	tprint("const T scale = max(SQRT(SQRT(r2)),T(1e-8));\n");
	tprint("const T scale = 1.0e-4;\n");
	tprint("X[0] *= scale;\n");
	tprint("X[1] *= scale;\n");
	tprint("X[2] *= scale;\n");
	flops += 12;
	tprint("tensor_trless_sym<T, %i> D;\n", P);
	flops += compute_dx(P);
	reference_trless("D", P);
	flops += compute_detrace<P>("x", "D");
	tprint("array<T, %i> rinv_pow;\n", P);
	tprint("r2 = sqr(X[0], X[1], X[2]);\n");
	tprint("const T r = SQRT(r2);\n");
	tprint("const T rinv = (r > T(0)) / max(r, T(1e-20));\n");
	tprint("const T rinv2 = rinv * rinv;\n");
	tprint("rinv_pow[0] = -rinv;\n");
	tprint("for (int i = 1; i < %i; i++) {\n", P);
	tprint("\trinv_pow[i] = -rinv2 * rinv_pow[i - 1];\n");
	tprint("}\n");
	flops += 11 + (P - 1) * 2;
	array<int, NDIM> k;
	tprint("T scale0 = scale;\n");
	for (int l = 1; l < P; l++) {
		tprint("T scale%i = scale%i * scale;\n", l, l - 1);
	}
	flops += P - 1;
	for (int l = 0; l < P; l++) {
		tprint("scale%i *= rinv_pow[%i];\n", l, l);
	}
	flops += P;
	for (k[0] = 0; k[0] < P; k[0]++) {
		for (k[1] = 0; k[1] < P - k[0]; k[1]++) {
			const int zmax = (k[0] == 0 && k[1] == 0) ? intmin(3, P) : intmin(P - k[0] - k[1], 2);
			for (k[2] = 0; k[2] < zmax; k[2]++) {
				const int k0 = k[0] + k[1] + k[2];
				tprint("D%i%i%i *= scale%i;\n", k[0], k[1], k[2], k0);
				flops++;
			}
		}
	}
	tprint("return D;\n");
	printf("/* FLOPS = %i*/\n", flops);
	deindent();
	tprint("}\n");

	flops = 0;
	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint(
			"inline void interaction(tensor_trless_sym<T, %i>& L, const tensor_trless_sym<T, %i>& M, const tensor_trless_sym<T, %i>& D) {\n",
			P, P - 1, P);
	indent();
	flops += const_reference_trless<P - 1>("M");
	flops += const_reference_trless<P>("D");
	reference_trless("L", P);
	array<int, NDIM> n;
	array<int, NDIM> m;
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
			for (n[2] = 0; n[2] < nzmax; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				const int q0 = intmin(P - n0, P - 1);
				for (m[0] = 0; m[0] < q0; m[0]++) {
					for (m[1] = 0; m[1] < q0 - m[0]; m[1]++) {
						for (m[2] = 0; m[2] < q0 - m[0] - m[1]; m[2]++) {
							const double coeff = 1.0 / vfactorial(m);
							if (close21(coeff)) {
								tprint("L%i%i%i = fmaf(M%i%i%i, D%i%i%i, L%i%i%i);\n", n[0], n[1], n[2], m[0], m[1], m[2],
										n[0] + m[0], n[1] + m[1], n[2] + m[2], n[0], n[1], n[2]);
								flops += 2;
							} else {
								flops += 3;
								tprint("L%i%i%i = fmaf(T(%.8e) * M%i%i%i, D%i%i%i, L%i%i%i);\n", n[0], n[1], n[2], coeff, m[0],
										m[1], m[2], n[0] + m[0], n[1] + m[1], n[2] + m[2], n[0], n[1], n[2]);
							}
							//			L(n) += M(m) * D(n + m);
						}
					}
				}
			}
		}
	}
	printf("/* FLOPS = %i*/\n", flops);
	deindent();
	tprint("}\n");
	if ( P > 2) {
		flops = 0;
		tprint("\n\ntemplate<class T>\n");
		tprint("CUDA_EXPORT\n");
		tprint(
				"inline void interaction(tensor_trless_sym<T, 2>& L, const tensor_trless_sym<T, %i>& M, const tensor_trless_sym<T, %i>& D) {\n",
				P - 1, P);
		indent();
		flops += const_reference_trless<P - 1>("M");
		flops += const_reference_trless<P>("D");
		reference_trless("L", 2);
		for (n[0] = 0; n[0] < 2; n[0]++) {
			for (n[1] = 0; n[1] < 2 - n[0]; n[1]++) {
				const int nzmax = (n[0] == 0 && n[1] == 0) ? 2 : intmin(2 - n[0] - n[1], 2);
				for (n[2] = 0; n[2] < nzmax; n[2]++) {
					const int n0 = n[0] + n[1] + n[2];
					const int q0 = intmin(P - n0, P - 1);
					for (m[0] = 0; m[0] < q0; m[0]++) {
						for (m[1] = 0; m[1] < q0 - m[0]; m[1]++) {
							for (m[2] = 0; m[2] < q0 - m[0] - m[1]; m[2]++) {
								const double coeff = 1.0 / vfactorial(m);
								if (close21(coeff)) {
									tprint("L%i%i%i = fmaf(M%i%i%i, D%i%i%i, L%i%i%i);\n", n[0], n[1], n[2], m[0], m[1], m[2],
											n[0] + m[0], n[1] + m[1], n[2] + m[2], n[0], n[1], n[2]);
									flops += 2;
								} else {
									flops += 3;
									tprint("L%i%i%i = fmaf(T(%.8e) * M%i%i%i, D%i%i%i, L%i%i%i);\n", n[0], n[1], n[2], coeff,
											m[0], m[1], m[2], n[0] + m[0], n[1] + m[1], n[2] + m[2], n[0], n[1], n[2]);
								}
								//			L(n) += M(m) * D(n + m);
							}
						}
					}
				}
			}
		}
		printf("/* FLOPS = %i*/\n", flops);
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

	for (n[0] = 0; n[0] < P - 1; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0] - 1; n[1]++) {
			for (n[2] = 0; n[2] < P - n[0] - n[1] - 1; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				tprint("Mb[%i] = Ma%i%i%i;\n", sym_index(n[0], n[1], n[2]), n[0], n[1], n[2]);
			}
		}
	}
	for (int n0 = P - 2; n0 >= 0; n0--) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				for (k[0] = 0; k[0] <= intmin(n0, n[0]); k[0]++) {
					for (k[1] = 0; k[1] <= intmin(n0 - k[0], n[1]); k[1]++) {
						for (k[2] = 0; k[2] <= intmin(n0 - k[0] - k[1], n[2]); k[2]++) {
							const auto factor = (vfactorial(n)) / double(vfactorial(k) * vfactorial(n - k));
							if (n != k) {
								if (close21(factor)) {
									tprint("Mb[%i] = fmaf( x%i%i%i, Mb[%i], Mb[%i]);\n", sym_index(n[0], n[1], n[2]),
											n[0] - k[0], n[1] - k[1], n[2] - k[2], sym_index(k[0], k[1], k[2]),
											sym_index(n[0], n[1], n[2]));
									flops += 2;
								} else {
									tprint("Mb[%i] = fmaf(T(%.8e) * x%i%i%i, Mb[%i], Mb[%i]);\n", sym_index(n[0], n[1], n[2]),
											factor, n[0] - k[0], n[1] - k[1], n[2] - k[2], sym_index(k[0], k[1], k[2]),
											sym_index(n[0], n[1], n[2]));
									flops += 3;
								}
							}
						}
					}
				}
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

	ewald();

}
