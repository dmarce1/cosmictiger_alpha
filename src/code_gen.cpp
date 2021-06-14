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

int compute_dx(int P) {
	array<int, NDIM> n;
	tprint("const T x000 = T(1);\n");
	tprint("const T& x100 = X[0];\n");
	tprint("const T& x010 = X[1];\n");
	tprint("const T& x001 = X[2];\n");
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
	tprint("%s%i%i%i = %e * %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], b, c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int fma(std::string a, array<int, NDIM> n, double b, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i = fmaf(%e, %s%i%i%i, %s%i%i%i);\n", a.c_str(), n[0], n[1], n[2], b, c.c_str(), j[0], j[1], j[2],
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
	for (int pass = 0; pass < 2; pass++) {
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
										double number = fnm * num / den;
										if( type == 'd') {
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

int main() {

	int flops = 0;


	tprint("#pragma once\n");


	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("inline tensor_trless_sym<T, %i> direct_greens_function(const array<T, NDIM> X) {\n", P);
	flops = 0;
	indent();
	tprint("tensor_trless_sym<T, %i> D;\n", P);
	flops += compute_dx(P);
	reference_trless("D", P);
	flops += compute_detrace<P>("x", "D");
	tprint("array<T, %i> rinv_pow;\n", P);
	tprint("const auto r2 = sqr(X[0], X[1], X[2]);\n");
	tprint("const auto r = sqrt(r2);\n");
	tprint("const auto rinv = (r > T(0)) / max(r, 1e-20);\n");
	tprint("const auto rinv2 = rinv * rinv;\n");
	tprint("rinv_pow[0] = -rinv;\n");
	tprint("for (int i = 1; i < %i; i++) {\n", P);
	tprint("\trinv_pow[i] = -rinv2 * rinv_pow[i - 1];\n");
	tprint("}\n");
	flops += 11 + (P - 1) * 2;
	array<int, NDIM> k;
	for (k[0] = 0; k[0] < P; k[0]++) {
		for (k[1] = 0; k[1] < P - k[0]; k[1]++) {
			const int zmax = (k[0] == 0 && k[1] == 0) ? intmin(3, P) : intmin(P - k[0] - k[1], 2);
			for (k[2] = 0; k[2] < zmax; k[2]++) {
				const int k0 = k[0] + k[1] + k[2];
				tprint("D%i%i%i *= rinv_pow[%i];\n", k[0], k[1], k[2], k0);
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
			"inline void direct_interaction(tensor_trless_sym<T, %i>& L, const tensor_trless_sym<T, %i>& M, const tensor_trless_sym<T, %i>& D) {\n",
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




	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("tensor_trless_sym<T, %i> monopole_translate(array<T, NDIM>& X) {\n", P - 1);
	flops = 0;
	indent();
	tprint("tensor_trless_sym<T, %i> M;\n", P - 1);
	tprint( "X[0] = -X[0];\n");
	tprint( "X[1] = -X[1];\n");
	tprint( "X[2] = -X[2];\n");
	reference_trless("M", P - 1);
	flops += 3;
	flops += compute_dx(P-1);
	flops += compute_detrace<P-1>("x", "M", 'd');
	tprint("return M;\n");
	printf("/* FLOPS = %i*/\n", flops);
	deindent();
	tprint("}\n");


}
