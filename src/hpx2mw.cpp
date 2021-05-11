#include <stdio.h>
#include <stdlib.h>
#include <silo.h>

#include <chealpix/chealpix.h>

#include <boost/program_options.hpp>

#include <vector>

void sph_to_cart(double psi, double lambda, double* x, double* y) {
	double theta = psi;
	double theta0;
	int iters = 0;
	do {
		theta0 = theta;
		theta -= (2.0 * theta + std::sin(2.0 * theta) - M_PI * std::sin(psi)) / (2.0 + 2.0 * std::cos(2.0 * theta));
		iters++;
	} while (std::abs(theta0 - theta) > 1.0e-6);
	*x = 2.0 * lambda * std::cos(theta) / (M_PI);
	*y = std::sin(theta);
}

bool cart_to_sph(double x, double y, double* psi, double* lambda) {
	const auto theta = std::asin(y);
	const auto arg = (2.0 * theta + std::sin(2.0 * theta)) / M_PI;

	*psi = std::asin(arg);
	*lambda = M_PI * x / (2.0 * std::cos(theta));
	if (*lambda < -M_PI || *lambda > M_PI) {
		return false;
	} else {
		return true;
	}
}

int main(int argc, char **argv) {
	std::string infile, outfile;

	namespace po = boost::program_options;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("in", po::value<std::string>(&infile)->default_value(""), "input file") //
	("out", po::value<std::string>(&outfile)->default_value(""), "output file") //
			;

	boost::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);

	if (infile == "") {
		printf("Input file not specified. Use --in=\n");
		return -1;
	}

	if (outfile == "") {
		printf("Output file not specified. Use --out=\n");
		return -1;
	}

	FILE* fp = fopen(infile.c_str(), "rb");
	if (!fp) {
		printf("Unable to open %s for writing.\n", infile.c_str());
		return -1;
	}
	fseek(fp, 0L, SEEK_END);
	const int npix = ftell(fp) / sizeof(float);
	fseek(fp, 0L, SEEK_SET);
	const int Nside = std::sqrt(npix / 12);
	printf("Reading HEALPix data with Nside = %i and %i pixels\n", Nside, npix);
	const int res = Nside * std::sqrt(1.5) + 0.5;
	std::vector<float> hpx_data(npix);
	if (fread(hpx_data.data(), sizeof(float), npix, fp) != npix) {
		printf("Unable to read data from %s.\n", infile.c_str());
	}

	fclose(fp);

	int ORDER = 3;
	std::vector<float> mw_data;
	for (int iy = -res; iy < res; iy++) {
		for (int ix = -2 * res; ix < 2 * res; ix++) {
			double value = 0.0;
			for (int nx = 0; nx < ORDER; nx++) {
				for (int ny = 0; ny < ORDER; ny++) {
					const double x = (ix + (nx + 0.5) / ORDER) / double(res);
					const double y = (iy + (ny + 0.5) / ORDER) / double(res);
					double psi, lambda;
					if (cart_to_sph(x, y, &psi, &lambda)) {
						long ipring;
						ang2pix_ring(Nside, psi + M_PI / 2.0, lambda, &ipring);
						value += hpx_data[ipring];
					}
				}
			}
			mw_data.push_back(value);

		}
	}

	auto db = DBCreate(outfile.c_str(), DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
	std::vector<float> x, y;
	for (int ix = -2 * res; ix <= 2 * res; ix++) {
		const double x0 = double(ix) / double(res);
		x.push_back(x0);
	}
	for (int iy = -res; iy <= res; iy++) {
		const double y0 = double(iy) / double(res);
		y.push_back(y0);
	}

	constexpr int ndim = 2;
	const int dims1[ndim] = { 4 * res + 1, 2 * res + 1 };
	const int dims2[ndim] = { 4 * res, 2 * res };
	const float* coords[ndim] = { x.data(), y.data() };
	const char* coord_names[ndim] = { "x", "y" };
	DBPutQuadmesh(db, "mesh", coord_names, coords, dims1, ndim, DB_FLOAT, DB_COLLINEAR, nullptr);
	DBPutQuadvar1(db, "intensity", "mesh", mw_data.data(), dims2, ndim, nullptr, 0, DB_FLOAT, DB_ZONECENT, nullptr);
	DBClose(db);
	return 0;
}
