#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

const double infinity = 1000000.;

// Xor operation
bool xor_ab(int a, int b)
{
	return (((a > 0) && (b == 0)) || ((a == 0) && (b > 0)));
}

// Checks if double is epsilon close to zero (used only for things that are meant to be zeros)
bool is_zero(double a, double epsilon = 0.00001)
{
	return ((a > -epsilon) && (a < epsilon));
}

// Sum logs of probs instead of Mult probs
double probability(const double noise, int pos, const int ch_h, int ch_w, int** alphabet_img, int** img)
{
	double prob = 0.;
	for (int x = pos - ch_w; x < pos; ++x)
		for (int y = 0; y < ch_h; ++y)
			prob += (xor_ab(img[x][y], alphabet_img[x - pos + ch_w][y]) ?
				(is_zero(noise) ? -infinity : log(noise)) :
				(is_zero(1. - noise) ? -infinity : log(1. - noise)));
	return prob;
}

// Function which generates letters from the input logs
int generate(int const alphabet_size, double* f, bool show=false)
{
	double* probs = new double[alphabet_size];

	// Separate infinite values from others
	for (int i = 0; i < alphabet_size; ++i)
		probs[i] = ((f[i] < -infinity + 1.) ? 0. : f[i]);

	// Finds the minimal value as a common denominator for all the powers
	double min = infinity;
	for (int i = 0; i < alphabet_size; ++i)
		if (probs[i] < min)
			min = probs[i];

	// Subtracts that minimal value
	for (int i = 0; i < alphabet_size; ++i)
		probs[i] += (is_zero(probs[i]) ? 0. : -min);

	// Takes exp from powers to convert logs to actual unnormalized probabilities
	for (int i = 0; i < alphabet_size; ++i)
		probs[i] = (is_zero(probs[i]) ? 0. : exp(probs[i]));

	// Summing up all values
	double sum = 0.;
	for (int i = 0; i < alphabet_size; ++i)
		sum += probs[i];
	
	// Normalizing probabilities
	for (int i = 0; i < alphabet_size; ++i)
		probs[i] /= sum;

	// Generating output
	int result = -1;
	double r = (double)rand() / RAND_MAX;
	double p = 0.;
	while ((p <= r) && (result != alphabet_size - 1))
	{
		result += 1;
		p += probs[result];
	}

	delete[] probs;

	return result;
}

// Sum logs of probs instead of Mult probs
int* solve(int &n,
	       const double noise,
	       int const alphabet_size,
	       const int ch_height,
	       const int ch_width,
	       double* freq,
	       int* alphabet_width,
	       const int height,
	       const int width,
	       int*** alphabet_img,
	       int** img)
{
	// Initialize fs
	double*** fs = new double** [width + 1];
	for (int p = 0; p < width + 1; ++p)
	{
		fs[p] = new double* [alphabet_size]();
		for (int c = 0; c < alphabet_size; ++c)
			fs[p][c] = new double[alphabet_size]();
	}

	// Initialize f
	double** f = new double* [width + 1];
	for (int p = 0; p < width + 1; ++p)
		f[p] = new double[alphabet_size]();

	// f[0][k_0] = p(k_0) = p(k_0 | " ")
	for (int c = 0; c < alphabet_size; ++c)
	{
		f[0][c] = is_zero(freq[(alphabet_size - 1) * alphabet_size + c]) ? -infinity : log(freq[(alphabet_size - 1) * alphabet_size + c]);
		fs[0][c][alphabet_size - 1] = is_zero(freq[(alphabet_size - 1) * alphabet_size + c]) ? -infinity : log(freq[(alphabet_size - 1) * alphabet_size + c]);
	}

	// Initialize k taken for f[i]
	int** k = new int* [width + 1]();
	for (int p = 0; p < width + 1; ++p)
		k[p] = new int[alphabet_size]();

	// k[0][c] = -1 for safety
	for (int c = 0; c < alphabet_size; ++c)
		k[0][c] = -1;

	// Calculate all f for i in [1, width - 1]
	for (int p = 1; p < width; ++p)
	{
		for (int c = 0; c < alphabet_size; ++c)
		{
			double prob_max = -infinity;
			int k_max = -1;
			// Find max log prob for all possible prev letters
			for (int c_ = 0; c_ < alphabet_size; ++c_)
			{
				int j = p - alphabet_width[c_];
				if (j >= 0)
				{
					double prob_cur = 0.;
					prob_cur += is_zero(freq[c_ * alphabet_size + c]) ? -infinity : log(freq[c_ * alphabet_size + c]);
					prob_cur += probability(noise, p, ch_height, alphabet_width[c_], alphabet_img[c_], img);
					prob_cur += f[j][c_];
					fs[p][c][c_] = prob_cur;
					if (prob_cur > prob_max)
					{
						prob_max = prob_cur;
						k_max = c_;
					}
				}
			}
			// Save max found
			f[p][c] = prob_max;
			k[p][c] = k_max;
		}
	}

	// Calculate f[width][alphabet_size - 1]
	double prob_max = -infinity;
	int k_max = -1;
	// Find max log prob for all possible prev letters
	for (int c_ = 0; c_ < alphabet_size; ++c_)
	{
		int j = width - alphabet_width[c_];
		if (j >= 0)
		{
			double prob_cur = 0.;
			prob_cur += is_zero(freq[c_ * alphabet_size + alphabet_size - 1]) ? -infinity : log(freq[c_ * alphabet_size + alphabet_size - 1]);
			prob_cur += probability(noise, width, ch_height, alphabet_width[c_], alphabet_img[c_], img);
			prob_cur += f[j][c_];
			fs[width][alphabet_size - 1][c_] = prob_cur;
			if (prob_cur > prob_max)
			{
				prob_max = prob_cur;
				k_max = c_;
			}
		}
	}
	// Save max found
	f[width][alphabet_size - 1] = prob_max;
	k[width][alphabet_size - 1] = k_max;

	// Initialize res
	int* res = new int[width + 1]();
	n = 0;

	// Generating answer
	int pos_cur = width;
	int ch_cur = generate(alphabet_size, fs[width][alphabet_size - 1], true);
	while (pos_cur > 0)
	{
		res[n] = ch_cur;
		n++;
		pos_cur -= alphabet_width[ch_cur];
		if (pos_cur > 0)
			ch_cur = generate(alphabet_size, fs[pos_cur][ch_cur]);
	}

	delete[] k;
	for (int p = 0; p < width + 1; ++p)
		delete[] f[p];
	delete[] f;
	for (int p = 0; p < width + 1; ++p)
	{
		for (int c = 0; c < alphabet_size; ++c)
			delete[] fs[p][c];
		delete[] fs[p];
	}
	delete[] fs;

	return res;
}

void usage(char* s){
    std::cout<<"Usage: "<<s<<" <noise level> <image>\n";
}

int main(int argc, char* argv[])
{
	srand(time(NULL));
	int const alphabet_size = 27;
	fstream file;
	file.open("freq.txt", ios::in);

	const int ch_height = 28;
	const int ch_width = 28;

	char alphabet[] = "abcdefghijklmnopqrstuvwxyz1";

	// Freq
	double* freq = new double[alphabet_size * alphabet_size];
	for (int i = 0; i < alphabet_size * alphabet_size; ++i)
		file >> freq[i];

	file.close();

	// Alphabet imgs
	string folder = "alphabet/";
	string suffix = ".png";
	int* alphabet_width = new int[alphabet_size];
	int*** alphabet_img = new int** [alphabet_size];
	for (int c = 0; c < alphabet_size; ++c)
	{
		Mat image;
		string name(1, alphabet[c]);
		image = imread(folder + name + suffix, IMREAD_UNCHANGED);
		alphabet_width[c] = image.size().width;
		alphabet_img[c] = new int * [ch_width];
		for (int x = 0; x < ch_width; ++x)
		{
			alphabet_img[c][x] = new int[ch_height];
			for (int y = 0; y < ch_height; ++y)
			{
				alphabet_img[c][x][y] = int(image.at<uchar>(y, x));
			}
		}
	}

	//fstream input;
	//input.open("input.txt", ios::in);
    double noise;
    if(argc!=3){
        usage(argv[0]);
        return 0;
    }
    try{
    	noise  = std::stod(argv[1]);
    } 
    catch(const std::invalid_argument& ia){
        usage(argv[0]);
        return 0;
    }

    // Input img
	string fn = argv[2];
	Mat image;
	image = imread(fn, IMREAD_UNCHANGED);

	const int height = image.size().height;
	const int width = image.size().width;

	// Get array from Mat
	int** img = new int* [width];
	for (int x = 0; x < width; ++x)
	{
		img[x] = new int[height];
		for (int y = 0; y < height; ++y)
			img[x][y] = int(image.at<uchar>(y, x));
	}
	int n = 0;
	int* res = solve(n, noise, alphabet_size, ch_height, ch_width, freq, alphabet_width, height, width, alphabet_img, img);
	
	cout << "Results" << endl;
	
	for (int c = n - 1; c >= 0; --c)
	{
		cout << ((alphabet[res[c]] != '1') ? alphabet[res[c]] : ' ');
	}
	cout << endl;

	waitKey(0);
	return 0;
}
