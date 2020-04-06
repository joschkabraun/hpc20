#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif


// coefficients in the Taylor series expansion of sin(x)
static constexpr double c2  = -1/((double)2);
static constexpr double c3  = -1/(((double)2)*3);
static constexpr double c4  =  1/(((double)2)*3*4);
static constexpr double c5  =  1/(((double)2)*3*4*5);
static constexpr double c6  = -1/(((double)2)*3*4*5*6);;
static constexpr double c7  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8);
static constexpr double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
static constexpr double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + c9*x^9 + c11*x^11
// cos(x) = 1 + c2*x^2 + c4*x^4 + c6*x^6 + c8*x^8 + c10*x^10

void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}

// This is the implementation of the extension of sin4_taylor to the entire period of sin.
// By the given hint, one gets the identity sin(theta + pi/2) = cos(theta). With this
// identity one can use a Taylor series expansion of cos around 0 to compute sin(x) for
// values in the range from pi/4 to 3*pi/4. Via symmetry of sin (i.e. sin(x)=-sin(-x)),
// one has now a way to calculate sin(x) for x in the range from -3*pi/4 to 3*pi/4.
// As sin(x)=sin(pi-x) we can model values of x from 3*pi/4 to pi via sin(x) in the range of
// 0 to pi/4. As sin(x) = sin(-pi-x), we can model values for x from -pi to 3*pi/4 via sin(x)
// in the range from -pi/4 to 0.
void sin4_taylor_all(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x1 = x[i];
    double s;
    
    // transfrom x if it lies in [-pi, -3*pi/4] or [3*pi/4,pi] as described above
    if ((-M_PI <= x1) & (x1 <= -3*M_PI/4)) {
      x1 = - M_PI - x1;
    } else if ((3*M_PI/4 <= x1) & (x1 <= M_PI)) {
      x1 = M_PI - x1;
    } 

    if ((-M_PI/4 <= x1) & (x1 <= M_PI/4)) { // [-pi/4, pi/4]
      double x2  = x1 * x1;
      double x3  = x1 * x2;
      double x5  = x3 * x2;
      double x7  = x5 * x2;
      double x9  = x7 * x2;
      double x11 = x9 * x2;

      s = x1;
      s += x3  * c3;
      s += x5  * c5;
      s += x7  * c7;
      s += x9  * c9;
      s += x11 * c11;
    } else if ((M_PI/4 <= x1) & (x1 <= M_PI/2)) {
      x1 = x1 - M_PI/2;
      double x2  = x1 * x1;
      double x4  = x2 * x2;
      double x6  = x4 * x2;
      double x8  = x6 * x2;
      double x10 = x8 * x2;

      s = ((double) 1);
      s += c2  * x2;
      s += c4  * x4;
      s += c6  * x6;
      s += c8  * x8;
      s += c10 * x10;
    } else if ((-M_PI/2 <= x1) & (x1 <= -M_PI/4)) {
      x1 = -x1 - M_PI/2; 
      double x2  = x1 * x1;
      double x4  = x2 * x2;
      double x6  = x4 * x2;
      double x8  = x6 * x2;
      double x10 = x8 * x2;

      s = ((double) 1);
      s += c2  * x2;
      s += c4  * x4;
      s += c6  * x6;
      s += c8  * x8;
      s += c10 * x10;

      s = s*(-1);
    } 

    sinx[i] = s;
  }
}

void sin4_taylor(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x1  = x[i];
    double x2  = x1 * x1;
    double x3  = x1 * x2;
    double x5  = x3 * x2;
    double x7  = x5 * x2;
    double x9  = x7 * x2;
    double x11 = x9 * x2;

    double s = x1;
    s += x3  * c3;
    s += x5  * c5;
    s += x7  * c7;
    s += x9  * c9;
    s += x11 * c11;
    sinx[i] = s;
  }
}

void sin4_intrin(double* sinx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)
  __m256d x1, x2, x3;
  x1  = _mm256_load_pd(x);
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);

  __m256d s = x1;
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3 )));
  _mm256_store_pd(sinx, s);
#elif defined(__SSE2__)
  constexpr int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);

    __m128d s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}

// this is an implementation for a Talor expansion for sin(x)
// over the full period from -pi to pi. It uses the same tricks
// as sin4_taylor_all() (cf. description of this function).
void sin4_vector_all(double* sinx, const double* x) {
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x4, x6, x8, x10;
  x1  = Vec4::LoadAligned(x); 
}

void sin4_vector(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x3, x5, x7, x9, x11;
  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x3  = x1 * x2;
  x5  = x3 * x2;
  x7  = x5 * x2;
  x9  = x7 * x2;
  x11 = x9 * x2;

  Vec4 s = x1;
  s += x3  * c3 ;
  s += x5  * c5;
  s += x7  * c7;
  s += x9  * c9;
  s += x11 * c11;
  s.StoreAligned(sinx);
}

double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i]));
  return error;
}

int main() {
  Timer tt;
  long N = 1000000;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector = (double*) aligned_malloc(N*sizeof(double));
  for (long i = 0; i < N; i++) {
    x[i] = (drand48()-0.5) * M_PI/2; // [-pi/4,pi/4]
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;
  }

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref+i, x+i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor(sinx_taylor+i, x+i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin(sinx_intrin+i, x+i);
    }
  }
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_intrin, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_vector(sinx_vector+i, x+i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_vector, N));

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(sinx_vector);
}

