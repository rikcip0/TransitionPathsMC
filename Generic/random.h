/*

extern void init_random(int seed,int seed2);
extern double Xrandom(void);
extern double Grandom(void);

*/
#ifndef INCLUDIRANDOM
#define INCLUDIRANDOM

#include <time.h>
#include <math.h>


#ifndef __RANDOM_64__

static int A[4];
static int B[4]={6698,7535,26792,30140};
                          /* 246913578*(2^32+1)=1060485742695258666 */

#define M0   17917        /* 13^13=302875106592253 */
#define M1   13895
#define M2   19930
#define M3   8

#define PW2  32767         /* 2^(15)-1 */

#define rpw2a  9.31322574615478516e-10     /* 2^(-30) */
#define rpw2b  1.11022302462515654e-16     /* 2^(-53) */
#define rpw2c  2.77555756156289135e-17     /* 2^(-55) */

double Xrandom(void)
  {
   A[0]=B[0]*M0;
   A[1]=(A[0]>>15)+B[1]*M0+B[0]*M1;
   A[2]=(A[1]>>15)+B[2]*M0+B[1]*M1+B[0]*M2;
   A[3]=(A[2]>>15)+B[3]*M0+B[2]*M1+B[1]*M2+B[0]*M3;
   B[0]=A[0]&PW2;
   B[1]=A[1]&PW2;
   B[2]=A[2]&PW2;
   B[3]=A[3]&PW2;
   return rpw2a*((B[3]<<15)+B[2])+rpw2b*((B[1]<<8)+(B[0]>>7))+rpw2c;
  }

int init_random(int seed,int seed2)
  {
   if (seed==0) seed=(int)time(NULL);
   B[0]=(2+(seed<<2))&PW2;
   B[1]=(seed>>13^seed2)&PW2;
   B[2]=(seed>>28^seed2>>15)&PW2;
   B[3]=seed2>>30&PW2;
   Xrandom();
   return seed;
  }

#else

#ifdef __LONGLONG_64__
typedef long long integer;
#else
typedef long integer;
#endif

static integer A[2];
static integer B[2]={246913578,987654312};
                                 /* 246913578*(2^32+1)=1060485742695258666 */

#define M0     455329277         /* 13^13=302875106592253 */
#define M1     282074

#define PW2    1073741823       /*** 2^30-1 ***/

#define rpw2a  9.31322574615478516e-10     /* 2^(-30) */
#define rpw2b  1.11022302462515654e-16     /* 2^(-53) */
#define rpw2c  2.77555756156289135e-17     /* 2^(-55) */

double Xrandom(void)
  {
   A[0]=B[0]*M0;
   A[1]=(A[0]>>30)+B[1]*M0+B[0]*M1;
   B[0]=A[0]&PW2;
   B[1]=A[1]&PW2;
   return rpw2a*B[1]+rpw2b*(B[0]>>7)+rpw2c;
  }

void init_random(int seed,int seed2)
  {
   if (seed==0) seed=(int)time(NULL);
   B[0]=((2+(seed<<2))^seed2<<15)&PW2;
   B[1]=(seed>>28^seed2>>15)&PW2;
   Xrandom();
  }

#endif

double Grandom(void)
  {
   static int flag=0;
   static double rho,phi;
   if (flag^=1)
     {
      rho=sqrt(-2*log(Xrandom()));
      phi=2*M_PI*Xrandom();
      return rho*cos(phi);
     }
   else return rho*sin(phi);
  }


#endif
