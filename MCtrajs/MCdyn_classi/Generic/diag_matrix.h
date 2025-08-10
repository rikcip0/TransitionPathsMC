#include<math.h>

using namespace std;

#ifndef DIAG_MATRIX_H
#define DIAG_MATRIX_H

template<class T, int n> class CMatrix{	
  T array[n][n];
 public:
  T* operator[](size_t i){
    return array[i];
  }
  size_t size()const{
    return n;
  }
  void print() {
    for (int i=0;i<n;i++) {
      for (int j=0;j<n;j++) cout<<array[i][j]<<" ";
      cout<<endl;
    }	  
  }
};

typedef CMatrix<long double,NCUT> matrice;

matrice zeromatrix() {
  matrice z;
  for (int i=0;i<NCUT;i++) 
    for (int j=0;j<NCUT;j++) z[i][j]=0;
  return z;
}
matrice idmatrix() {
  matrice z;
  for (int i=0;i<NCUT;i++) 
    for (int j=0;j<NCUT;j++) z[i][j]=(i==j ? 1 : 0);
  return z;
}
matrice per(matrice A,matrice B) {
  matrice z=zeromatrix();
  for (int n1=0;n1<NCUT;n1++)
    for (int n2=0;n2<NCUT;n2++) {
      for (int n=0;n<NCUT;n++) z[n1][n2]+=A[n1][n]*B[n][n2];
    }
  return z;
}


/* auxiliary functions */

inline long double pythag(long double a, long double b){
  return sqrt(a*a+b*b);
}
void tqli(long double * d, long double * e, int n,matrice * z){
  /* input: d[0,...,n-1]: diagonal; e[0,...,n-2]: 1st layer, e[n-1]=0 */
  /* input: z must be the identity matrix */
  /* output: d[0,...,n-1]: eigenvalues; z[][0,...,n-1] eigenvectors */
  int m,l,iter,i,k;
  long double s,r,p,g,f,dd,c,b;
  for(l=0;l<n;l++){                                            // l->l-1
    iter=0;
    do{
      for(m=l;m<n-1;m++){                                      // m->m-1
	dd=fabs(d[m])+fabs(d[m+1]);
	if ((long double)(fabs(e[m])+dd) == dd) break;
      }
      if(m!=0){                                                // m->m-1
	g=(d[l+1]-d[l])/(2.0*e[l]);
	r=pythag(g,1.0);
	g=d[m]-d[l]+e[l]/(g+(g>=0?fabs(r):-fabs(r)));
	s=c=1.0;
	p=0.0;
	for (i=m-1;i>=l;i--) {                                 // i->i-1
	  f=s*e[i];
	  b=c*e[i];
	  e[i+1]=(r=pythag(f,g));
	  if (r == 0.0) {
	    d[i+1] -= p;
	    e[m]=0.0;
	    break;
	  }
	  s=f/r;
	  c=g/r;
	  g=d[i+1]-p;
	  r=(d[i]-g)*s+2.0*c*b;
	  d[i+1]=g+(p=s*r);
	  g=c*r-b;
	  for (k=0;k<n;k++) {                                 // k->k-1
	    f=(*z)[k][i+1];
	    (*z)[k][i+1]=s*(*z)[k][i]+c*f;
	    (*z)[k][i]=c*(*z)[k][i]-s*f;
	  }
	}
	if (r == 0.0 && i >= l) continue;
	d[l] -= p;
	e[l]=g;
	e[m]=0.0;
      }
    } while (m != l);
    }
}
void tred2(matrice * a, int n, long double d[], long double e[])
//Householder reduction of a real, symmetric matrix a[0..n-1][0..n-1]. On output, a is replaced
//by the orthogonal matrix Q effecting the transformation. d[0..n-1] returns the diagonal elements
//of the tridiagonal matrix, and e[0..n-1] the off-diagonal elements, with e[0]=0. Several
//statements, as noted in comments, can be omitted if only eigenvalues are to be found, in which
//case a contains no useful information on output. Otherwise they are to be included.
{
  int l,k,j,i;
  float scale,hh,h,g,f;
  for (i=n;i>=2;i--) {
    l=i-1;
    h=scale=0.0;
    if (l > 1) {
      for (k=1;k<=l;k++)
	scale += fabs((*a)[i-1][k-1]);
      if (scale == 0.0) //Skip transformation.
	e[i-1]=(*a)[i-1][l-1];
      else {
	for (k=1;k<=l;k++) {
	  (*a)[i-1][k-1] /= scale;  // Use scaled a's for transformation.
	  h += (*a)[i-1][k-1]*(*a)[i-1][k-1]; // Form  in h.
	}
	f=(*a)[i-1][l-1];
	g=(f >= 0.0 ? -sqrt(h) : sqrt(h));
	e[i-1]=scale*g;
	h -= f*g;            // Now h is equation (11.2.4).
	(*a)[i-1][l-1]=f-g;         //Store u in the ith row of a.
	f=0.0;
	for (j=1;j<=l;j++) {
	  /* Next statement can be omitted if eigenvectors not wanted */
	  (*a)[j-1][i-1]=(*a)[i-1][j-1]/h; //Store u=H in ith column of a.
	  g=0.0; //Form an element of A  u in g.
	  for (k=1;k<=j;k++)
	    g += (*a)[j-1][k-1]*(*a)[i-1][k-1];
	  for (k=j+1;k<=l;k++)
	    g += (*a)[k-1][j-1]*(*a)[i-1][k-1];
	  e[j-1]=g/h; //Form element of p in temporarily unused element of e.
	  f += e[j-1]*(*a)[i-1][j-1];
	}
	hh=f/(h+h);  //Form K, equation (11.2.11).
	for (j=1;j<=l;j++) { //Form q and store in e overwriting p.
	  f=(*a)[i-1][j-1];
	  e[j-1]=g=e[j-1]-hh*f;
	  for (k=1;k<=j;k++) //Reduce a, equation (11.2.13).
	    (*a)[j-1][k-1] -= (f*e[k-1]+g*(*a)[i-1][k-1]);
	}
      }
    } else
      e[i-1]=(*a)[i-1][l-1];
    d[i-1]=h;
  }
  /* Next statement can be omitted if eigenvectors not wanted */
  d[0]=0.0;
  e[0]=0.0;
  /* Contents of this loop can be omitted if eigenvectors not
     wanted except for statement d[i]=a[i][i]; */
  for (i=1;i<=n;i++) { //Begin accumulation of transformation matrices
    l=i-1;
    if (d[i-1]) {         // This block skipped when i=1.
      for (j=1;j<=l;j++) {
	g=0.0;
	for (k=1;k<=l;k++)       // Use u and u=H stored in a to form PQ.
	  g += (*a)[i-1][k-1]*(*a)[k-1][j-1];
	for (k=1;k<=l;k++)
	  (*a)[k-1][j-1] -= g*(*a)[k-1][i-1];
      }
    }
    d[i-1]=(*a)[i-1][i-1];         // This statement remains.
    (*a)[i-1][i-1]=1.0;              // Reset row and column of a to identity
    for (j=1;j<=l;j++) (*a)[j-1][i-1]=(*a)[i-1][j-1]=0.0; //matrix for next iteration.
  }
}
void mdiag(matrice *a,int n,long double * d){
  /* input: a matrix to be diagonalized */
  /* output: d[0,...,n-1]: eigenvalues; a[][0,...,n-1] eigenvectors */

  long double e[n];
  /* reduction to tridiag matrix */
  tred2(a,n,d,e);
  /* stupid shift to match tqli */
  for (int i=0;i<n-1;i++) e[i]=e[i+1];
  e[n-1]=0; 
  /* diagonalization */
  tqli(d,e,n,a);

}




#endif
