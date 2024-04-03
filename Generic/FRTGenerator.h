#ifndef FRTGenerator
#define FRTGenerator

#define FNORM (2.3283064365e-10)
#define RANDOM ((_ira[_ip++] = _ira[_ip1++] + _ira[_ip2++]) ^ _ira[_ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)

unsigned int myrand, _ira[256];
unsigned char _ip, _ip1, _ip2, _ip3;

unsigned int randForInit(void)
{
  unsigned long long int y;

  y = myrand * 16807LL;
  myrand = (y & 0x7FFFFFFF) + (y >> 31);
  myrand = (myrand & 0x7FFFFFFF) + (myrand >> 31);
  return myrand;
}

void initRandom(void)
{
  int i;

  _ip = 128;
  _ip1 = _ip - 24;
  _ip2 = _ip - 55;
  _ip3 = _ip - 61;

  for (i = _ip3; i < _ip; i++)
  {
    _ira[i] = randForInit();
  }
}

float gaussRan(void)
{
  static int iset = 0;
  static float gset;
  float fac, rsq, v1, v2;

  if (iset == 0)
  {
    do
    {
      v1 = 2.0 * FRANDOM - 1.0;
      v2 = 2.0 * FRANDOM - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 1;
    return v2 * fac;
  }
  else
  {
    iset = 0;
    return gset;
  }
}

#endif