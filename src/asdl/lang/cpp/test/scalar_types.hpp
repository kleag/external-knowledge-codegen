char c;
unsigned char uc;
signed char sc;

short s;
unsigned short us;

int i;
unsigned int ui;

long l = 1l;
unsigned long ul = 1ul;

long long ll = 1ll;
unsigned long long ull = 1ull;

float f = 1.2f;
double d = 1.2;
long double ld =1.2L;

__int128 i128;
//_Complex cc;
_Complex float cf = 2.5fi;
_Complex double cd = 2.5i;
_Complex long double cld = 2.5Li;

__float128 f128 = 1.2q;

// bit ints
_BitInt(3) b3;
unsigned _BitInt(3) u3;
using i33 = _BitInt(33);
using u33 = unsigned _BitInt(33);
