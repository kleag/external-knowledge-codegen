//int w[] = {1, 2, 3};
int x[2] = {1, 2};
int y[2] = {};
int z[2];
//int w2[][1] = {{1},  {2}, {3}};
int x2[2][3] = {{1,1,1}, {2,2,2}};
int y2[2][3] = {};
int z2[2][3];

void foo() {
  x[0];
  x2[0][3];
}

// vla
void bar(int n, float a[n + 1]);
void bar(int n) {
  typedef int ty[n];
  ty a;
  int b[n];
  int (*c)[n];
}

// templated size
template<int N>
void babar(float a[N]);

template<int N>
void babar() {
  typedef int ty[N];
  ty a;
  int b[N];
  int (*c)[N];
}
