void foo() {
  int x;
  bool y = x; // implicit cast
  float z0 = (float)x;
  float z1 = x;
  float z2 = (int)x; // chained cast
  void * p0 = &x; // implicit pointer cast
  void * p1 = (void*)&x; // implicit pointer cast
  const void * p2 = &x;
  const void * p3 = (const void*)&x;
  (void)x;
}

void bar() {
  int(1);

  // value static_cast
  int a;
  static_cast<float>(1);
  static_cast<float>(a);

  // pointer/inheritance static_cast
  struct U{};
  struct V : U{};
  V v;
  U* ptr_u = static_cast<U*>(&v);
  V* ptr_v = static_cast<V*>(ptr_u);
  U& ref_u = static_cast<U&>(v);
  V& ref_v = static_cast<V&>(ref_u);

  // reinterpret_cast
  unsigned long i;
  reinterpret_cast<long *>(&i);
  int buffer[10];
  reinterpret_cast<char*>(&buffer);

  // const_cast reference or pointer
  int x = 0;
  const int& x_const = const_cast<const int&>(x);
  int& x_not_const = const_cast<int&>(x_const);
  const int* ptr_x_const = const_cast<const int*>(&x);
  int* ptr_x = const_cast<int*>(ptr_x_const);
}


struct X { X(int, float); };

X create_X() {
  return X(1, 3.14f); // creates a CXXTemporaryObjectExpr
};

// functional cast to unresolved type
template<class T>
T foo(int x) { return T(x);}
