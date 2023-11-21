template<typename T>
void foo(int);

template<typename>
void anonymous_foo(int);

template<class T>
void footix(int);

template<typename T>
void foo_with_body(int) {
  return;
}

template<typename T>
void foo(T);
template<typename T>
void foo(const T&);
template<typename T>
void foo(T&&);

template<int I, typename T>
T foot(T x) {
  return x;
}

template<typename T>
T foo(const T & x) {
  return static_cast<T>(x);
}

// instantiation
void bar() {
  // implicit are converted to explicit in the AST
  // foot<1213>(1);

  // explicit
  foot<101010, float>(1.1f);
}


// specialization

// parameter pack
template <typename... Ts> void packs(Ts... a) {
}

