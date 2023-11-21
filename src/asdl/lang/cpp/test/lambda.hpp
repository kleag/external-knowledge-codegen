void no_capture() {
  auto x = [](){};
  auto y = [](int a, float b){};
  auto z = [](int a){ return a;};
}

void with_capture(int X, int &Y) {
  auto x = [X](){ return X;};
  auto x_ref = [&X](){ return X;};
  auto y = [Y](){ return Y;};
  auto y_ref = [&Y](){ return Y;};
  auto z = [X,&Y](){ return X, Y;};
  auto z_p = [&X,Y](){ return X, Y;};
}


template <typename... Ts> void test(Ts... a) {
  struct V {
    void f() {
      // [this] {}; parsed but dumped with ()
      // [*this] {}; parsed but dumped with ()
      [this] () {};
      [*this] () {};
    }
  };
  int b; int c;
  []() {};
  [](int a, ...) {};
  [a...] () {};
  [a...,b] () {};
//  [=] {};
//  [=] { return b; };
//  [&] {};
//  [&] { return c; };
  [b, &c] { return b + c; };
//  [a..., x = 12] () {};
  //[]() constexpr {};
//  []() mutable {};
  []() noexcept {};
  []() -> int { return 0; };
  [] [[noreturn]] () {};
}

