auto x = 1;
const auto cx = 1;
const auto & cxr = 1;
decltype(auto) y = 2;
__auto_type z = 3;

auto foo() {
  return 1;
}

auto lambda = [](auto y) { return y; };

auto bar(int x) {
  return [&x](auto * z) { return *z + x; };
}

typedef auto foobar() -> int;

using barfoo = auto () -> int;

auto trailing_type(int x) -> decltype( x + 1);

struct some {
  auto trailing_type(int x) -> decltype( x + 1);
};

auto other_lambda = [](auto y) -> decltype(y) { return y; };
