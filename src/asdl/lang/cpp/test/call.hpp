void call_builtin() {
  __builtin_trap();
}

void foo(int, float);
void call_args(int i, float f) {
  return foo(i, f);
}

struct s {
  s();
  s(int);
  void foo() const;
};

void call_method(const s & obj) {
  (&obj)->foo();
  return obj.foo();
}

void call_constructor() {
  s a;
  s b(1);
  s c{};
  s d{1};
  s e = 1;
}
