void foo() {
  1, 2;
  1 ? 2 : 3;
  sizeof(int);
  sizeof(1);
  sizeof(1 + 3);
  alignof(int);
  alignof(1);
  alignof(1 + 3);
  (1 + 2) * 3;
}

int expr_stmt() {
  return ({int X = 4; X;});
}

void conditional_with_opitted_operands(int x, int y) {
  x ? : y;
}

void predefined() {
  __PRETTY_FUNCTION__;
  __func__;
    __FUNCTION__;
}

struct titi { int A; int B;};
struct toto {int a; int b; int c[10]; int d[3][4]; struct titi e;};

void offset() {
  __builtin_offsetof(struct toto, b);
  __builtin_offsetof(struct toto, c[2]);
  __builtin_offsetof(struct toto, d[2][1]);
  __builtin_offsetof(struct toto, e.A);
}
