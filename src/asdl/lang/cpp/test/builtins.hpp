void choose() {
  __builtin_choose_expr(1, 2, 3);
}

void bitcast() {
  __builtin_bit_cast(float, 1);
}
