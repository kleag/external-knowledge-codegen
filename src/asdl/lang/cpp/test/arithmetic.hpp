void foo(int x, int y) {
  x + y;
  x - y;
  x * y;
  x / y;
  x % y;

  x << y;
  x >> y;
  x ^ y;
  x & y;
  x | y;

  ~x;

  +x;
  -x;

  ++x;
  x++;
  --x;
  y--;
}

void bar(bool x, bool y) {
  x && y;
  x || y;
  !x;
}
