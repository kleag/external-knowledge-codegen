void for_() {
  for(;;) {}

  for(0;;) {}
  for(;1;) {}
  for(;;2) {}

  for(0;1;2) 3;
  for(0;1;2) {3;4;}
}

void for_decl() {
  for(int i = 0; i; ++i);
  int j;
  for(; int i = j; ++i);
  for(int k; int i = 0;);
}

struct s {};
class c {};
using u = s;
typedef s t;

void for_multi_decl() {
  for(int i = 0, j=1;;);
  for(int i = 0, *j;;);
  for(const int i = 0, *j;;);
  for(int i = 0, * const j = nullptr;;);
  for(int i = 0, j[2] = {1,3};;);
  for(int i = 0, (*j)(float);;);
  for(s i, j;;);
  for(c i, *j, k[2];;);
  for(t i, *j, k[2];;);
  for(u const i = {}, *j, k[2];;);
}

void for_break() {
  for(;;) break;
}

void for_continue() {
  for(;;) continue;
}

void range_for() {
  char a[10];
  for(char b: a) ;
}
