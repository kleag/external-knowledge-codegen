#include <typeinfo>
class foo;
struct bar {};
void foo() {
  typeid(int);
  typeid(foo);
  int a;
  bar b;
  typeid(a);
  typeid(b);
}
