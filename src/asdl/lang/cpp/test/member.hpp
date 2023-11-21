struct foo {
  int ibar;
  int abar[2];
  void bar(int);
};

auto t = &foo::bar;

typedef int foo::* pointer_to_scalar_member_type;
int foo::* pointer_to_scalar_member_value;
using pointer_to_scalar_member_type_alias = int foo::*;

typedef int foo::* pointer_to_array_member_type[2];
int foo::* pointer_to_array_member_value [2];
using pointer_to_array_member_type_alias = int foo::* [2];

typedef void (foo::* pointer_to_function_member_type)(int);
void (foo::* pointer_to_function_member_value)(int);
using pointer_to_function_member_type_alias = void (foo::*)(int);
