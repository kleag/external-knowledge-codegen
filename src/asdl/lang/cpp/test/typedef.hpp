typedef int simple;
typedef int * simple_ptr;
typedef const float qualified_simple;
typedef const int * simple_ptr_to_qualified;
typedef const int * const simple_qualified_ptr_to_qualified;

typedef int array[2];
typedef int matrix[2][3];
typedef int three_d[2][3][4];

typedef int* array_of_pointer[5];
typedef int (*pointer_to_array)[6];

typedef int (*function_pointer)(float);
typedef int (function_proto)(float, bool);
typedef int (*(array_of_functions[8]))(float, bool);

struct foo {};
typedef foo record;

typedef int (&function_ref)(float);

typedef const char (&char_array)[10];

simple user;

typedef long long my_int64;
typedef my_int64 my_int64;

template<class T>
struct bar {
  typedef T type;
};

template<class T, int N>
struct babar {
  typedef T type[N];
};

template<class T>
struct outer {
  template<class P>
  struct inner {
    typedef T type_outer;
    typedef P type_inner;
    T outer_decl;
    P inner_decl;
    void foo(T, P);
  };
};
