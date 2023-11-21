// member initialization in declaration

int new_value();

struct A {};
A* new_A();

// Pre-c++ 11 way of initializing members
class with_member_initializer_list
{
    int a;
    int b;

    A* pa;
    A* pb;

    struct Impl {};
    Impl* impl;
    Impl* null;

    with_member_initializer_list()
        : a(0)
        , b(0)
        , pa(new_A())
        , pb(new A)
        , impl(new Impl)
        , null(nullptr)
    {}
};

// post-C++11 default member initialization
// note: this case will not generate `CXXDefaultInitExpr` nodes
class no_user_defined_constructor
{
    int a = 0;
    int b = new_value();

    A* pa = new_A();
    A* pb = new A;

    struct Impl {};
    Impl* impl = new Impl;
    Impl* null = nullptr;
};

// note: this case will not generate the same node-tree as without a constructor
// and will add `CXXDefaultInitExpr` nodes
class with_user_defined_constructor
{
    with_user_defined_constructor() {}
    with_user_defined_constructor(int value) : a(value), impl(nullptr) {}

    int a = 0;
    int b = new_value();

    A* pa = new_A();
    A* pb = new A;

    struct Impl {};
    Impl* impl = new Impl;
    Impl* null = nullptr;
};
