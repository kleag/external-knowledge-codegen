// Simplest namespace
namespace A { }

// Namespaces can contain other namespaces
namespace A
{
    namespace B
    {

    }
}

// Namespaces can contain declarations
namespace A
{
    int x = 42;
    void function() {}
    class X {};
}

namespace A
{
    namespace B
    {
        int x = 42;
        void function() {}
        class X{};
    }
}

// Namespaces can be anonymous (any object or function in it is isolated and static to the translation unit)
namespace {}

// // Anonymous namespaces can contain declarations, like any namespaces
namespace
{
    int actually_static = 42;
    void function_static() {}
    class translation_unit_X{};
}

// // Anonymous namespaces can appear inside a named namespace
namespace A
{
    namespace
    {
        int actually_static = 42;
        void function_static() {}
        class translation_unit_X{};
    }
}

// Inline namespaces from c++11
inline namespace D {
}
