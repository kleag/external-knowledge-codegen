// here we will only handle simple functions

constexpr int function() { return 1; }
constexpr static int static_function() { return 1; }

struct X
{
    constexpr int function() { return 1; }
    constexpr static int static_function() { return 1; }
};

namespace A
{
    constexpr int function() { return 1; }
    constexpr static int static_function() { return 1; }

    struct X
    {
        constexpr int function() { return 1; }
        constexpr static int static_function() { return 1; }
    };

}


