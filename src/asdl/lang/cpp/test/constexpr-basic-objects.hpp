// here we will not handle user-defined constexpr, just basic native types for now.

constexpr int x = 0;
constexpr float y = 0.05f;

constexpr static int static_constexpr = 42; // note: `static constexpr int` will fail the test but is equivalent

struct Struct
{
    constexpr static int static_constexpr = 42; // note: `static constexpr int` will fail the test but is equivalent
};


void function()
{
    constexpr int a = 0;
    constexpr static int b = 0;
}

namespace A {

    constexpr int x = 0;
    constexpr float y = 0.05f;

    constexpr static int static_constexpr = 42; // note: `static constexpr int` will fail the test but is equivalent

    struct Struct
    {
        constexpr static int static_constexpr = 42; // note: `static constexpr int` will fail the test but is equivalent
    };


    void function()
    {
        constexpr int a = 0;
        constexpr static int b = 0;
    }

}

