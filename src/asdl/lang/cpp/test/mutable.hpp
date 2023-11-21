
struct X{};

class Class
{
    mutable int mut_a;
    mutable X mut_x;

    // FIXME: reactivate the examples below once comparison is done through ast, not syntax
    // These below should work but the `mutable` keyword will be moved at the first place
    // int mutable mut_b;
    // X mutable mut_y;
};

