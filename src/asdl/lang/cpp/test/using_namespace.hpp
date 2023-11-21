
namespace A {}

// names in A are brought in the current scope
using namespace A;

// must work whatever the scope (except class/struct scope):
namespace X
{
    // in another namespace's scope
    using namespace A;
}

namespace Y
{
    // in a function scope
    void function()
    {
        using namespace A;
    }
}

// sub-namespace access
// FIXME:
// namespace A { namespace B { } }
// namespace D
// {
//     // in another namespace's scope
//     using namespace A::B;
//     using namespace ::A::B;
// }

// namespace W
// {
//     // in a function scope
//     void function()
//     {
//         using namespace A::B;
//         using namespace ::A::B;
//     }
// }
