// tests for usage of `using` to import names from namespaces
// FIXME: using types is not supported yet
struct X{};

namespace A
{
    int object = 0;
    struct Type{};
    void function(int) {}
    void function(X) {}
    void function(Type) {}

    namespace B
    {
        int object = 0;
        struct Type{};
        void function(int) {}
        void function(X) {}
        void function(Type) {}
    }
}

// import specific names from namespace
void function_specific_names()
{
    // from relative namespace
    {
        using A::object;
        using A::function;
        // using A::Type;
        function(object);
        function(X());
        // function(Type{});
    }

    // from absolute namespace
    {
        using ::A::object;
        using ::A::function;
        // using ::A::Type;
        function(object);
        function(X());
        // function(Type{});
    }

    // from named sub-namespace
    {
        using A::B::object;
        using A::B::function;
        // using A::B::Type;
        function(object);
        function(X());
        // function(Type{});
    }

    // from absolute named sub-namespace
    {
        using ::A::B::object;
        using ::A::B::function;
        // using ::A::B::Type;
        function(object);
        function(X());
        // function(Type{});
    }
}

// import names from parent type into current type interface/scope
class TypeA
{
protected:
    struct Type {};
    TypeA(int) {}
    TypeA(X) {}
    void function() {}
    void function(X) {}
    void function(Type) {}
    int member_data = 0;
};

class TypeB : private TypeA
{
public:
    // using TypeA::TypeA;         // import multiple constructors, make them public FIXME: this needs to work
    using TypeA::function;      // import multiple functions, make them public
    using TypeA::member_data;   // make data member public
    // using TypeA::Type;          // make inner-type public FIXME: this needs to work
};
