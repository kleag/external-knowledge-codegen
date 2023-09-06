using type = int;

struct X{};
using Y = X;

// FIXME:
// template<class T> class V{};
// template<class T> using U = V<T>;

using ::type;
