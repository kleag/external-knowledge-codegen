template<class T>
struct foo {
  using type = typename T::foo::popop;
  typename T::foo::popop var;
};
