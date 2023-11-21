struct foo {

};

void func();

template<class T> class tbar;
template<class T> T tfunc();

struct bar {
  friend struct foo;
  friend void func();
  template<class T>
  friend class tbar;
  template<class T>
  friend T tfunc();
};
