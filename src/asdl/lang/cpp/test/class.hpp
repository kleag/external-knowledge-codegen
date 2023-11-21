class fwd;

class empty {};

class non_empty {
  int field;
  non_empty();
};

class with_default_constructor {
  with_default_constructor() = default;
};

class with_deleted_constructor {
  with_deleted_constructor() = delete;
};

class with_ctors {
  with_ctors(int x);
  with_ctors(int x, int y);
  with_ctors(int x, int y, int);
};

class with_ctors_init_list {
  int x;
  int y;
  with_ctors_init_list(int x) : x(x), y(0) {}
};

struct base {
  base() = default;
  base(int);
};

class with_inheritance : base {
  with_inheritance() : base() {}
  with_inheritance(int i) : base(i) {}
};

class with_public_inheritance : public base {
};

class with_protected_inheritance : protected base {
};

class with_private_inheritance : private base {
};

class with_copy_ctor {
  with_copy_ctor(const with_copy_ctor &);
};

class with_explicit_ctor {
  explicit with_explicit_ctor(int);
};

class with_move_ctor {
  with_move_ctor(with_move_ctor &&);
};

class with_named_arguments_ctors {
  with_named_arguments_ctors(const with_copy_ctor & other);
  with_named_arguments_ctors(with_move_ctor && other);
};


class with_destructor {
  ~with_destructor();
};

class with_default_destructor {
  ~with_default_destructor() = default;
};

class with_deleted_destructor {
  ~with_deleted_destructor() = delete;
};

class with_virtual_destructor {
  virtual ~with_virtual_destructor();
};


class access_spec {
  public:
    int i;
  private:
    int j;
  protected:
    int k;
  private:
};


struct S {
  S() { }  // User defined constructor makes S non-POD.
  ~S() { } // User defined destructor makes it non-trivial.
};
void test() {
  const S &s_ref = S(); // Requires a CXXBindTemporaryExpr.
}


// virtual inheritance
struct B { int n; };
class X : public virtual B {};
//class Y : virtual public B {}; // We don't keep info on the order
class Y : public virtual B {};
class Z : public B {};

struct AA : X, Y, Z
{
};
