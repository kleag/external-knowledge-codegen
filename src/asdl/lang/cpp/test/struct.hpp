struct fwd;

struct empty {
};

struct some {
  int field;
};

struct nested {
  struct nest {
    int a;
  };
  int b;
};

struct list {
  int val;
  struct list * next;
};

struct const_list {
  int val;
  struct const_list * const next;
};

// This is correctly modeled, but using a generated name
// struct { int x; } anonymous;


// This is correctly modeled, but split in two, as clang does
// struct foo { int x; } inline_decl;

struct nested_with_indirect_field {
  int val;
  struct {
    int indirect;
  };
};


void init() {
  empty e0;
  empty e1 = {};

  some s0;
  some s1 = { 1 };

  nested n0 = { 3 };

  list l0 = { 4, nullptr };

  nested_with_indirect_field nif0 = { 4, { 5 } };
}

void field() {
  list l0 = { 4, nullptr };

  l0.val;
  l0.val = 1;

  l0.next = &l0;

  l0.next->next = nullptr;
}

struct complex_fields {
  float (*((*a) [3]))(bool);
  float b [1];
  char data[];
};

struct struct_hack {
  float a;
  float b [1];
  char data[];
};
