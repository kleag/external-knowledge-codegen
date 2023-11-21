union empty {};

union a {
  int val;
};

struct tagged_enum {
  int choice;
  union {
    int a;
    float b;
  };
};
