struct bitfields {
  int a : 2;
  int b : 4;
};

struct bitfields_with_init {
  int a : 2;
  int b : 4 = 1;
};

struct bitfields_with_init_and_attrs {
  int a : 2;
  __attribute__ ((aligned)) int b : 4 = 1;
};
