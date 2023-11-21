struct foo {
  foo* get() { return this;}
  foo* implicit() { return get(); }
  foo* explicit_() { return this->get(); }
};
