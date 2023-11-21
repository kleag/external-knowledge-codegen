class foo {
  operator int();
  operator int() const;
  inline operator float() {
    return {};
  }

  foo operator+(const foo &) const;
};
