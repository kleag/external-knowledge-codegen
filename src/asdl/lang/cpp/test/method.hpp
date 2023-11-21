struct foo {
  void bar();
  void bar(int i);
  void bar(int i) const;
  void babar(int i) &&;
  void babar(int i) &;
  void babar(int i) const &;
  virtual void bar(int i, int j);
  virtual void bar(int i, int j, int k) = 0;
  virtual void bar(float) final;

  void barbare() noexcept;
  void barbare(int) throw();
  void barbare(int, int) noexcept(false);

  void barbant(int, ...);

  inline void barbiche();

  static void barbie();
};

struct foofoo : foo {
  virtual void bar(int i, int j, int k) override;
};
