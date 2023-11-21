void foo() throw();
#if  __cplusplus < 201703L
void bar() throw(int, float);
#endif

void foobar() noexcept;

void foobar() noexcept(1 + 2 != 0);
