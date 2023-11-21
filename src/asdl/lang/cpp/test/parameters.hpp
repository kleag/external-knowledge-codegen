void empty();
void empty() {
}

// NOTE: parsed as above by clang.
// void empty_void(void);
// void empty_void(void) {
// }


void foo(int x);
void foo(int x) {
}

void bar(const int x, float *y) {
}

void variadic(int x, ...) {
}

void fn_ptr(int (*x)(float)) {
}

void array_param(int x[2], int y[]) {
}

void default_param(int x = 0) {
}

void rref(int &x);
void cref(const int &x);
void lref(int &&x);

