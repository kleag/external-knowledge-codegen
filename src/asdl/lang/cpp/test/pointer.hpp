int * x;
int ** y;
const int * z;
const int * const w = 0;

void * p = (void*)0;

void foo() {
  y = &x;
  x = *y;
}

