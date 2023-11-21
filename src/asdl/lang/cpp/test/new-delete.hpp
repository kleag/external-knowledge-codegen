#include <new>

int * default_constructor_scalar() {
  return new int();
}

int ** default_constructor_pointer() {
  return new int*();
}

int * default_constructor_array() {
  return new int [12];
}

int * constructor_scalar() {
  return new int(1);
}

void constructor_placement() {
  int buffer;
  new (&buffer) int(1);
}

void constructor_array_placement() {
  int buffer[3];
  new (&buffer[0]) int [3];
}

void constructor_array_placement_with_initializer() {
  int buffer[3];
  new (&buffer[0]) int [3] {0, 5, 9};
}

int ** constructor_pointer() {
  return new int*(nullptr);
}

int * constructor_array() {
  return new int [3];
}

int * constructor_array_with_initializer() {
  return new int [3]{1,2,3};
}

void delete_pointer(int * p) {
  delete p;
}

void delete_array(int * p) {
  delete [] p;
}
