void basic_jump() {

from:
  1;
  goto from;

}

void jump_loop() {
  while(1)
    goto target;
target:
  2;
}

void local_label() {
foo:
  void *ptr;
  ptr = &&foo;
  goto *ptr;
}
