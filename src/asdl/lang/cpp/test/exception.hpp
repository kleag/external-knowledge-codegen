void foo() {
  throw;
  throw 1;
}

void bar() {
  try { 1; }
  catch(float f) { 2; }
  catch(int&) { 2; }
  catch(...) { 3; }
}
