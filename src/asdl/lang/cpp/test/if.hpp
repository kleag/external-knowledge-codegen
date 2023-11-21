void if_() {
  if (true);

  if(1) 1;

  if(0) {
    1;
    2;
  }
}

void if_else() {
  if (true);
  else ;

  if(true) 1;
  else 2;

  if(true) {
    1;
    2;
  }
  else {
    3;
    4;
  }
  if(false) {
    1;
  }
  else if (false){
    2;
  }
  else
    3;
}

void decl_in_if() {
  if(int i = 0) i;
}
