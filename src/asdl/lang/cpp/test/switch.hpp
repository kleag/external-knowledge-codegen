void switch_() {
  switch(0) {
  }

  switch(0) {
    case 0:
      1;
  }

  switch(0) {
    case 0:
      1;
      break;
  }

  switch(0) {
    case 0:
      1;
      break;
    case 2:
      2;
  }

  switch(0) {
    case 0:
    case 2:
      2;
  }

  switch(0) {
    case 0:
      break;
    default:
      2;
  }
}

void duff_device(short* to, short* from, int count) {
  int n = (count + 7) / 8;
  switch (count % 8) {
  case 0:
    do {
      *to = *from++;
    case 7:
      *to = *from++;
    case 6:
      *to = *from++;
    case 5:
      *to = *from++;
    case 4:
      *to = *from++;
    case 3:
      *to = *from++;
    case 2:
      *to = *from++;
    case 1:
      *to = *from++;
    } while (--n > 0);
  }
}

void range(int n) {
  switch(n) {
    case 1 ... 10:
      break;
  }
}
