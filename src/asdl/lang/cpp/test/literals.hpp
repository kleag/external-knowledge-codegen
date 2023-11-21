void c() {
  "";
//  "\x10"; Clang already does the parsing of the hex form
  "\001";
  "\r";
  "string literal";

  'c';
  '\0';
  '\7';

  1;
  1l;
  1u;
  1ul;
  1ll;
  1ull;
//  0xFF; Clang doesn't keep the hexadecimal form

  0.12;
  1.12e+10; // there's a lot of variant here

  0.12f;
  1.12e-10f;

  0.12l;
  1.12e+10l;
}

void cxx() {
  true;
  false;
  nullptr;
}

double operator""_Z(long double);

auto x = 1._Z;
auto y = 1.2_Z;
