enum {
};

enum {
  A
};

enum {
  B,
  C
};

enum {
  D = 12,
  E,
  F = 14
};

enum named {
};

enum renamed {
  Some
};

// Usage
renamed xxx;
renamed yyy = Some;

// Note: the following definitions will not result in exactly the same code generated,
// they will look like an enum definition followed by an object declaratoin of that enum's type.
// TODO: support for types definition in object definitions, maybe by checking `TagType::isBeingDefined()`?
// FIXME: commented examples below because the output is equivalent but not the same.

// enum {
//   x, y, z,
// } object;


// enum with_name {
//   u, v, w,
// } object_with_typed_name;

