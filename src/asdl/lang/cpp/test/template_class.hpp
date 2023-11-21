// Template type
template<typename T>
struct forward_decl;

template<typename>
struct anonymous_forward_decl;

template<typename T, typename S>
struct multiple_params;

template<typename T>
struct decl {
  T member;
  const T * const_ptr_member;
};

template<typename T=int>
struct with_default;

// Template value
template<int T>
struct forward_decl_value;

template<int T, int S>
struct multiple_params_value;

template<int N>
struct decl_value {
  static const int member = N;
};

template<int T=3>
struct with_default_value;


// instantiation
using d = decl<int>;
using dv = decl_value<3>;

// declare variables
decl<float> dd_float;
decl<decl<float>> dd_type;
decl_value<8> dd8;

// specialization
template<>
struct decl<int> {
  int imember;
};

template<>
struct decl_value<5> {
  static const int member = 0;
};

// partial specialization
template<typename T0, typename T1>
struct foo;

template<typename T>
struct foo<T, float> {};

template<typename P>
struct foo<decl<P>, bool> {
  P p;
};

template<typename T, typename P>
struct foo<decl<P>, T> {
  P p;
};

template<typename T, typename P>
struct foo<T, decl<P>> {
  template<typename S>
  P doit(S);
};

template<typename Tp>
struct some_type
{
};

template<typename Tp>
struct some_type<volatile Tp>
{
};

// Parameter pack
template<class... T> struct parameter_pack;
template<int... I> struct iparameter_pack;

template<class... Ts> struct parameter_pack_expansion : parameter_pack<Ts...> {};
parameter_pack_expansion<int, float> *templated_decl;

template<int... Is> struct iparameter_pack_expansion : iparameter_pack<Is...> {};
iparameter_pack_expansion<1, 2> *itemplated_decl;

// Template template

template<template<typename X, X W> class Z> struct Y {};
template<template<typename X, X W> class> struct anonymous_Y {};
template<template<typename X, X> class> struct other_anonymous_Y {};


// Injected template
template<class T>
struct injected {
  injected foo();
  template<class Tp>
  injected<Tp> bar();
};

// Injected template, dependent class ref and integral template
template <typename E> struct RR;

template <unsigned int I, typename H, int = RR<H>::value> struct base;

template <unsigned int I, typename H> struct base<I, H, 1> {
  base() = default;
};

// Parameter pack
template<unsigned int Idx, typename... Elements>
  struct Tuple_impl;

template<unsigned int Idx, typename Head, typename... Tail>
  struct Tuple_impl<Idx, Head, Tail...>
  : public Tuple_impl<Idx + 1, Tail...>
  {
    static Head
    M_head(Tuple_impl<Idx, Head, Tail...>&t) ;


  };
