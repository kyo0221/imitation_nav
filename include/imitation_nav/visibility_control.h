#ifndef IMITATION_NAV__VISIBILITY_CONTROL_H_
#define IMITATION_NAV__VISIBILITY_CONTROL_H_

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define IMITATION_NAV_EXPORT __attribute__ ((dllexport))
    #define IMITATION_NAV_IMPORT __attribute__ ((dllimport))
  #else
    #define IMITATION_NAV_EXPORT __declspec(dllexport)
    #define IMITATION_NAV_IMPORT __declspec(dllimport)
  #endif
  #ifdef IMITATION_NAV_BUILDING_LIBRARY
    #define IMITATION_NAV_PUBLIC IMITATION_NAV_EXPORT
  #else
    #define IMITATION_NAV_PUBLIC IMITATION_NAV_IMPORT
  #endif
  #define IMITATION_NAV_PUBLIC_TYPE IMITATION_NAV_PUBLIC
  #define IMITATION_NAV_LOCAL
#else
  #define IMITATION_NAV_EXPORT __attribute__ ((visibility("default")))
  #define IMITATION_NAV_IMPORT
  #if __GNUC__ >= 4
    #define IMITATION_NAV_PUBLIC __attribute__ ((visibility("default")))
    #define IMITATION_NAV_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define IMITATION_NAV_PUBLIC
    #define IMITATION_NAV_LOCAL
  #endif
  #define IMITATION_NAV_PUBLIC_TYPE
#endif

#endif  // IMITATION_NAV__VISIBILITY_CONTROL_H_
