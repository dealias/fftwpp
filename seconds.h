#ifndef __seconds_h__
#define __seconds_h__ 1

#include <chrono>

#ifdef _WIN32
#include <Winsock2.h>

#include <time.h>
#include <windows.h>
#include <iostream>

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

struct timezone
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};

// Definition of a gettimeofday function

inline int gettimeofday(struct timeval *tv, struct timezone *tz)
{
// Define a structure to receive the current Windows filetime
  FILETIME ft;

// Initialize the present time to 0 and the timezone to UTC
  size_t __int64 tmpres = 0;
  static int tzflag = 0;

  if (NULL != tv)
    {
      GetSystemTimeAsFileTime(&ft);

// The GetSystemTimeAsFileTime returns the number of 100 nanosecond
// intervals since Jan 1, 1601 in a structure. Copy the high bits to
// the 64 bit tmpres, shift it left by 32 then or in the low 32 bits.
      tmpres |= ft.dwHighDateTime;
      tmpres <<= 32;
      tmpres |= ft.dwLowDateTime;

// Convert to microseconds by dividing by 10
      tmpres /= 10;

// The Unix epoch starts on Jan 1 1970.  Need to subtract the difference
// in seconds from Jan 1 1601.
      tmpres -= DELTA_EPOCH_IN_MICROSECS;

// Finally change microseconds to seconds and place in the seconds value.
// The modulus picks up the microseconds.
      tv->tv_sec = (long)(tmpres / 1000000UL);
      tv->tv_usec = (long)(tmpres % 1000000UL);
    }

  if (NULL != tz)
    {
      if (!tzflag)
        {
          _tzset();
          tzflag++;
        }

// Adjust for the timezone west of Greenwich
      tz->tz_minuteswest = _timezone / 60;
      tz->tz_dsttime = _daylight;
    }

  return 0;
}

#else

#include <sys/time.h>

#endif

namespace utils {

// thread-safe; return number of nanoseconds since initialization
inline double nanoseconds()
{
  static auto begin=std::chrono::steady_clock::now();
  auto end=std::chrono::steady_clock::now();
  auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds> (end-begin);
  return elapsed.count();
}

// thread-safe; return number of seconds since initialization
inline double seconds()
{
  return nanoseconds()*1.0e-9;
}

}

#endif
