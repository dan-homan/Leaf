EXchess v7.97 (beta), A chess playing program 
Copyright (C) 1997-2017 by Daniel C Homan, dchoman@gmail.com

=> For updates see: https://sites.google.com/site/experimentalchessprogram/

This is "beta" software, meaning it has not been extensively tested on 
a variety of systems and may still contain bugs.  Use at your own risk.

-------------------------------------------------
License Information
-------------------------------------------------

This software comes with ABSOLUTELY NO WARRANTY of any kind.  

This program is freeware, distributed under the GNU public
license.  See file "license.txt" for details.  I ask only that you 
e-mail me your impressions/suggestions so that I may 
improve future versions.

*** NOTE: See notes on the source code version (below) for 
discussion of parts of this program that do not fall under the 
GNU public license and retain their own copyrights. 

-------------------------------------------------
A few General Notes
-------------------------------------------------

The file "search.par" along with the opening book, 
should be in the directory from which EXchess is being run. 

Values in the "search.par" file can be used to modify how 
EXchess thinks. The THREADS parameter controls how many
processors EXchess will use during the search; however,
this can be overridden by the 'cores' command from xboard
compatible interfaces. Do not exceed the number of 
processors on your machine. The maximum value allowed by
the program is 32 at this time.

To use the endgame tablebases, put the endgame tablebase path 
and cache size in the "search.par" file.

EXchess has a logging function which can be turned on by 
setting the relevant parameters in the "search.par" file. 
When logging is "on", a new log file is created each time
the program is started.  The "MAX_LOGS" number is the maximum number 
of log files that can be created.  No new log files will
be created beyond this number - you must delete the old log 
files first.

The opening book released with EXchess was built from an 
collection of PGN games downloaded from the CCRL 40/40 testing
results of top engines: http://www.computerchess.org.uk/ccrl/4040/
concatenated with a very large number of EXchess self-play games. 
The included opening book should work on most systems, 
but if not, see the notes below on how to build your own opening 
book with the source code version.  

-------------------------------------------------
Notes on Included Executable Versions (Binaries)
-------------------------------------------------

For convenience, a few pre-built versions of EXchess are 
included in the distribution.  These are for 32 bit Windows, 
64 bit Windows, and Mac OSX.  They are labeled appropriately.  
The pre-built versions include Namilov Tablebase support. 

These are all built on my home Linux or Mac computers, either 
directly or using cross-compilation.  I have only tested 
them on one or two different systems, so if they don't work
well for you, I recommend simply building from source (see below).

If you wish to use the pre-built versions, find the correct 
one for your system and delete the rest.  In the notes
below, I assume the program is named "EXchess_console", 
so you may rename it if you wish.

-------------------------------------------------
Notes on the console version for xboard/winboard.
-------------------------------------------------
 - Once compiled (see below) "EXchess_console" will be a text-based 
   interface version of the program.

 - This version has a serviceable text interface or it can be run with
   Tim Mann's Xboard (Unix/Linux) or Winboard.  EXchess should detect
   that it is running under xboard or winboard, but to make sure, you 
   can include the 'xb' command-line option if you choose...

   winboard -fcp "c:\directory\EXchess_console xb" 

   or

   xboard -fcp "/directory/EXchess_console xb" 

 - To run EXchess in plain text mode, just type "EXchess_console" in the 
   EXchess directory

 - There is an additional command line option "hash", you can set the
   hash size in megabytes like

   ./EXchess_console hash 256      (for a 256 megabyte hash file, for example)

   If this option is used with xboard or winboard, use the line

   winboard -fcp "c:\directory\EXchess_console xb hash 256"

   or
  
   xboard -fcp "/directory/EXchess_console xb hash 256"   

   The default hash size is set to 256 megabytes in search.par and
   can be modified there as well. 

   Note: Don't set the hash table size larger than about one-half of
   the available RAM. Doing so may cause swapping to the hard disk and 
   slow the program down considerably.  

 - Some basic help is available by typing "help" at the command prompt.

 - The file "wac.epd" is a testsuite and can be run by the "testsuite"
   command inside the program or directly from the command line:

   ./EXchess_console hash 256 test wac.epd wac_results.txt 5

   Which will run the wac.epd test at 5 seconds per position and record
   the results into the file wac_results.txt...  For the command line
   "test" option, the hash command must be specified first as indicated
   above.

 - The 'build' command lets you make your own opening book out of a pgn
   text file.  It requires 1-2 times the size of the pgn file in 
   temporary storage space on the disk.  The 'build' command can currently
   handle pgn files up to 60 MBytes in size... To use a larger file, you
   will need to modify some definitions in book.cpp and recompile.

---------------------------------
Notes on the source code version.  
---------------------------------
This software naturally includes some code and ideas gathered 
from other places and these are indicated in the individual 
source code files.  All of these instances remain the copyright 
of the original authors who reserve all rights and their original
licenses still apply.  A specific example is that EXchess supports 
endgame tablebases by Dr. Eugene Nalimov.  Two functions in the file 
"probe.cpp" are modified from the examples given in Dr. Nalimov's 
'tbgen' distribution.  They are for interfacing with Dr. Nalimov's 
endgame tablebase code and are copyrighted by Eugene Nalimov.  The 
functions are NOT part of the GNU Public License release of EXchess' 
source code.  They are only useful if one separately downloads the 
'tbgen' distribution, see "probe.cpp" for instructions.  Other 
examples include...

** interrupt code for winboard on MS-Windows in the function 
   "inter()" in the file main.cpp is from Crafty by Bob Hyatt.  
   Crafy's license is open source with the provision that changes 
   to the code are shared with the community

Compilation and other points:
 - In my experience, EXchess compiles on Mac OS X, Windows (under mingw) 
    and Linux systems, although I only use Mac OS X and Linux on a 
    regular basis.  Earlier versions of EXchess would also compile 
    fine under MSVC (via setting #define UNIX 0, and 
    #define MSVC 1 in define.h), but I haven't tried it in several years.

 - The file "EXchess.cc" is provided which includes all the necessary 
   source to compile EXchess.  On my Mac, I use the following 
   command to compile the console version...

   g++ -o EXchess_console src/EXchess.cc -O3 -pthread

   Your command line and compiler options may be different.

 - The included opening book (main_bk.dat) should work on 
   most systems, but if not, you will have to build your own opening book.  
   You can do this starting from a pgn game collection file using the 
   "build" command in the console version, see above.  Don't bother with 
   a 'starting book' unless you know what you are doing.  






