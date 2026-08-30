#
# Perl script to compile and name console versions of Leaf
# during development, and to produce portable release binaries.
#
#   perl comp.pl <version> [FLAG=VALUE ...]
#
# Build-system arguments (consumed here, never passed to the compiler):
#
#   OVERWRITE        Skip the interactive overwrite prompt
#   NATIVE=1         Tune for THIS machine (fast, non-portable).  Do not use
#                    for anything you intend to distribute
#   CXX=<compiler>   Compiler to invoke (default: g++).  Set this to a cross
#                    compiler, e.g. CXX=x86_64-w64-mingw32-g++
#   WINDOWS=1        Target Windows: appends .exe, defines MINGW=1, links
#                    statically.  Combine with a MinGW CXX= to cross compile
#   STATIC=1         Static-link libstdc++/libgcc (Linux; implied on Windows).
#                    Use for release binaries so they run on older distros
#   MACOS_MIN=<ver>  macOS deployment target (default 11.0, the first macOS
#                    with Apple Silicon).  macOS targets only
#
# Every other FLAG=VALUE is forwarded to the compiler as -D FLAG=VALUE
# (NNUE=1, TDLEAF=1, NNUE_NET=<file>, ... — see docs/TRAINING.md).
#
# Release builds are portable by DEFAULT.  In particular, macOS builds are no
# longer tuned to the build machine, and they carry an explicit deployment
# target: clang otherwise stamps the binary with the build host's OS version,
# which makes it refuse to launch on any older macOS.
#

$date = `date "+%Y_%m_%d"`;
chomp($date);

if($ARGV[0]) {
    $vers = $ARGV[0];
} else {
    $vers = $date;
}
$filename = "Leaf_v" . $vers;

$extra_arg = "";
$overwrite = 0;
$nnue_net_file = "";

# Build-system arguments.  These configure the compiler invocation itself and
# must be consumed here — forwarding them as -D would at best define a dead
# macro and at worst (CXX=) corrupt the command line.
my $native    = 0;
my $windows   = 0;
my $static    = 0;
my $cxx       = "g++";
my $macos_min = "11.0";

for my $i (1..$#ARGV) {
    my $arg = $ARGV[$i];
    # NNUE_NET=filename and NNUE_TDLEAF_BIN=filename need string-literal quoting.
    # Also auto-derive NNUE_TDLEAF_BIN from NNUE_NET when only NNUE_NET is given.
    if ($arg eq "OVERWRITE") {
        $overwrite = 1;
    } elsif ($arg eq "NATIVE=1") {
        $native = 1;
    } elsif ($arg eq "WINDOWS=1") {
        $windows = 1;
    } elsif ($arg eq "STATIC=1") {
        $static = 1;
    } elsif ($arg =~ /^CXX=(.+)$/) {
        $cxx = $1;
    } elsif ($arg =~ /^MACOS_MIN=(.+)$/) {
        $macos_min = $1;
    } elsif ($arg =~ /^NNUE_NET=(.+)$/) {
        my $net = $1;
        (my $tdleaf = $net) =~ s/\.nnue$/.tdleaf.bin/;
        $extra_arg .= " \"-D NNUE_NET=\\\"$net\\\"\"";
        $extra_arg .= " \"-D NNUE_TDLEAF_BIN=\\\"$tdleaf\\\"\"";
        $nnue_net_file = $net;
    } elsif ($arg =~ /^NNUE_TDLEAF_BIN=(.+)$/) {
        my $val = $1;
        $extra_arg .= " \"-D NNUE_TDLEAF_BIN=\\\"$val\\\"\"";
    } else {
        $extra_arg .= " -D $arg";
    }
}

my $os   = `uname -s`; chomp $os;
my $arch = `uname -m`; chomp $arch;
my $is_arm = ($arch =~ /^(arm|aarch)/i);

# Building *on* Windows (MSYS2 / Git Bash) is a Windows target too.  Detected
# here, before the suffix below, so a native Windows build also gets its .exe.
$windows = 1 if $os =~ /MINGW|MSYS|CYGWIN/i;

# Windows executables need the .exe suffix — applied before the overwrite check
# so the prompt tests the file that will actually be written.
$filename .= ".exe" if $windows;

if(-e "./$filename" && !$overwrite) {
    print "File $filename already exists!  Overwrite (y/n)? ";
    $resp = <STDIN>;
    chomp($resp);
    if($resp =~ /n/) {
	print "Quitting without compile...\n";
	print "Try again with a different name specified on command line.\n";
	exit;
    }
}

$verstring = "\\" . "\"" . $vers . "\\" . "\"";

# When NNUE_EMBED=1 is set, derive NNUE_NET_PATH for incbin from the net filename.
# The .nnue file must exist next to the binary (in the run directory).
if ($extra_arg =~ /NNUE_EMBED=1/) {
    my $net = $nnue_net_file || "nn-leaf-260414.nnue";
    my $net_path = (-e $net) ? $net : "../run/$net";
    if (!-e $net_path) {
        use Cwd 'abs_path';
        # Try resolving relative to run directory
        $net_path = $net;
    }
    use Cwd 'abs_path';
    $net_path = abs_path($net_path) if -e $net_path;
    if (!-e $net_path) {
        die "NNUE_EMBED=1 but net file '$net' not found (tried ./ and ../run/)\n";
    }
    $extra_arg .= " \"-D NNUE_NET_PATH=\\\"$net_path\\\"\"";
    print "Embedding NNUE net: $net_path\n";
}

# Resolve the TARGET, which is not necessarily the host: WINDOWS=1 with a MinGW
# CXX= cross-compiles from macOS or Linux, and the flags below must follow the
# target, not `uname`.  ($os/$arch/$is_arm/$windows are set above.)
my $target = $windows          ? "windows"
           : ($os eq "Darwin") ? "macos"
           :                     "linux";

# A Windows target is always x86-64 here, even when cross-compiled from an
# arm64 Mac — so key the ISA flags off the target, not the host CPU.
my $target_is_arm = $is_arm && $target ne "windows";

my @flags;

if ($target eq "macos") {
    # Apple Silicon only (project decision for 1.0).  No -march: measured
    # identical node counts and NPS with and without it on Apple clang, so
    # tuning to the build machine buys nothing and costs portability.
    if ($native) {
        # -march=native is the wrong spelling on arm64 and degrades the target.
        push @flags, $is_arm ? "-mcpu=native" : "-march=native -mtune=native";
    }
    # Without this, clang stamps the binary with the BUILD HOST's OS version
    # and it refuses to launch on anything older.  Verify with:
    #   otool -l <binary> | grep -A3 LC_BUILD_VERSION
    push @flags, "-mmacosx-version-min=$macos_min";
} elsif ($target_is_arm) {
    # ARM Linux.  Not a 1.0 release target, but emitting x86 flags here would
    # simply fail to compile, so handle it rather than assuming x86.
    push @flags, "-mcpu=native" if $native;
} else {
    # x86-64 targets (Linux and Windows).  x86-64-v3 = AVX2, i.e. Intel
    # Haswell (2013+) and AMD Zen (2017+).  Both vendors, neither ancient.
    push @flags, $native ? "-march=native -mtune=native"
                         : "-march=x86-64-v3 -mtune=generic";
    push @flags, "-mpopcnt";
}

if ($target eq "windows") {
    # MINGW=1 selects the Windows console/polling paths in main.cpp and
    # protocol.cpp.  define.h guards it with #ifndef, so this wins.
    push @flags, "-D MINGW=1";
    # One self-contained .exe: no libgcc/libstdc++/winpthread DLLs to ship.
    push @flags, "-static";
} elsif ($target eq "linux" && $static) {
    # Distributable Linux binaries: drop the libstdc++/libgcc version
    # dependency.  glibc itself still binds to the build host's version —
    # check with: objdump -T <binary> | grep GLIBC_ | sort -u
    push @flags, "-static-libstdc++", "-static-libgcc";
}

if ($native && $target ne "macos") {
    print "NOTE: NATIVE=1 tunes for this machine — do not distribute this binary.\n";
}

my $arch_flags = join(" ", @flags);

print "Compiling $filename (target: $target, compiler: $cxx)...\n";
$compile = "$cxx -o $filename ../src/Leaf.cc -O3 $arch_flags -funroll-loops -ffast-math -flto -D VERS=$verstring $extra_arg -pthread -Wno-unused-result";
print "$compile\n";
system($compile) == 0
    or die "\ncomp.pl: COMPILE FAILED (exit " . ($? >> 8) . ") — no binary produced.\n"
         . "Nothing was written to $filename; any file of that name is from an earlier build.\n";
print "Built $filename\n";

