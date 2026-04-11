#
# Perl script to compile and name console versions of Leaf
# during development
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
for my $i (1..$#ARGV) {
    my $arg = $ARGV[$i];
    # NNUE_NET=filename and NNUE_TDLEAF_BIN=filename need string-literal quoting.
    # Also auto-derive NNUE_TDLEAF_BIN from NNUE_NET when only NNUE_NET is given.
    if ($arg eq "OVERWRITE") {
        $overwrite = 1;
    } elsif ($arg =~ /^NNUE_NET=(.+)$/) {
        my $net = $1;
        (my $tdleaf = $net) =~ s/\.nnue$/.tdleaf.bin/;
        $extra_arg .= " \"-D NNUE_NET=\\\"$net\\\"\"";
        $extra_arg .= " \"-D NNUE_TDLEAF_BIN=\\\"$tdleaf\\\"\"";
    } elsif ($arg =~ /^NNUE_TDLEAF_BIN=(.+)$/) {
        my $val = $1;
        $extra_arg .= " \"-D NNUE_TDLEAF_BIN=\\\"$val\\\"\"";
    } else {
        $extra_arg .= " -D $arg";
    }
}

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

my $native = grep { $_ eq "NATIVE=1" } @ARGV;
my $os   = `uname -s`; chomp $os;
my $arch = `uname -m`; chomp $arch;
my $is_arm = ($arch =~ /^(arm|aarch)/i);

my $arch_flags = ($native || $os eq "Darwin")
    ? "-march=native -mtune=native"
    : "-march=x86-64-v3 -mtune=generic";
my $x86_flags = $is_arm ? "" : "-mpopcnt";

print "Compiling $filename...\n";
$compile = "g++ -o $filename ../src/Leaf.cc -O3 $arch_flags $x86_flags -funroll-loops -ffast-math -flto -D VERS=$verstring $extra_arg -pthread -Wno-unused-result";
print "$compile\n";
print "Tablebase support not included, use TABLEBASES=1 to include\n";
$temp = `$compile`;

