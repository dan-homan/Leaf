# 
# Perl script to compile and name console versions of EXchess
# during development
#

$date = `date "+%Y_%m_%d"`;
chomp($date);

if($ARGV[0]) {
    $vers = $ARGV[0];
} else {
    $vers = $date;
}
$filename = "EXchess_v" . $vers;

$extra_arg = "";
for my $i (1..$#ARGV) {
    $extra_arg .= " -D $ARGV[$i]";
}

if(-e "./$filename") { 
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

print "Compiling $filename...\n";
$compile = "g++ -o $filename ../src/EXchess.cc -O3 -D VERS=$verstring -D TABLEBASES=1 $extra_arg -pthread";
#$compile = "g++-mp-11 -o $filename ../src/EXchess.cc -O3 -D VERS=$verstring -D TABLEBASES=1 $extra_arg -pthread";
print "$compile\n";
$temp = `$compile`;

