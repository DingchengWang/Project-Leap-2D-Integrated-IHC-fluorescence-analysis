#!/bin/sh
set -eu
if test "$#" -ne 0; then
  printf '%s\n' 'Usage: ./Repair.command' >&2
  exit 64
fi
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec /usr/bin/perl - "$PROJECT_DIR" <<'PERL'
use strict;
use warnings;
use JSON::PP;
use Digest::SHA;
use Encode qw(encode decode FB_CROAK);
use Fcntl qw(:DEFAULT :flock);
use POSIX qw(dup2);
use File::Path qw(make_path remove_tree);
use File::Temp qw(tempdir);
use File::Copy qw(copy);
use IO::Uncompress::Unzip qw($UnzipError);

my $project = shift @ARGV;
my $support = $ENV{PROJECT_LEAP_SUPPORT_DIR} || "$ENV{HOME}/Applications/Project Leap 2D Support";
my $repo = 'DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis';
my $version = '1.0.1';
my $package = 'Project Leap 2D V1.0.1';
my $tag = "v$version";
my $base = "Project-Leap-2D-V$version";
my $helper = 'Analysis Package/project_leap_2d/maintenance/repair_package.pl';
my $work;
my $download_pid;
$^F = 9; # Both kernel-held locks survive exec into the verified helper.
$| = 1;
for my $signal (qw(INT TERM HUP)) {
    $SIG{$signal} = sub {
        if (defined $download_pid) {
            kill 'TERM', $download_pid;
            waitpid($download_pid, 0);
            undef $download_pid;
        }
        die "REPAIR STOPPED SAFELY: Repair interrupted. No program files were changed.\n";
    };
}
sub stop { die "REPAIR STOPPED SAFELY: $_[0]\n" }
sub plain_dir {
    my ($p) = @_;
    stop("A required directory is a symbolic link: $p") if -l $p;
    stop("A required directory is not a directory: $p") if -e $p && !-d $p;
    make_path($p, {mode => 0700}) unless -d $p;
}
sub lock_fd {
    my ($p, $fd) = @_;
    stop("The lock is not a regular file: $p") if -l $p || (-e $p && !-f $p);
    sysopen(my $fh, $p, O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW, 0600)
        or stop("Cannot open lock: $p");
    flock($fh, LOCK_EX | LOCK_NB) or stop('Another analysis, installation, or repair is running.');
    dup2(fileno($fh), $fd) >= 0 or stop('Cannot retain the maintenance lock.');
    # Keep the original handle too, including the case that open returned fd 8/9.
    return $fh;
}
sub file_path { return $_[0] . '/' . encode('UTF-8', $_[1]); }
sub digest {
    my ($p) = @_;
    open(my $f, '<:raw', $p) or stop("Cannot read $p");
    return Digest::SHA->new(256)->addfile($f)->hexdigest;
}
sub json_file {
    my ($p) = @_;
    open(my $f, '<:raw', $p) or stop("Cannot read $p");
    local $/;
    my $value = eval { JSON::PP->new->utf8->decode(<$f>) };
    stop("Invalid JSON received: $p") if $@ || ref($value) ne 'HASH';
    return $value;
}
sub download {
    my ($url, $dest) = @_;
    $download_pid = fork();
    stop('Cannot start the release download.') unless defined $download_pid;
    if ($download_pid == 0) {
        $SIG{$_} = 'DEFAULT' for qw(INT TERM HUP);
        # Only the bootstrap owns these locks. A cancelled download must not
        # leave an orphan network process holding the installation open.
        POSIX::close($_) for 3 .. 9;
        exec('/usr/bin/curl', '--fail', '--location', '--silent', '--show-error',
        '--proto', '=https', '--proto-redir', '=https', '--tlsv1.2',
        '--connect-timeout', '20', '--max-time', '300', '--output', $dest, $url)
            or POSIX::_exit(127);
    }
    waitpid($download_pid, 0);
    my $download_status = $?;
    undef $download_pid;
    $download_status == 0
        or stop("The fixed release $tag could not be downloaded. It may not yet be published, or GitHub is unavailable. No program files were changed.");
}
sub valid_relative {
    my ($p) = @_;
    return 0 if !defined($p) || ref($p) || $p =~ /[\x00-\x1f\x7f\\]/ || $p =~ m{(?:^/|//|(?:^|/)\.{1,2}(?:/|$)|/$)};
    return 0 if $p =~ m{\A(?:Sample Image|Result|Analysis Package/Run State)(?:/|\z)}i;
    return $p =~ m{\AAnalysis Package/project_leap_2d/[^/].*} ||
        $p =~ m{\A(?:Repair\.command|Run Analysis\.command|Analysis Package/VERSION|Manual Command\.txt|README/README (?:EN|CN)\.md)\z};
}
sub safe_ancestors {
    my ($root, $relative) = @_;
    my @parts = split '/', $relative;
    pop @parts;
    my $p = $root;
    for my $part (@parts) {
        $p = file_path($p, $part);
        stop("A program parent is not a plain directory: $p") if -l $p || (-e $p && !-d $p);
    }
}
sub matches {
    my ($root, $rel, $entry) = @_;
    safe_ancestors($root, $rel);
    my $p = file_path($root, $rel);
    return 0 if -l $p || !-f $p;
    my @s = stat($p);
    return $s[7] == $entry->{size} && ($s[2] & 0777) == $entry->{mode} && digest($p) eq $entry->{sha256};
}

my $ok = eval {
    stop('The Support directory must be an absolute path.') unless $support =~ m{^/};
    plain_dir($support);
    plain_dir("$project/Analysis Package");
    plain_dir("$project/Analysis Package/Run State");
    plain_dir("$project/Analysis Package/Run State/locks");
    my $environment_lock = lock_fd("$support/Environment Usage Lock", 9);
    my $workspace_lock = lock_fd("$project/Analysis Package/Run State/locks/workspace.lock", 8);
    $ENV{PROJECT_LEAP_REPAIR_LOCK_HELD} = "$support/Environment Usage Lock";
    $work = tempdir('Project-Leap-Repair-XXXXXXXX', TMPDIR => 1);
    print "Checking the published Project Leap 2D $tag program files...\n";
    download("https://api.github.com/repos/$repo/releases/tags/$tag", "$work/release.json");
    my $release = json_file("$work/release.json");
    stop('GitHub did not return the requested published version.') unless
        ($release->{tag_name} // '') eq $tag && exists($release->{draft}) && !$release->{draft} && ref($release->{assets}) eq 'ARRAY';
    my %assets;
    for my $name ("$base.manifest.json", "$base.zip") {
        my @found = grep { ref($_) eq 'HASH' && ($_->{name} // '') eq $name } @{$release->{assets}};
        stop("The fixed release asset is missing or ambiguous: $name") unless @found == 1;
        my $a = $found[0];
        stop("The release asset is incomplete: $name") unless ($a->{state} // '') eq 'uploaded';
        stop("A valid SHA-256 digest is unavailable for $name") unless ($a->{digest} // '') =~ /\Asha256:([0-9a-f]{64})\z/;
        $a->{expected_sha256} = $1;
        stop("The release download URL is not the fixed GitHub asset: $name") unless
            ($a->{browser_download_url} // '') eq "https://github.com/$repo/releases/download/$tag/$name";
        $assets{$name} = $a;
    }
    my $manifest_path = "$work/manifest.json";
    download($assets{"$base.manifest.json"}->{browser_download_url}, $manifest_path);
    stop('The downloaded program manifest failed its SHA-256 check.') unless
        digest($manifest_path) eq $assets{"$base.manifest.json"}->{expected_sha256};
    my $manifest = json_file($manifest_path);
    stop('The program manifest does not describe this fixed version.') unless
        ($manifest->{schema_version} // '') eq '1' && ($manifest->{version} // '') eq $version &&
        ($manifest->{package_name} // '') eq $package && ref($manifest->{files}) eq 'HASH';
    my $files = $manifest->{files};
    for my $required ('Repair.command', 'Run Analysis.command', 'Analysis Package/VERSION', $helper, 'Analysis Package/project_leap_2d/maintenance/manage_environment.command') {
        stop("The manifest omits a required program file: $required") unless exists $files->{$required};
    }
    my %folded;
    for my $rel (keys %$files) {
        my $e = $files->{$rel};
        stop("Unsafe or conflicting program path in manifest: $rel") unless valid_relative($rel) && !$folded{lc($rel)}++;
        stop("Invalid program file metadata: $rel") unless ref($e) eq 'HASH' &&
            ($e->{sha256} // '') =~ /\A[0-9a-f]{64}\z/ && !ref($e->{size}) && ($e->{size} // '') =~ /\A\d+\z/ &&
            !ref($e->{mode}) && ($e->{mode} // '') =~ /\A\d+\z/ && $e->{mode} <= 0777;
    }
    my @damaged = grep { !matches($project, $_, $files->{$_}) } sort keys %$files;
    my $source = '';
    if (@damaged || -d "$support/Source Repair State v$version") {
        print scalar(@damaged), " program file(s) differ, or an interrupted recovery needs completion; downloading this same version.\n";
        my $zip_path = "$work/package.zip";
        download($assets{"$base.zip"}->{browser_download_url}, $zip_path);
        stop('The downloaded program ZIP failed its SHA-256 check.') unless digest($zip_path) eq $assets{"$base.zip"}->{expected_sha256};
        # zipinfo reads Unix file types from the central directory. The stream
        # reader below materializes only regular manifest-listed file bytes.
        open(my $listing, '-|', '/usr/bin/zipinfo', '-l', $zip_path) or stop('Cannot inspect ZIP file types.');
        while (my $line = <$listing>) {
            stop('The program ZIP contains a symbolic link or special file.') if $line =~ /^[bclps][rwxstST?\-]{9}\s/;
        }
        close($listing) or stop('The program ZIP directory is invalid.');
        $source = "$work/source";
        mkdir($source, 0700) or stop('Cannot create the verified source staging directory.');
        my $zip = IO::Uncompress::Unzip->new($zip_path, Strict => 1) or stop("Cannot read ZIP: $UnzipError");
        my $prefix = "$package Distribution/$package/";
        my %seen;
        while (1) {
            my $name = $zip->getHeaderInfo->{Name};
            $name = decode('UTF-8', $name, FB_CROAK) unless utf8::is_utf8($name);
            stop('The program ZIP contains an unsafe or unexpected member path.') if
                $name =~ /[\x00-\x1f\x7f\\]/ || $name =~ m{(?:^/|//|(?:^|/)\.{1,2}(?:/|$))} ||
                index($name, "$package Distribution/") != 0;
            if (index($name, $prefix) == 0) {
                my $rel = substr($name, length($prefix));
                if (exists $files->{$rel}) {
                    stop("A program file occurs more than once in the ZIP: $rel") if $seen{$rel}++;
                    my $p = file_path($source, $rel);
                    my $parent = $p; $parent =~ s{/[^/]+$}{};
                    make_path($parent, {mode => 0700});
                    open(my $out, '>:raw', $p) or stop("Cannot stage $rel");
                    my $size = 0;
                    while (1) {
                        my $n = $zip->read(my $buffer, 131072);
                        stop("Cannot decompress $rel: $UnzipError") if $n < 0;
                        last if $n == 0;
                        $size += $n;
                        stop("Unexpected decompressed size: $rel") if $size > $files->{$rel}->{size};
                        print {$out} $buffer or stop("Cannot write staged program file: $rel");
                    }
                    close($out) or stop("Cannot finish staged program file: $rel");
                    chmod($files->{$rel}->{mode}, $p) or stop("Cannot set staged program permissions: $rel");
                    stop("A staged program file failed verification: $rel") unless matches($source, $rel, $files->{$rel});
                }
            }
            my $next = $zip->nextStream();
            stop("Cannot finish ZIP: $UnzipError") if $next < 0;
            last if $next == 0;
        }
        stop('The ZIP does not contain every manifest-listed program file.') unless keys(%seen) == keys(%$files);
    }
    my $trusted_helper = "$work/repair_package.pl";
    my $helper_source = file_path($source || $project, $helper);
    copy($helper_source, $trusted_helper) or stop('Cannot prepare the verified recovery helper.');
    stop('The recovery helper failed verification.') unless digest($trusted_helper) eq $files->{$helper}->{sha256};
    exec('/usr/bin/perl', $trusted_helper, '--project', $project, '--source', $source,
        '--manifest', $manifest_path, '--work', $work, '--support', $support)
        or stop('Cannot start the verified recovery helper.');
};
if (!$ok) {
    my $error = $@ || "REPAIR STOPPED SAFELY: Unknown bootstrap failure.\n";
    remove_tree($work) if defined($work) && -d $work;
    print STDERR $error;
    exit 1;
}
PERL
