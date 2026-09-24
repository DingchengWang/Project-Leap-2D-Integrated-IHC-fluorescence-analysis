#!/usr/bin/perl
use strict;
use warnings;
use JSON::PP;
use Digest::SHA;
use Encode qw(encode);
use Fcntl qw(:DEFAULT :flock);
use File::Path qw(make_path remove_tree);
use File::Temp qw(tempdir tempfile);
use File::Copy qw(copy);
use IO::Handle;

# This helper runs only after Repair.command verifies it against the published
# same-version manifest. Program recovery needs no project Python interpreter.
my %arg;
while (@ARGV) { my $key = shift @ARGV; $arg{$key} = shift @ARGV; }
my ($project, $source, $manifest_path, $work, $support) = @arg{qw(--project --source --manifest --work --support)};
my $version = '1.0.1';
my $journal_dir = "$support/Source Repair State v$version";
my $state_path = "$support/installation_state.json";
my ($manifest, $journal);
$^F = 9;
$| = 1;
sub stop { die "REPAIR STOPPED SAFELY: $_[0]\n" }
sub path_for { return $_[0] . '/' . encode('UTF-8', $_[1]); }
sub read_bytes {
    my ($p) = @_;
    stop("Expected a regular file: $p") unless -f $p && !-l $p;
    open(my $f, '<:raw', $p) or stop("Cannot read $p");
    local $/; return <$f>;
}
sub sha { return Digest::SHA::sha256_hex(read_bytes($_[0])); }
sub read_json {
    my $value = eval { JSON::PP->new->utf8->decode(read_bytes($_[0])) };
    stop("Invalid recovery record: $_[0]") if $@ || ref($value) ne 'HASH';
    return $value;
}
sub atomic_bytes {
    my ($p, $bytes, $mode) = @_;
    stop("Refusing a symbolic-link destination: $p") if -l $p;
    my $parent = $p; $parent =~ s{/[^/]+$}{};
    my ($f, $tmp) = tempfile('.source-repair-XXXXXXXX', DIR => $parent, UNLINK => 0);
    binmode $f;
    print {$f} $bytes or stop("Cannot stage $p");
    $f->flush && $f->sync or stop("Cannot sync $p");
    close($f) or stop("Cannot close $p");
    chmod($mode, $tmp) or stop("Cannot set permissions for $p");
    rename($tmp, $p) or stop("Cannot atomically replace $p");
}
sub write_journal {
    atomic_bytes("$journal_dir/journal.json", JSON::PP->new->utf8->canonical->pretty->encode($journal), 0600);
}
sub valid_relative {
    my ($p) = @_;
    return 0 if !defined($p) || ref($p) || $p =~ /[\x00-\x1f\x7f\\]/ || $p =~ m{(?:^/|//|(?:^|/)\.{1,2}(?:/|$)|/$)};
    return 0 if $p =~ m{\A(?:Sample Image|Result|Analysis Package/Run State)(?:/|\z)}i;
    return $p =~ m{\AAnalysis Package/project_leap_2d/[^/].*} ||
        $p =~ m{\A(?:Repair\.command|Run Analysis\.command|Analysis Package/VERSION|Manual Command\.txt|README/README (?:EN|CN)\.md)\z};
}
sub checked_parent {
    my ($root, $rel, $create) = @_;
    stop('Unsafe program path in recovery record.') unless valid_relative($rel) && exists($manifest->{files}->{$rel});
    my @parts = split '/', $rel; pop @parts;
    my $p = $root;
    for my $part (@parts) {
        $p = path_for($p, $part);
        stop("A program parent is not a plain directory: $p") if -l $p || (-e $p && !-d $p);
        mkdir($p, 0755) or stop("Cannot create program directory: $p") if $create && !-d $p;
    }
    return path_for($root, $rel);
}
sub matches {
    my ($root, $rel) = @_;
    my $p = checked_parent($root, $rel, 0);
    my $e = $manifest->{files}->{$rel};
    return 0 if -l $p || !-f $p;
    my @s = stat($p);
    return $s[7] == $e->{size} && ($s[2] & 0777) == $e->{mode} && sha($p) eq $e->{sha256};
}
sub verify_lock {
    my ($fd, $p) = @_;
    stop("The inherited lock path is unsafe: $p") if -l $p || !-f $p;
    open(my $f, '+<&=', $fd) or stop("The inherited lock descriptor $fd is unavailable.");
    my @a = stat($f); my @b = stat($p);
    stop('The inherited maintenance lock belongs to a different path.') unless @a && @b && $a[0] == $b[0] && $a[1] == $b[1];
    flock($f, LOCK_EX | LOCK_NB) or stop('The inherited maintenance lock is unavailable.');
    return $f;
}
sub original_state {
    return undef unless $journal->{state_existed};
    my $p = "$journal_dir/installation_state.original";
    stop('The saved installation state failed its checksum.') unless sha($p) eq ($journal->{state_sha256} // '');
    return read_bytes($p);
}
sub mark_repairing {
    my $bytes = original_state();
    return unless defined $bytes;
    my $state = eval { JSON::PP->new->utf8->decode($bytes) };
    return unless ref($state) eq 'HASH' && ($state->{status} // '') eq 'ready';
    $state->{status} = 'repairing';
    atomic_bytes($state_path, JSON::PP->new->utf8->canonical->pretty->encode($state), $journal->{state_mode});
}
sub restore_installation_state {
    my $bytes = original_state();
    atomic_bytes($state_path, $bytes, $journal->{state_mode}) if defined $bytes;
}
sub backup_entry {
    my ($dir, $rel, $index) = @_;
    my $p = checked_parent($project, $rel, 0);
    stop("The damaged program file is not a regular file: $rel") if -l $p || (-e $p && !-f $p);
    my $e = {relative => $rel, existed => (-f $p ? JSON::PP::true : JSON::PP::false), backup => "file-$index"};
    if ($e->{existed}) {
        $e->{mode} = (stat($p))[2] & 0777;
        my $bytes = read_bytes($p);
        $e->{sha256} = Digest::SHA::sha256_hex($bytes);
        atomic_bytes("$dir/$e->{backup}", $bytes, 0600);
    }
    return $e;
}
sub validate_journal {
    stop('A source recovery record belongs to another package or has an unsupported format.') unless
        ($journal->{schema_version} // '') eq '1' && ($journal->{version} // '') eq $version &&
        ($journal->{project} // '') eq $project && ($journal->{phase} // '') =~ /\A(?:prepared|committed)\z/ &&
        ref($journal->{entries}) eq 'ARRAY';
    my %seen;
    for my $e (@{$journal->{entries}}) {
        stop('Invalid source recovery file record.') unless ref($e) eq 'HASH' &&
            ($e->{backup} // '') =~ /\Afile-\d+\z/ && !$seen{$e->{relative}}++;
        checked_parent($project, $e->{relative}, 0);
        if ($e->{existed}) {
            stop('A source recovery backup is invalid.') unless !ref($e->{mode}) &&
                ($e->{mode} // '') =~ /\A\d+\z/ && $e->{mode} <= 0777 &&
                sha("$journal_dir/$e->{backup}") eq ($e->{sha256} // '');
        }
    }
    if ($journal->{state_existed}) {
        stop('Invalid saved installation-state permissions.') unless !ref($journal->{state_mode}) &&
            ($journal->{state_mode} // '') =~ /\A\d+\z/ && $journal->{state_mode} <= 0777;
    }
    original_state();
}
sub rollback_program {
    # Every original file was backed up before publication of this journal.
    # Replaying all entries is safe even if interruption preceded a replacement.
    print "Restoring the pre-repair program files from the recovery record...\n";
    for my $e (reverse @{$journal->{entries}}) {
        my $p = checked_parent($project, $e->{relative}, 1);
        if ($e->{existed}) {
            my $backup = "$journal_dir/$e->{backup}";
            stop('A source recovery backup changed before rollback.') unless sha($backup) eq $e->{sha256};
            atomic_bytes($p, read_bytes($backup), $e->{mode});
        } elsif (-e $p || -l $p) {
            stop("Cannot remove an unexpected recovery destination: $p") if -l $p || !-f $p;
            unlink($p) or stop("Cannot roll back newly created file: $p");
        }
    }
    # The original files may already have been damaged. Keep 'repairing' and
    # the journal until a later attempt verifies the complete published set.
    mark_repairing();
}
sub finish_committed {
    restore_installation_state();
    # Detach the complete record atomically before deleting its contents. A
    # kill during deletion cannot leave a half-deleted active recovery record.
    my $completed = tempdir('Completed Source Repair.XXXXXXXX', DIR => $support);
    rename($journal_dir, $completed) or stop('The completed source recovery record could not be retired.');
    $journal = undef;
    remove_tree($completed, {safe => 1, error => \my $errors});
    stop('The retired source recovery files could not be cleared.') if @$errors;
}

my $environment_lock;
my $workspace_lock;
my $success = eval {
    for my $p ($project, $manifest_path, $work, $support) { stop('Missing or invalid recovery arguments.') unless defined($p) && $p =~ m{^/}; }
    stop('The recovery project, Support, or staging directory is unsafe.') if -l $project || !-d $project || -l $support || !-d $support || -l $work || !-d $work;
    stop('The inherited environment lock marker is invalid.') unless
        ($ENV{PROJECT_LEAP_REPAIR_LOCK_HELD} // '') eq "$support/Environment Usage Lock";
    $environment_lock = verify_lock(9, "$support/Environment Usage Lock");
    $workspace_lock = verify_lock(8, "$project/Analysis Package/Run State/locks/workspace.lock");
    $manifest = read_json($manifest_path);
    stop('The recovery manifest does not identify this version.') unless
        ($manifest->{schema_version} // '') eq '1' && ($manifest->{version} // '') eq $version &&
        ($manifest->{package_name} // '') eq 'Project Leap 2D V1.0.1' && ref($manifest->{files}) eq 'HASH';
    stop('Unsafe program path in recovery manifest.') if grep { !valid_relative($_) } keys %{$manifest->{files}};
    stop('The source recovery record path is unsafe.') if -l $journal_dir || (-e $journal_dir && !-d $journal_dir);
    if (-d $journal_dir) {
        $journal = read_json("$journal_dir/journal.json");
        validate_journal();
        if ($journal->{phase} eq 'committed') {
            stop('A completed source repair no longer matches the published program.') if grep { !matches($project, $_) } keys %{$manifest->{files}};
            finish_committed();
        } else {
            rollback_program();
        }
    }
    my @damaged = grep { !matches($project, $_) } sort keys %{$manifest->{files}};
    if (@damaged) {
        stop('Verified recovery source is unavailable; run Repair.command again.') unless defined($source) && $source ne '' && -d $source && !-l $source;
        for my $rel (@damaged) { stop("The staged recovery source is invalid: $rel") unless matches($source, $rel); }
        if (!$journal) {
            stop('The installation state is not a plain file.') if -l $state_path || (-e $state_path && !-f $state_path);
            my $stage = tempdir('Source Repair Preparation.XXXXXXXX', DIR => $support);
            $journal = {schema_version => 1, version => $version, project => $project,
                phase => 'prepared', entries => [], state_existed => (-f $state_path ? JSON::PP::true : JSON::PP::false)};
            if ($journal->{state_existed}) {
                my $bytes = read_bytes($state_path);
                my $prior = eval { JSON::PP->new->utf8->decode($bytes) };
                stop('The installation is marked repairing but has no source recovery record.') if ref($prior) eq 'HASH' && ($prior->{status} // '') eq 'repairing';
                $journal->{state_mode} = (stat($state_path))[2] & 0777;
                $journal->{state_sha256} = Digest::SHA::sha256_hex($bytes);
                atomic_bytes("$stage/installation_state.original", $bytes, 0600);
            }
            for my $rel (@damaged) { push @{$journal->{entries}}, backup_entry($stage, $rel, scalar @{$journal->{entries}}); }
            atomic_bytes("$stage/journal.json", JSON::PP->new->utf8->canonical->pretty->encode($journal), 0600);
            rename($stage, $journal_dir) or stop('Cannot publish the source recovery record.');
        } else {
            my %recorded = map { $_->{relative} => 1 } @{$journal->{entries}};
            for my $rel (@damaged) {
                next if $recorded{$rel};
                push @{$journal->{entries}}, backup_entry($journal_dir, $rel, scalar @{$journal->{entries}});
            }
            write_journal();
        }
        mark_repairing();
        local $SIG{INT} = sub { die "REPAIR STOPPED SAFELY: Repair interrupted.\n" };
        local $SIG{TERM} = sub { die "REPAIR STOPPED SAFELY: Repair interrupted.\n" };
        local $SIG{HUP} = sub { die "REPAIR STOPPED SAFELY: Repair interrupted.\n" };
        for my $rel (@damaged) {
            my $dest = checked_parent($project, $rel, 1);
            atomic_bytes($dest, read_bytes(path_for($source, $rel)), $manifest->{files}->{$rel}->{mode});
            print "Recovered program file: $rel\n";
        }
        stop('Recovered program files failed final verification.') if grep { !matches($project, $_) } keys %{$manifest->{files}};
        $journal->{phase} = 'committed';
        write_journal();
        finish_committed();
    } elsif ($journal) {
        $journal->{phase} = 'committed';
        write_journal();
        finish_committed();
    }
    print "Program files match the published version. Checking the managed environment...\n";
    remove_tree($work, {safe => 1, error => \my $errors});
    stop('The temporary recovery directory could not be cleared.') if @$errors;
    exec('/bin/zsh', "$project/Analysis Package/project_leap_2d/maintenance/manage_environment.command")
        or stop('Cannot start environment maintenance.');
};
if (!$success) {
    my $error = $@ || "REPAIR STOPPED SAFELY: Unknown program recovery failure.\n";
    if ($journal && -d $journal_dir && ($journal->{project} // '') eq ($project // '') && ($journal->{phase} // '') eq 'prepared') {
        my $rolled_back = eval { validate_journal(); rollback_program(); 1 };
        $error .= "Automatic rollback could not finish: $@" unless $rolled_back;
        $error .= "The recovery record was retained. Run Repair.command again before analysis.\n";
    }
    remove_tree($work, {safe => 1}) if defined($work) && -d $work && !-l $work;
    print STDERR $error;
    exit 1;
}
