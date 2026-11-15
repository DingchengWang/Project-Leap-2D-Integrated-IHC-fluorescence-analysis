package org.projectleap.desktop;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Properties;
import java.util.prefs.AbstractPreferences;
import java.util.prefs.BackingStoreException;
import java.util.prefs.Preferences;
import java.util.prefs.PreferencesFactory;

/** Per-task Java preferences, avoiding macOS CFPreferences outside the task. */
public final class TaskPreferencesFactory implements PreferencesFactory {
    private static final Store USER = new Store(null, "", root("java.util.prefs.userRoot"));
    private static final Store SYSTEM = new Store(null, "", root("java.util.prefs.systemRoot"));

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                USER.flush();
                SYSTEM.flush();
            } catch (BackingStoreException error) {
                System.err.println("Cannot flush task Java preferences: " + error.getMessage());
            }
        }, "project-leap-task-preferences"));
    }

    private static Path root(String property) {
        String value = System.getProperty(property);
        if (value == null || !Paths.get(value).isAbsolute()) {
            throw new IllegalStateException("Task preferences require " + property);
        }
        return Paths.get(value);
    }

    public Preferences userRoot() { return USER; }
    public Preferences systemRoot() { return SYSTEM; }

    private static final class Store extends AbstractPreferences {
        private final Path directory;
        private final Properties values = new Properties();

        Store(Store parent, String name, Path directory) {
            super(parent, name);
            this.directory = directory;
            this.newNode = !Files.exists(directory.resolve("values.properties"));
            try {
                load();
            } catch (IOException error) {
                throw new IllegalStateException("Cannot read task Java preferences", error);
            }
        }

        private void load() throws IOException {
            values.clear();
            Path file = directory.resolve("values.properties");
            if (Files.isRegularFile(file)) {
                try (InputStream stream = Files.newInputStream(file)) {
                    values.load(stream);
                }
            }
        }

        protected void putSpi(String key, String value) { values.setProperty(key, value); }
        protected String getSpi(String key) { return values.getProperty(key); }
        protected void removeSpi(String key) { values.remove(key); }
        protected String[] keysSpi() { return values.stringPropertyNames().toArray(new String[0]); }

        protected String[] childrenNamesSpi() throws BackingStoreException {
            List<String> names = new ArrayList<>();
            if (!Files.isDirectory(directory)) return new String[0];
            try (java.nio.file.DirectoryStream<Path> children = Files.newDirectoryStream(directory)) {
                for (Path child : children) {
                    if (Files.isDirectory(child)) {
                        names.add(new String(Base64.getUrlDecoder().decode(child.getFileName().toString()), StandardCharsets.UTF_8));
                    }
                }
            } catch (IOException | IllegalArgumentException error) {
                throw new BackingStoreException(error);
            }
            return names.toArray(new String[0]);
        }

        protected AbstractPreferences childSpi(String name) {
            String encoded = Base64.getUrlEncoder().withoutPadding().encodeToString(name.getBytes(StandardCharsets.UTF_8));
            return new Store(this, name, directory.resolve(encoded));
        }

        protected void removeNodeSpi() throws BackingStoreException {
            try {
                Files.deleteIfExists(directory.resolve("values.properties"));
                Files.deleteIfExists(directory);
            } catch (IOException error) {
                throw new BackingStoreException(error);
            }
        }

        protected void syncSpi() throws BackingStoreException {
            // Preserve this process's pending updates before refreshing them.
            flushSpi();
            try {
                load();
            } catch (IOException error) {
                throw new BackingStoreException(error);
            }
        }

        protected void flushSpi() throws BackingStoreException {
            if (isRemoved()) return;
            try {
                Files.createDirectories(directory);
                Path temporary = Files.createTempFile(directory, "values-", ".tmp");
                try (OutputStream stream = Files.newOutputStream(temporary)) {
                    values.store(stream, "Project Leap task Java preferences");
                }
                Files.move(temporary, directory.resolve("values.properties"),
                           StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException error) {
                throw new BackingStoreException(error);
            }
        }
    }
}
