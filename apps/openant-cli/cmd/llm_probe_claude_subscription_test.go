package cmd

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
)

// writeFakeClaudeCLI creates an executable script standing in for
// `claude auth status`, exiting with the given code. Mirrors the
// writeHangScript pattern in internal/python/invoke_test.go.
func writeFakeClaudeCLI(t *testing.T, exitCode int) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake-CLI test uses a POSIX shell script")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "claude")
	script := "#!/bin/sh\nexit " + strconv.Itoa(exitCode) + "\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake claude CLI: %v", err)
	}
	return path
}

func TestProbeClaudeSubscription_LoggedIn(t *testing.T) {
	prev := claudeCLIBinary
	claudeCLIBinary = writeFakeClaudeCLI(t, 0)
	t.Cleanup(func() { claudeCLIBinary = prev })

	if err := probeClaudeSubscription(); err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
}

func TestProbeClaudeSubscription_NotLoggedIn(t *testing.T) {
	prev := claudeCLIBinary
	claudeCLIBinary = writeFakeClaudeCLI(t, 1)
	t.Cleanup(func() { claudeCLIBinary = prev })

	err := probeClaudeSubscription()
	if err == nil {
		t.Fatal("expected an error when the CLI reports not logged in")
	}
}

func TestProbeClaudeSubscription_BinaryNotFound(t *testing.T) {
	prev := claudeCLIBinary
	claudeCLIBinary = filepath.Join(t.TempDir(), "does-not-exist-claude-binary")
	t.Cleanup(func() { claudeCLIBinary = prev })

	err := probeClaudeSubscription()
	if err == nil {
		t.Fatal("expected an error when the CLI binary is missing")
	}
}
