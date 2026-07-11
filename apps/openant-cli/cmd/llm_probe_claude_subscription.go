package cmd

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"time"
)

// claudeCLIBinary is the executable used to check the local Claude Code
// CLI's login status. A package var (not a const) so tests can point it
// at a stub script instead of requiring the real `claude` binary on PATH.
var claudeCLIBinary = "claude"

// probeClaudeSubscription checks the two preconditions the
// claude_subscription provider needs: the Claude Code CLI is installed
// and the user is logged in (`claude login`, Pro/Max subscription).
//
// Unlike probeAnthropic/probeOpenAI/probeGoogle, this does NOT validate
// the specific model ID — doing so would mean spending a real turn of
// the user's subscription usage during setup just to catch a typo. A
// bad model ID surfaces at the first `openant scan` instead, via the
// Python adapter's validate() (see
// libs/openant-core/utilities/llm/providers/claude_subscription.py).
func probeClaudeSubscription() error {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// `claude auth status` exits 0 when logged in, 1 otherwise — cheap
	// and well-documented, unlike spending a real completion turn.
	cmd := exec.CommandContext(ctx, claudeCLIBinary, "auth", "status")
	err := cmd.Run()
	if err == nil {
		return nil
	}

	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return fmt.Errorf(
			"`%s auth status` reports not logged in (exit %d) — run `claude login` to authenticate with your Claude Pro/Max subscription",
			claudeCLIBinary, exitErr.ExitCode(),
		)
	}
	return fmt.Errorf(
		"could not run `%s auth status`: %w — is the Claude Code CLI installed and on PATH?",
		claudeCLIBinary, err,
	)
}
