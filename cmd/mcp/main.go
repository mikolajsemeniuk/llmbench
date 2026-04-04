package main

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func main() {
	s := server.NewMCPServer("k8s-tools", "1.0.0")

	s.AddTool(
		mcp.NewTool("get_pods",
			mcp.WithDescription("List pods with status, restarts, age"),
			mcp.WithString("namespace", mcp.Description("Kubernetes namespace"), mcp.DefaultString("default")),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			ns := param(req, "namespace", "default")
			return run(ctx, "kubectl", "get", "pods", "-n", ns, "-o", "wide")
		},
	)

	srv := server.NewStreamableHTTPServer(s)
	log.Println("MCP server on :9090")
	log.Fatal(srv.Start(":9090"))
}

func param(req mcp.CallToolRequest, key, fallback string) string {
	raw, ok := req.GetArguments()[key]
	if !ok {
		return fallback
	}
	s, ok := raw.(string)
	if !ok || s == "" {
		return fallback
	}
	return s
}

func run(ctx context.Context, name string, args ...string) (*mcp.CallToolResult, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	out, err := cmd.CombinedOutput()
	text := strings.TrimSpace(string(out))

	if err != nil {
		// return as tool-level error, not RPC error —
		// so the LLM sees the failure and can reason about it
		msg := fmt.Sprintf("command failed: %v\n%s", err, text)
		return mcp.NewToolResultError(msg), nil
	}

	if len(text) > 8000 {
		text = text[:8000] + "\n... (truncated)"
	}

	return mcp.NewToolResultText(text), nil
}
